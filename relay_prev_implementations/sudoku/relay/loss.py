from typing import Any, List, Optional, cast

import torch

from .streaming_batch import StreamingBatch
from .types import MLMBatch, MLMLossDict, MLMModel
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from xlm.utils.nn import masked_mean, select_random_indices
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def ensure_batch_fixed_for_streaming(
    batch: MLMBatch,
    batch_shape_key: str = "input_ids",
) -> None:
    """Mutate ``batch`` in place: add ``fixed`` (clue mask) if absent.

    ``StreamingBatch`` and the relay loss expect a bool ``fixed`` mask
    (True = never-remask / clue position). If the collator did not set it
    (e.g. plain MLM), default to all-False.
    """
    if isinstance(batch.get("fixed"), torch.Tensor):
        return
    if batch_shape_key not in batch:
        raise KeyError(batch_shape_key)
    ref = batch[batch_shape_key]
    if not isinstance(ref, torch.Tensor):
        raise TypeError(f"{batch_shape_key} must be a torch.Tensor")
    batch["fixed"] = torch.zeros(
        ref.shape,
        dtype=torch.bool,
        device=ref.device,
    )


class MLMLoss(LossFunction[MLMBatch, MLMLossDict]):
    def __init__(
        self,
        loss_on_visible_tokens: bool = False,
        model: Optional[MLMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id_tensor = None

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(  # type: ignore
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        input_ids = batch["input_ids"]
        targets = batch["target_ids"]
        assert targets is not None

        model = cast(MLMModel, self.model)
        logits = model(input_ids)

        ignore = torch.zeros_like(input_ids, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(input_ids != self.mask_token_id_tensor)
        targets[ignore] = -100
        if ignore.all():
            # PyTorch quirk: when every position is ignored, F.cross_entropy returns
            # NaN instead of zero (https://github.com/pytorch/pytorch/issues/70348).
            return {
                "loss": torch.tensor(
                    0.0,
                    device=logits.device,
                    dtype=logits.dtype,
                    requires_grad=True,
                )
            }

        logits_T = logits.transpose(1, 2)
        ce_per_token = torch.nn.functional.cross_entropy(
            logits_T, targets, reduction="none", ignore_index=-100
        )
        ce = masked_mean(ce_per_token.flatten(), ~ignore.flatten(), dim=-1)
        return {"loss": ce}


class RelayBPTTLoss(LossFunction[MLMBatch, MLMLossDict]):
    """Multi-step relay rollout loss with optional truncated BPTT through ``h``.

    The loss maintains a rollout buffer (``StreamingBatch``) of partially unmasked
    examples across training batches. Each call:

      1. evicts finished slots and refills them with fresh examples from ``batch``;
      2. runs ``num_steps`` of the relay rollout on the buffer:
         when ``with_relay=True`` the model takes ``(x_t, h_t)`` and returns
         ``(logits, h_s)``; when ``with_relay=False`` it is a plain
         ``model(x_t)`` MLM forward (rollout-buffer-only baseline);
      3. teacher-forces high-confidence positions, persists the new
         ``input_ids`` and (optionally) ``h_s`` back to the buffer, and updates
         eviction state.

    With ``stop_grad_h_s=True`` the carried ``h_s`` is detached between steps,
    cutting backprop-through-time across the relay edge (Relay-sg ablation).
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Tokenizer] = None,
        predictor: Optional[Any] = None,
        loss_on_visible_tokens: bool = False,
        use_model_confidence: bool = True,
        batch_shape_key: str = "input_ids",
        num_steps: int = 2,
        stop_grad_h_s: bool = False,
        with_relay: bool = True,
        threshold_sampling_sigma: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.predictor = predictor
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.use_model_confidence = use_model_confidence
        self.batch_shape_key = batch_shape_key
        self.num_steps = num_steps
        self.stop_grad_h_s = stop_grad_h_s
        self.with_relay = with_relay
        self.threshold_sampling_sigma = threshold_sampling_sigma
        self.pl_module = None
        self.mask_token_id_tensor = None
        if self.threshold_sampling_sigma < 0:
            raise ValueError(
                f"threshold_sampling_sigma must be >= 0, got {self.threshold_sampling_sigma}."
            )

    def configure(self, pl_module: Harness) -> None:
        self.pl_module = pl_module
        if self.predictor is None:
            self.predictor = pl_module.predictor
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )
        self.streaming_batch = StreamingBatch()
        self.val_streaming_batch = StreamingBatch()

    def __call__(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def _compute_ce(
        self,
        logits: torch.Tensor,
        x_clean: torch.Tensor,
        x_input: torch.Tensor,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        targets = x_clean.clone()
        ignore = torch.zeros_like(x_input, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(x_input != self.mask_token_id_tensor)
        else:
            ignore = ignore.logical_or(fixed)
        targets[ignore] = -100
        ce = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2),
            targets,
            reduction="none",
            ignore_index=-100,
        )
        return masked_mean(ce.flatten(), ~ignore.flatten(), dim=-1)

    def _select_unmask(
        self,
        logits: torch.Tensor,
        masked_mutable: torch.Tensor,
        confidence_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.use_model_confidence:
                return self.predictor.compute_confidence_unmask(
                    logits,
                    masked_mutable,
                    threshold=confidence_threshold,
                )
            num_unmask = (masked_mutable.sum(dim=-1) // 2).clamp(min=1)
            return select_random_indices(
                inp_shape=logits.shape[:2],
                num_unmask=num_unmask,
                select_from_mask=masked_mutable,
                selection_score=None,
                selection_mode="sample",
            )

    def _sample_confidence_threshold(
        self,
        *,
        device: torch.device,
        training: bool,
    ) -> Optional[float]:
        if not self.use_model_confidence:
            return None
        mu = getattr(self.predictor, "threshold", None)
        if mu is None:
            return None
        mu = float(mu)
        if (not training) or self.threshold_sampling_sigma == 0:
            return mu
        # .item() is unfriendly to torch.compile but only fires on training steps
        # and is needed because compute_confidence_unmask expects a Python float.
        sampled = mu + self.threshold_sampling_sigma * torch.randn(
            (), device=device
        ).item()
        return max(0.0, sampled)

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        is_training = bool(self.pl_module.training) if self.pl_module is not None else True

        ensure_batch_fixed_for_streaming(batch, self.batch_shape_key)
        if not is_training:
            # Validation runs an independent rollout per call: reset the buffer so
            # evict_and_fill re-initializes from the current batch and we sidestep
            # the training buffer's fixed-capacity constraint.
            self.val_streaming_batch.reset()
        sb = self.val_streaming_batch if not is_training else self.streaming_batch
        evicted_candidates: Optional[torch.Tensor] = None
        if sb.storage is not None:
            evicted_candidates = sb.ready_to_evict.nonzero(as_tuple=True)[0]

        evicted_count = sb.evict_and_fill(
            batch,
            mask_token_id=self.mask_token_id_tensor,
            batch_shape_key=self.batch_shape_key,
        )

        device = sb.storage["input_ids"].device
        capacity, seq_len = sb.capacity, sb.seq_len

        if self.with_relay:
            d_model = self.model.d_model
            # Persist h_s in the buffer; zero it on freshly evicted slots so a new
            # example never inherits the previous occupant's relay state.
            if "h_s" not in sb.storage:
                sb.storage["h_s"] = torch.zeros(
                    capacity, seq_len, d_model,
                    device=device,
                    dtype=torch.float32,
                )
            if evicted_count > 0 and evicted_candidates is not None:
                evicted_slots = evicted_candidates[:evicted_count]
                sb.storage["h_s"][evicted_slots] = 0.0
        else:
            sb.storage.pop("h_s", None)

        x = sb.storage["input_ids"].clone()
        x_clean = sb.storage["target_ids"]
        fixed = sb.storage["fixed"]
        # Clone so the live buffer never participates in autograd; we copy back below.
        h: Optional[torch.Tensor] = (
            sb.storage["h_s"].clone() if self.with_relay else None
        )

        loss_terms: List[torch.Tensor] = []
        last_unmask: Optional[torch.Tensor] = None

        for t in range(self.num_steps):
            masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
            has_any_mask = masked_mutable.any()

            if self.with_relay:
                assert h is not None
                logits, h_s = self.model(x, h)
            else:
                logits = self.model(x)
                h_s = None

            L_t = self._compute_ce(logits, x_clean, x, fixed)
            # Force a gradient path to model parameters even when no positions
            # are masked on this rank — keeps DDP bucket shapes in sync.
            anchor = 0.0 * logits.sum()
            L_t = L_t + anchor
            loss_terms.append(L_t)

            sampled_threshold = self._sample_confidence_threshold(
                device=logits.device,
                training=is_training,
            )
            threshold_t = (
                torch.tensor(sampled_threshold, device=logits.device, dtype=logits.dtype)
                if sampled_threshold is not None
                else None
            )
            unmask = self._select_unmask(
                logits.detach(),
                masked_mutable,
                confidence_threshold=threshold_t,
            )
            unmask = unmask & masked_mutable
            last_unmask = unmask

            x = x.clone()
            with torch.no_grad():
                x[unmask] = x_clean[unmask]

            if self.with_relay:
                assert h is not None and h_s is not None
                # When nothing is masked we keep h unchanged (matches inference behaviour).
                h_old = h
                h_next = h_s.detach() if self.stop_grad_h_s else h_s
                h = torch.where(has_any_mask, h_next, h_old)

        total_loss = torch.stack(loss_terms).sum()
        # Persist state for the next call. Replace tensor references rather than
        # copy_ in place, so no aliased storage leaks into the autograd graph.
        with torch.no_grad():
            sb.storage["input_ids"] = x.detach().clone()
            if self.with_relay:
                assert h is not None
                sb.storage["h_s"] = h.detach().clone()
        sb.update_after_unmask(last_unmask, x_clean, self.mask_token_id_tensor)

        return {"loss": total_loss}
