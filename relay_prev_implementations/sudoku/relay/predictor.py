from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
    Literal,
)
from functools import partial
import torch
from jaxtyping import Bool, Integer, Float
from xlm.datamodule import Tokenizer
from torch import Tensor as TT
from .types import (
    MLMBatch,
    MLMPredictionDict,
    MLMModel,
    MLMSeq2SeqPredictionBatch,
)
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
    select_random_indices,
)
from xlm.utils.rank_zero import RankedLogger
import time

logger = RankedLogger(__name__, rank_zero_only=True)

MLMStepResults = Dict[str, Any]


class ConfidenceBasedPredictor(torch.nn.Module, Predictor[MLMBatch, MLMPredictionDict]):
    """Base predictor for MLM. Stochastically selects positions to unmask based on max_steps."""

    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[MLMModel] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        skip_special_tokens: bool = True,
        confidence: Optional[Literal["prob_diff", "entropy", "top_prob",]] = None,
        threshold: Optional[float] = 0.15,
        with_relay: bool = False,
        log_rollout_diagnostics: bool = False,
        log_rollout_diagnostics_max_steps: int = 8,
        capture_trajectory: bool = False,
    ):
        """Initialize MLM Predictor.

        Args:
            max_steps: Maximum number of prediction steps.
            tokenizer: Tokenizer for encoding/decoding.
            noise_schedule: Noise schedule for the diffusion process.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            model: The MLM model to use for predictions.
            with_relay: If True, use RotaryTransformerRelayModel (forward(x, h) -> (logits, h_s)).
                Carries h across steps for validation infill.
            log_rollout_diagnostics: If True, log relay step stats to the logger (when
                ``with_relay``) and add ``inference_steps`` per sample to
                :meth:`to_dict` output (jsonl debug logs from ``LogPredictions``).
            log_rollout_diagnostics_max_steps: Max relay steps to print per batch when logging.
            capture_trajectory: If True, :meth:`predict` includes ``trajectory`` — per-sample list of
                token-id snapshots after each decoding step (CPU ints). Off by default for training.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.with_relay = with_relay
        self.max_new_tokens = max_new_tokens
        if top_k is not None and top_p is not None:
            self.sampling_function = sample_from_logits
        elif top_k is not None and top_p is None:
            self.sampling_function = partial(sample_from_top_k, top_k)
        elif top_k is None and top_p is not None:
            self.sampling_function = partial(sample_from_top_p, top_p)
        else:
            raise ValueError("Both top_k and top_p cannot be non-None")

        self.confidence = confidence
        self.threshold = threshold
        self.noise_schedule = noise_schedule
        self.skip_special_tokens = skip_special_tokens
        self.log_rollout_diagnostics = log_rollout_diagnostics
        self.log_rollout_diagnostics_max_steps = log_rollout_diagnostics_max_steps
        self.capture_trajectory = capture_trajectory

    def _record_relay_rollout_step_diagnostics(
        self,
        step_results: MLMStepResults,
        logits: Float[TT, " batch seq_len vocab_size"],
        masked_mutable: Bool[TT, " batch seq_len"],
        h_new: Float[TT, " batch seq_len d_model"],
    ) -> None:
        h_prev = step_results.get("h")
        if h_prev is None:
            return
        with torch.no_grad():
            step_idx = int(step_results["steps_taken"][0].item())
            masked_count = masked_mutable.sum(dim=-1).float().mean().item()
            if masked_mutable.any():
                post_conf = logits.softmax(dim=-1).max(dim=-1)[0]
                mean_conf_masked = post_conf[masked_mutable].mean().item()
            else:
                mean_conf_masked = 0.0
            diagnostics_steps = step_results.setdefault(
                "rollout_diagnostics_steps", []
            )
            diagnostics_steps.append(
                {
                    "step": step_idx,
                    "masked_tokens_per_sample": float(masked_count),
                    "mean_confidence_masked": float(mean_conf_masked),
                    "h_t_norm": float(h_prev.norm(dim=-1).mean().item()),
                    "h_s_norm": float(h_new.norm(dim=-1).mean().item()),
                }
            )

    def reset(self):
        # simple predictor has no state
        pass

    def decode(self, results: MLMStepResults) -> Tuple[
        List[str],
        Integer[TT, " batch seq_len"],
    ]:
        """
        Args:
            results:
                x: Integer[TT, " batch seq_len"] Current predicted sequence.
        Returns:
            out: List[str] Decoded sequence with special tokens.
            x: Integer[TT, " batch seq_len"] Current predicted sequence.
        """
        x: Integer[TT, " batch seq_len"] = results["x"]
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            x, skip_special_tokens=self.skip_special_tokens
        )
        return out_with_spl_tokens, x

    def stop(
        self,
        step_results: MLMStepResults,
    ) -> Bool[TT, " batch"]:
        steps_done = bool(step_results["done"].all())
        fixed = step_results.get("fixed")
        x = step_results["x"]
        # Only consider mutable positions; done when no masked mutable positions remain
        masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
        all_filled = bool((masked_mutable.sum() == 0))
        max_steps_reached = bool((step_results["steps_taken"] >= self.max_steps).all())
        return steps_done or all_filled or max_steps_reached

    def compute_confidence(
        self,
        logits: Float[TT, " batch seq_len vocab_size"],
    ) -> Float[TT, " batch seq_len"]:
        """Compute per-position confidence scores from logits.

        Returns raw (B, L) scores — higher means more confident.
        Gradients are not blocked here; callers are responsible for torch.no_grad() if needed.
        """
        if self.confidence == "prob_diff":
            temp = logits.softmax(dim=-1)  # (B, L, V)
            top2_probs, _ = torch.topk(temp, k=2, dim=-1)  # (B, L, 2)
            return top2_probs[:, :, 0] - top2_probs[:, :, 1]  # (B, L)
        elif self.confidence == "entropy":
            temp = logits.softmax(dim=-1)  # (B, L, V)
            return torch.sum(temp * torch.log(temp + 1e-10), dim=-1)  # (B, L)
        elif self.confidence == "top_prob":
            return logits.softmax(dim=-1).max(dim=-1)[0]  # (B, L)
        else:
            raise ValueError(f"Unknown confidence: {self.confidence}")

    def compute_confidence_unmask(
        self,
        logits: Float[TT, " batch seq_len vocab_size"],
        masked_mutable: Bool[TT, " batch seq_len"],
        threshold: Optional[Union[float, torch.Tensor]] = None,
        done: Optional[Bool[TT, " batch"]] = None,
    ) -> Bool[TT, " batch seq_len"]:
        """Compute which positions to unmask based on confidence.

        Returns a boolean mask. No gradients flow through this — the result is
        a discrete selection used in both predict_single_step and RelayBPTTLoss.

        Args:
            done: Per-sample flag indicating samples that are already fully decoded.
                  Rows where done[i]=True are returned as all-False — nothing is
                  unmasked for a sample that has no work left.
        """
        with torch.no_grad():
            confidence = self.compute_confidence(logits).clone()
            confidence[~masked_mutable] = float("-inf")
            top_confidence_values, top_confidence_indices = torch.sort(1 - confidence, dim=-1)
            # TODO(DP): We can convert this into exponential race to add some stochasticity to the selection.
            threshold_value = self.threshold if threshold is None else threshold
            if threshold_value is None:
                raise ValueError(
                    "compute_confidence_unmask requires a threshold override or "
                    "predictor.threshold to be set."
                )
            unmask = torch.cumsum(top_confidence_values, dim=-1) < threshold_value
            unmask = unmask.scatter(-1, top_confidence_indices, unmask)

            no_unmask = ~unmask.any(dim=-1)
            has_mutable = masked_mutable.any(dim=-1)
            force_unmask = no_unmask & has_mutable
            # Force-unmask the single highest-confidence position for any row that would
            # otherwise produce an all-False mask.  Implemented via scatter rather than
            # boolean indexing so torch.compile never specialises on the runtime count of
            # True entries in `force_unmask` (which would trigger one recompile per new
            # cardinality seen during training, accumulating compiled graphs in CPU RAM).
            best_idx = top_confidence_indices[:, 0]  # [batch] — highest-confidence pos
            force_update = torch.zeros_like(unmask).scatter_(-1, best_idx.unsqueeze(-1), True)
            unmask = unmask | (force_update & force_unmask.unsqueeze(-1))

            # Samples already marked done have no mutable positions remaining; suppress
            # any residual selection (including the force-unmask fallback above) so that
            # fixed tokens are never overwritten after a sample finishes early.
            if done is not None:
                unmask = unmask & ~done.unsqueeze(-1)

            return unmask

    def predict_single_step(
        self,
        step_results: MLMStepResults,
        final_step: bool = False,
    ) -> MLMStepResults:
        # fmt: off
        x = step_results["x"]
        fixed = step_results.get("fixed")
        # fmt: on
        # Only consider masked positions that are mutable (fixed=0)
        masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
        assert self.model is not None, "Model is not initialized"
        if self.with_relay:
            h = step_results.get("h")
            logits, h_new = self.model(x, h)
            output_z = None
            self._record_relay_rollout_step_diagnostics(
                step_results, logits, masked_mutable, h_new
            )
        else:
            logits = self.model(x)
            output_z = None
        steps_left = self.max_steps - step_results["steps_taken"]
        if not final_step:
            
            if not self.confidence: # uniform sampling
                num_unmask = (
                    masked_mutable.sum(dim=-1) / (steps_left).clamp(min=1)
                ).long()
                unmask = select_random_indices(
                    inp_shape=x.shape,
                    num_unmask=num_unmask,
                    select_from_mask=masked_mutable,
                    selection_score=None,  # uniform
                    selection_mode="sample",
                )
                
            else: # confidence-based sampling
                unmask = self.compute_confidence_unmask(
                    logits, masked_mutable, done=step_results["done"]
                )

        else:
            unmask = masked_mutable
        # Teacher forcing: fill with target_ids; else use model predictions
        target_ids = step_results.get("target_ids")
        if target_ids is not None:
            x[unmask] = target_ids[unmask]
        else:
            x[unmask] = self.sampling_function(logits[unmask])
        # compute stopping condition
        step_results["steps_taken"] += 1
        masked_mutable_after = (x == self.tokenizer.mask_token_id) & ~fixed
        per_sample_filled = ~masked_mutable_after.any(dim=-1)
        done = (step_results["steps_taken"] >= self.max_steps) | per_sample_filled
        result: MLMStepResults = {
            "x": x,
            "fixed": fixed,
            "logits": logits,
            "steps_taken": step_results["steps_taken"],
            "done": done,
        }
        if self.with_relay:
            result["h"] = h_new
        if step_results.get("target_ids") is not None:
            result["target_ids"] = step_results["target_ids"]
        return result

    def to_dict(
        self,
        batch: MLMBatch,  # type: ignore
        preds: MLMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        per_sample_steps: Optional[List[int]] = None
        if self.log_rollout_diagnostics:
            raw = preds.get("rollout_steps_per_sample")
            if raw is not None:
                per_sample_steps = [int(x) for x in raw]

        dicts: List[Dict[str, Any]] = []
        for i, (text, ids) in enumerate(
            zip(
                preds["text"],
                preds["ids"].tolist(),
            )
        ):
            row: Dict[str, Any] = {"text": text, "ids": ids}
            if per_sample_steps is not None and i < len(per_sample_steps):
                row["inference_steps"] = per_sample_steps[i]
            dicts.append(row)
        return dicts

    @torch._dynamo.disable()
    def predict(
        self,
        batch: MLMBatch,  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMPredictionDict:
        _start_time = time.time()
        x = batch["input_ids"].clone()
        if "fixed" not in batch:
            batch["fixed"] = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        fixed = batch["fixed"]

        step_results: MLMStepResults = {
            "x": x,
            "fixed": fixed,
            "logits": None,  # type: ignore # ok for first step
            "steps_taken": torch.zeros(
                x.shape[0], dtype=torch.int, device=x.device
            ),
            "done": torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
        }
        if self.with_relay and self.model is not None:
            # h_i = 0 (zero vector, no randomness), shape (B, L, d_model)
            step_results["h"] = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.model.d_model,
                device=x.device,
            )
            step_results["rollout_diagnostics_steps"] = []
        trajectory_snapshots: Optional[List[List[List[int]]]] = None
        if self.capture_trajectory:
            trajectory_snapshots = [step_results["x"].detach().cpu().tolist()]
        while not self.stop(step_results):
            step_results = self.predict_single_step(
                step_results,
            )
            if self.capture_trajectory:
                assert trajectory_snapshots is not None
                trajectory_snapshots.append(
                    step_results["x"].detach().cpu().tolist()
                )
        step_results = self.predict_single_step(
            step_results,
            final_step=True,
        )
        if self.capture_trajectory:
            assert trajectory_snapshots is not None
            trajectory_snapshots.append(
                step_results["x"].detach().cpu().tolist()
            )
        # decode the final step
        (
            out,
            final_x,
        ) = self.decode(step_results)

        _end_time = time.time()
        _time_taken = _end_time - _start_time
        rollout_steps = step_results["steps_taken"].float().mean().item()

        rollout_diagnostics: Dict[str, Any] = {}
        if self.with_relay:
            rollout_diagnostics["steps"] = step_results.get(
                "rollout_diagnostics_steps", []
            )
            if self.log_rollout_diagnostics:
                for step_diag in rollout_diagnostics["steps"][
                    : self.log_rollout_diagnostics_max_steps
                ]:
                    logger.info(
                        "Rollout step=%d masked=%.3f conf=%.4f h_t_norm=%.4f h_s_norm=%.4f",
                        step_diag["step"],
                        step_diag["masked_tokens_per_sample"],
                        step_diag["mean_confidence_masked"],
                        step_diag["h_t_norm"],
                        step_diag["h_s_norm"],
                    )
        self.reset()
        result: Dict[str, Any] = {
            "text": out,
            "ids": final_x,
            "loss": None,
            "time_taken": [_time_taken]
            * len(out),  # cannot separate time for each sample
            "rollout_steps": rollout_steps,
            "rollout_diagnostics": rollout_diagnostics,
        }
        if self.log_rollout_diagnostics:
            # Per-sequence step count after the full predict loop (includes final fill step).
            result["rollout_steps_per_sample"] = (
                step_results["steps_taken"].detach().cpu().tolist()
            )
        if self.capture_trajectory and trajectory_snapshots is not None:
            # trajectory_snapshots: list over time of shape (batch, seq) token ids
            bsz = len(trajectory_snapshots[0])
            traj_per_sample: List[List[List[int]]] = []
            for b in range(bsz):
                traj_per_sample.append(
                    [trajectory_snapshots[t][b] for t in range(len(trajectory_snapshots))]
                )
            result["trajectory"] = traj_per_sample
        return result
