"""HuggingFace ``Trainer`` subclass that swaps Fast-dLLM v2's in-model MDM
loss for the 2-step RELAY loss defined in
``block_bptt_loss.FastDLLMBlockBPTTLoss``.
"""

import torch
from transformers import Trainer

from .block_bptt_loss import FastDLLMBlockBPTTLoss


# Under DeepSpeed ZeRO-1/2/3 the user-facing parameter does NOT have a
# populated ``.grad`` after ``backward`` (DeepSpeed manages gradients in a
# flat partitioned bucket and only attaches them to ``p.grad`` transiently
# inside the engine). Same story for ``.data`` under ZeRO-3, where the
# "param" is a 0-element stub until gathered. The ``deepspeed.utils.safe_*``
# helpers handle all of those cases (including FSDP and plain DDP) and
# return a fully-materialised tensor / ``None`` when no grad exists yet.
# Fall back to ``p.grad`` / ``p.data`` if DeepSpeed is not installed.
try:
    from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param  # type: ignore

    _HAS_DS = True
except Exception:  # pragma: no cover - non-DS environments
    _HAS_DS = False

    def safe_get_full_grad(p):  # type: ignore[no-redef]
        return p.grad

    def safe_get_full_fp32_param(p):  # type: ignore[no-redef]
        return p.data


class FastDLLMBPTTTrainer(Trainer):
    def __init__(
        self,
        *args,
        mask_token_id: int,
        threshold: float = 0.85,
        top_p: float = 0.95,
        temperature: float = 0.0,
        use_relay: bool = True,
        stop_grad_h_s: bool = False,
        unmask_strategy: str = "bd",
        inner_block_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Opt out of HF's "GA loss-fix" path (transformers >= 4.46).
        # ``Fast_dLLM_QwenForCausalLM.forward`` declares ``**kwargs``, so the
        # parent ``Trainer.__init__`` autodetects
        # ``self.model_accepts_loss_kwargs = True`` and assumes the model's
        # loss already self-normalizes by ``num_items_in_batch`` (the total
        # label tokens across the whole gradient-accumulation window). With
        # that flag set, ``training_step`` skips the ``loss /= GA`` step.
        #
        # ``FastDLLMBlockBPTTLoss`` does NOT honor ``num_items_in_batch`` --
        # it returns a per-microstep MEAN of (L1 + L2). Without this override
        # the 16 micro-step losses get summed into ``tr_loss`` and the
        # backward gradient accumulates without dividing, so both ``train/loss``
        # and the pre-clip ``train/grad_norm`` come out scaled by
        # ``gradient_accumulation_steps``. Forcing the flag to ``False``
        # restores the classic per-microstep ``loss /= GA`` normalization
        # both in reporting and in ``accelerator.backward``.
        self.model_accepts_loss_kwargs = False
        self.loss_fn = FastDLLMBlockBPTTLoss(
            mask_token_id=mask_token_id,
            threshold=threshold,
            top_p=top_p,
            temperature=temperature,
            use_relay=use_relay,
            tokenizer=self.tokenizer,
            stop_grad_h_s=stop_grad_h_s,
            unmask_strategy=unmask_strategy,
            inner_block_size=inner_block_size,
        )
        # Cache the names of relay-module parameters so we can isolate their
        # gradient norm post-backward (most diagnostic signal: is the relay
        # actually learning, or is it getting noise / adversarial gradients?).
        self._carry_param_names = self._collect_carry_param_names(self.model)

    _CARRY_MODULE_NAMES = frozenset({"relay_layer_norm"})

    @classmethod
    def _is_carry_param_name(cls, name: str) -> bool:
        """True for the relay LayerNorm weights.

        Match per dotted segment so that the classification is invariant to
        any wrapping prefixes (``module.`` for DDP, ``_orig_mod.`` for
        ``torch.compile``, etc.) **and** still works when the model passed in
        is the inner ``Fast_dLLM_QwenModel`` directly (no ``model.`` prefix).
        """
        return any(p in cls._CARRY_MODULE_NAMES for p in name.split("."))

    @staticmethod
    def _collect_carry_param_names(model):
        return [n for n, _ in model.named_parameters() if FastDLLMBPTTTrainer._is_carry_param_name(n)]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, metrics, outputs = self.loss_fn(
            model,
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
            attention_mask=inputs.get("attention_mask"),
        )
        # Stash metrics on the trainer so we can pad with post-backward
        # signals (carry grad norm) before logging.
        self._pending_bptt_metrics = metrics
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)

        if self.state.global_step % max(self.args.logging_steps, 1) != 0:
            return loss

        metrics = getattr(self, "_pending_bptt_metrics", None) or {}

        # Post-backward: carry-only gradient norm vs base-model grad norm.
        # ``safe_get_full_grad`` gathers the full (un-partitioned) gradient
        # for ZeRO-1/2/3 / FSDP and returns ``None`` if no grad exists; for
        # plain DDP / single-GPU it falls back to ``p.grad``. Each rank then
        # only contributes the slice it owns; we still all-reduce the squared
        # sums so rank 0 reports the right total.
        with torch.no_grad():
            device = next(model.parameters()).device
            carry_sq = torch.zeros((), device=device, dtype=torch.float32)
            base_sq = torch.zeros((), device=device, dtype=torch.float32)
            n_carry_with_grad = 0
            n_base_with_grad = 0
            for n, p in model.named_parameters():
                g = safe_get_full_grad(p)
                if g is None:
                    continue
                # ``safe_get_full_grad`` under ZeRO-3 returns the *local* slice
                # of the full grad; under ZeRO-1/2 and DDP it returns the full
                # tensor on every rank. Both are correct as long as we
                # all-reduce sums (sharded -> SUM gives the full norm; full
                # -> SUM divided by world_size below).
                g_sq = g.detach().float().pow(2).sum()
                if self._is_carry_param_name(n):
                    carry_sq = carry_sq + g_sq
                    n_carry_with_grad += 1
                else:
                    base_sq = base_sq + g_sq
                    n_base_with_grad += 1

            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(carry_sq, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(base_sq, op=torch.distributed.ReduceOp.SUM)

            # If grads are NOT sharded (DDP / ZeRO-1/2 unsharded grads, or
            # single-GPU), every rank contributed the full squared sum, so
            # divide by world_size to undo the all-reduce overcount. Under
            # ZeRO-3 each rank owned a disjoint slice so the SUM IS the full
            # norm. We probe this by checking whether DeepSpeed was actually
            # used: if ``_HAS_DS`` and any DS engine attribute is present on
            # the model, assume sharded; otherwise assume unsharded.
            grads_are_sharded = _HAS_DS and any(
                hasattr(p, "ds_id") for p in model.parameters()
            )
            if not grads_are_sharded and world_size > 1:
                carry_sq = carry_sq / world_size
                base_sq = base_sq / world_size

            if self.is_world_process_zero():
                c = float(carry_sq.sqrt().item())
                b = float(base_sq.sqrt().item())
                metrics["bptt/carry_grad_norm"] = c
                metrics["bptt/base_grad_norm"] = b
                metrics["bptt/carry_to_base_grad_ratio"] = c / max(b, 1e-12)
                metrics["bptt/n_carry_params_with_grad"] = float(n_carry_with_grad)
                metrics["bptt/n_base_params_with_grad"] = float(n_base_with_grad)
                self.log(metrics)

        self._pending_bptt_metrics = None
        return loss
