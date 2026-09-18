"""2-step RELAY loss for Fast-dLLM v2 (paper Algorithm 1).

Preserves Fast-dLLM v2's BD3-LM-style seq-dim ``[x_t || x0]`` doubling and the
``gen_mask`` block-diagonal + offset-block-causal attention pattern. Bypasses
only the parts of the model's training-time noising that are incompatible with
deterministic all-mask init (the random ``t ~ U[0, 1]`` noising and the
batch-dim complement).

The relay state h_s flows between the two forward passes through a
position-guarded LayerNorm additive: h_s is zero-padded to ``(B, 2L, D)``
(matching the doubled ``[x_t || x0]`` embedding) and added through a
zero-initialized LayerNorm at **mask-token positions only**, so the carried
state never overrides committed tokens (paper Algorithm 1 line 7,
``x_emb + R_theta(h)``).

Trajectory training uses a persistent rollout buffer (``StreamingBatch``).
Each slice's partial-mask ``x_input`` and detached relay state ``h_s`` persist
across calls. Each call advances a slice by ``2 * num_unmasks_per_step``
reveals; once a slice has no remaining masks it is evicted and replaced by a
fresh batch row. The model therefore trains on the full unmasking trajectory
rather than only the high-mask tail it would otherwise see in 2 on-policy
steps. (See ``streaming_batch.py`` for buffer mechanics.)

Step 1
    Build ``doubled_x1 = cat([x_t1, x0], dim=1)`` of shape ``(B, 2L)``.
    Forward with ``bypass_noising=True``. Apply Fast-dLLM v2's own
    threshold-based selection rule (verbatim from
    ``generation_functions.py:114-123``) **per BD attention block** so every
    block is guaranteed at least one teacher-force reveal per RELAY step,
    matching the block-by-block decoding loop in
    ``Fast_dLLM_QwenForCausalLM.generate``. ``L1`` is the mean cross-entropy
    over **all masked positions** at step 1 (standard MDM loss).

Step 2
    Reveal the ground-truth tokens at the selected positions. Build
    ``doubled_x2 = cat([revealed_x_t, x0], dim=1)`` and the relay carry
    ``h_t`` from forward 1's ``h_s`` (zero-padded over the clean half).
    Forward with the carry. ``L2`` is the mean cross-entropy over the
    positions still masked at step 2.

The total loss is ``L1 + L2`` and its gradient flows back through the
relay state ``h_s1`` into both the relay LayerNorm parameters and the
shared backbone parameters (the RELAY row in Table 2). The
``stop_grad_h_s`` ablation detaches ``h_s1`` between the two forwards, so
only forward-2 parameters receive gradients via the relay path (the
RELAY (sg) row in Table 2). ``h_s2`` is detached and stashed back into
the rollout buffer to seed the next call's forward 1.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming_batch import StreamingBatch


# DeepSpeed ZeRO-3 partitions large parameters across ranks; on every rank
# ``param.data`` is then the local shard (often a 0-element stub). The
# ``safe_get_full_fp32_param`` helper materialises the full parameter on the
# current rank, regardless of partitioning. Smaller params that stay below
# DeepSpeed's persistence threshold are unaffected (it just returns
# ``param.data`` for them). We fall back to a no-op when DeepSpeed is not
# installed, so single-GPU and plain-DDP runs keep working unchanged.
try:
    from deepspeed.utils import safe_get_full_fp32_param  # type: ignore
except Exception:  # pragma: no cover - non-DS environments
    def safe_get_full_fp32_param(p):  # type: ignore[no-redef]
        return p.data


def _safe_param_norm(p: torch.nn.Parameter) -> float:
    """Return ``||p||_2`` as a Python float, gathering the full param under
    ZeRO-3 first. Returns 0.0 if the gather fails (e.g. the param hasn't
    been initialised yet on this rank)."""
    full = safe_get_full_fp32_param(p)
    if full is None:
        return 0.0
    return full.detach().float().norm().item()


class FastDLLMBlockBPTTLoss(nn.Module):
    def __init__(
        self,
        mask_token_id: int,
        threshold: float = 0.85,
        top_p: float = 0.95,
        temperature: float = 0.0,
        use_relay: bool = True,
        tokenizer: Optional[Any] = None,
        stop_grad_h_s: bool = False,
        unmask_strategy: str = "bd",
        inner_block_size: int = 8,
    ):
        super().__init__()
        if unmask_strategy not in ("bd", "decode_aligned"):
            raise ValueError(
                "unmask_strategy must be one of 'bd' / 'decode_aligned', "
                f"got {unmask_strategy!r}"
            )
        if inner_block_size <= 0:
            raise ValueError(f"inner_block_size must be positive, got {inner_block_size}")
        self.mask_token_id = mask_token_id
        self.threshold = threshold
        self.top_p = top_p
        self.temperature = temperature
        self.unmask_strategy = unmask_strategy
        self.inner_block_size = int(inner_block_size)
        # ``use_relay=False`` keeps the 2-step rollout schedule + buffer but
        # passes ``h_t=None`` into both forwards, isolating the loss-schedule
        # contribution from the relay channel itself.
        self.use_relay = bool(use_relay)
        # ``stop_grad_h_s`` detaches ``h_s1`` between the two forwards so
        # gradients from forward 2 cannot flow back through the relay into
        # forward 1 (the RELAY (sg) row of Table 2). Mirrors the equivalent
        # flag in the Sudoku codebase (``relay/loss.py``). No-op when the
        # relay is off (no relay state to detach).
        self.stop_grad_h_s = bool(stop_grad_h_s) and self.use_relay
        self.train_buffer: StreamingBatch = StreamingBatch()
        self.eval_buffer: StreamingBatch = StreamingBatch()
        self.tokenizer = tokenizer

    def decode_ids_for_debug(
        self,
        ids: torch.Tensor,
        row: int = 0,
        *,
        is_labels: bool = False,
    ) -> str:
        """Turn one batch row of token ids into text for logging / debugger watches.

        When ``is_labels`` is True, ``-100`` (HF ignore index) is replaced with
        the tokenizer's ``pad_token_id`` only for decoding so you can read the
        string; ignored positions are not real pads.
        """
        if self.tokenizer is None:
            return "<no tokenizer: pass tokenizer=... into FastDLLMBlockBPTTLoss or set .tokenizer>"
        row_t = ids[row].detach().cpu()
        if is_labels:
            t = row_t.clone()
            t[t == -100] = self.tokenizer.pad_token_id
            row_t = t
        return self.tokenizer.decode(row_t.tolist(), skip_special_tokens=False)

    @staticmethod
    def _shift_logits(logits: torch.Tensor) -> torch.Tensor:
        """Right-shift logits so ``shifted[i]`` predicts token *at* position ``i``.

        Fast-dLLM v2 (like all GPT-style models) uses next-token prediction:
        raw ``logits[i]`` is the distribution over position ``i+1``. The
        model's own ``ForCausalLMLoss`` accounts for this by left-shifting
        labels, but we need position-aligned tensors for mask/unmask indexing,
        so we shift the logits instead (same as ``eval.py:get_logits``).
        """
        return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

    @staticmethod
    def _shift_h_s(h_s: torch.Tensor) -> torch.Tensor:
        """Right-shift the carried hidden states by one position.

        The same alignment story as ``_shift_logits``: at row ``i`` the raw
        backbone hidden state was used to *predict* token ``i+1``, so to
        carry "state describing position i" we right-shift, duplicating
        position 0 at the boundary. This way, when the next forward
        consumes ``h_t`` (additively for Loopholing or as KV for CAB), the
        position-i query is matched with the carry that was responsible
        for position-i, not position (i+1).

        Discussed with Dhruvesh; mirrors what stateflow does for its
        carried latent reps.
        """
        return torch.cat([h_s[:, :1, :], h_s[:, :-1, :]], dim=1)

    @torch.no_grad()
    def _select_unmask(
        self,
        base: nn.Module,
        logits: torch.Tensor,
        mask_idx: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Per-block Fast-dLLM v2 threshold rule.

        Mirrors the inference loop in
        ``Fast_dLLM_QwenForCausalLM.generate``: per (small) block, take all
        positions with confidence above ``self.threshold`` AND the per-block
        argmax as a safety net. Block boundaries follow the BD attention
        pattern in ``block_diff_mask`` -- absolute positions ``[b*K, (b+1)*K)``
        for ``K = block_size`` -- so the loss schedule and the model's
        attention mask see the same block grid.

        Without the per-block argmax fallback, a single global argmax (one
        reveal across all blocks, the previous behaviour) means most blocks
        receive zero teacher-force reveals between forwards 1 and 2 early
        in training when most positions have ``p < threshold``, and L2
        collapses onto L1.

        ``logits`` must already be position-aligned (i.e. passed through
        ``_shift_logits``).
        """
        x_1, p_1t = base.sample_with_top_p(
            logits, top_p=self.top_p, temperature=self.temperature
        )
        x1_p = torch.gather(p_1t, dim=-1, index=x_1.unsqueeze(-1)).squeeze(-1)
        x1_p = torch.where(mask_idx, x1_p, torch.full_like(x1_p, float("-inf")))

        unmask_idx = x1_p > self.threshold

        # Per-block argmax safety net: one reshape + one argmax + one scatter,
        # no Python loop over blocks. Pad the tail with -inf so the last
        # (possibly partial) block's argmax stays a no-op when all of its
        # positions are non-maskable / already-revealed.
        B, L = x1_p.shape
        K = block_size
        n_blocks = (L + K - 1) // K
        pad = n_blocks * K - L
        if pad:
            padded = F.pad(x1_p, (0, pad), value=float("-inf"))
        else:
            padded = x1_p
        per_block = padded.view(B, n_blocks, K)
        local_argmax = per_block.argmax(dim=-1)                          # (B, nb)
        # Skip blocks with no candidate (all -inf): prompt-only blocks, or
        # blocks that became fully revealed in a previous step. Without this
        # the scatter would write True at an arbitrary position; the final
        # ``& mask_idx`` would clear it, but the explicit skip is cleaner.
        has_candidate = torch.isfinite(per_block).any(dim=-1)            # (B, nb)
        block_offsets = torch.arange(n_blocks, device=x1_p.device) * K   # (nb,)
        global_idx = (local_argmax + block_offsets).clamp_max(L - 1)     # (B, nb)

        b_idx = (
            torch.arange(B, device=x1_p.device).unsqueeze(-1).expand_as(global_idx)
        )
        sel_b = b_idx[has_candidate]
        sel_p = global_idx[has_candidate]
        unmask_idx[sel_b, sel_p] = True

        return unmask_idx & mask_idx

    @torch.no_grad()
    def _select_unmask_decode_aligned(
        self,
        base: nn.Module,
        logits: torch.Tensor,
        mask_idx: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Decode-aligned threshold rule with inner-block autoregressive gating.

        Within each BD block, only the earliest inner block that still contains
        maskable masked positions may reveal tokens. This mirrors inference,
        where decoding advances inner block by inner block inside each BD block.
        The threshold rule and argmax fallback are both restricted to that
        active inner block.
        """
        if self.inner_block_size > block_size:
            raise ValueError(
                f"inner_block_size ({self.inner_block_size}) cannot exceed "
                f"BD block_size ({block_size})"
            )
        if block_size % self.inner_block_size != 0:
            raise ValueError(
                f"BD block_size ({block_size}) must be divisible by "
                f"inner_block_size ({self.inner_block_size})"
            )

        x_1, p_1t = base.sample_with_top_p(
            logits, top_p=self.top_p, temperature=self.temperature
        )
        x1_p = torch.gather(p_1t, dim=-1, index=x_1.unsqueeze(-1)).squeeze(-1)
        x1_p = torch.where(mask_idx, x1_p, torch.full_like(x1_p, float("-inf")))

        B, L = x1_p.shape
        K = block_size
        inner = self.inner_block_size
        n_bd_blocks = (L + K - 1) // K
        pad = n_bd_blocks * K - L
        padded = F.pad(x1_p, (0, pad), value=float("-inf")) if pad else x1_p

        n_inner = K // inner
        per_inner = padded.view(B, n_bd_blocks, n_inner, inner)
        has_candidate = torch.isfinite(per_inner).any(dim=-1)  # (B, bd, inner)
        bd_has_candidate = has_candidate.any(dim=-1)           # (B, bd)
        active_inner = has_candidate.float().argmax(dim=-1)    # earliest True

        inner_idx = torch.arange(n_inner, device=x1_p.device).view(1, 1, n_inner)
        active_mask = (
            (inner_idx == active_inner.unsqueeze(-1))
            & bd_has_candidate.unsqueeze(-1)
        )
        candidate_mask = active_mask.unsqueeze(-1) & torch.isfinite(per_inner)

        active_scores = torch.where(
            candidate_mask,
            per_inner,
            torch.full_like(per_inner, float("-inf")),
        )
        unmask_padded = (active_scores > self.threshold).view(B, n_bd_blocks * K)

        # Argmax fallback within the same active inner block for each BD block.
        flat_scores = active_scores.view(B, n_bd_blocks, K)
        local_argmax = flat_scores.argmax(dim=-1)                       # (B, bd)
        block_offsets = torch.arange(n_bd_blocks, device=x1_p.device) * K
        global_idx = (local_argmax + block_offsets).clamp_max(L - 1)
        b_idx = (
            torch.arange(B, device=x1_p.device).unsqueeze(-1).expand_as(global_idx)
        )
        sel_b = b_idx[bd_has_candidate]
        sel_p = global_idx[bd_has_candidate]
        unmask_padded[sel_b, sel_p] = True

        return unmask_padded[:, :L] & mask_idx

    def _select_unmask_for_strategy(
        self,
        base: nn.Module,
        logits: torch.Tensor,
        mask_idx: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        if self.unmask_strategy == "decode_aligned":
            return self._select_unmask_decode_aligned(base, logits, mask_idx, block_size)
        return self._select_unmask(base, logits, mask_idx, block_size)

    def _prepare_h_t(self, h_s: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Reshape ``h_s`` from the previous forward into ``h_t`` for the next.

        The relay injection is an element-wise add on the doubled
        ``(B, 2L, D)`` ``[x_t || x0]`` embedding, so we zero-pad the ``x0``
        half here. The model's mask-position guard then restricts the
        non-zero LayerNorm delta to the noisy-half mask tokens.

        We always right-shift the carry first via ``_shift_h_s`` so that the
        relay state at position ``i`` corresponds to "the state used to
        predict token i", aligning with the right-shifted logits used for
        loss / unmask selection. Without this shift, every relay add at
        position ``i`` would be fed the state for position ``i+1`` (an
        off-by-one mismatch).

        Returns ``None`` when the relay is disabled, so the model's forward
        short-circuits the injection branch entirely (used by the
        rollout-only ablation that keeps the 2-step loss but no relay).
        """
        if h_s is None or not self.use_relay:
            return None
        h_s = self._shift_h_s(h_s)
        return torch.cat([h_s, torch.zeros_like(h_s)], dim=1)

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask=None,
    ):
        """Run the 2-step BPTT and return ``(loss, metrics_dict, out2)``.

        ``model`` is the (possibly DDP/FSDP-wrapped) ``Fast_dLLM_QwenForCausalLM``;
        ``input_ids`` is the original GT sequence ``(B, L)``; ``labels`` is
        ``-100`` outside the response region and the GT token id inside it.
        """
        base = model.module if hasattr(model, "module") else model
        if not hasattr(base, "sample_with_top_p"):
            raise RuntimeError(
                "FastDLLMBlockBPTTLoss requires the model to expose "
                "`sample_with_top_p`; ensure the vendored Fast-dLLM v2 class "
                "(lmflow.models.fast_dllm) is loaded."
            )

        # Block size for the BD attention pattern. Same value the model's
        # ``gen_mask`` is built with (modeling.py: ``self.bd_size``), so the
        # per-block unmask schedule and the attention mask see the same grid.
        block_size = int(base.config.bd_size)

        is_training = bool(model.training)
        sb = self.train_buffer if is_training else self.eval_buffer
        if not is_training:
            sb.reset()

        d_model = base.config.hidden_size

        batch = {"x_clean": input_ids, "labels": labels}
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask
        n_filled = sb.evict_and_fill(
            batch=batch, mask_token_id=self.mask_token_id, d_model=d_model
        )

        x_clean = sb.storage["x_clean"]
        x_t1 = sb.storage["x_input"]
        labels_b = sb.storage["labels"]
        attn_b = sb.storage.get("attention_mask")
        h_carry = sb.storage["h_s"]

        maskable = labels_b != -100

        doubled_x1 = torch.cat([x_t1, x_clean], dim=1)

        h_t_for_fwd1 = self._prepare_h_t(h_carry)
        out1 = model(
            input_ids=doubled_x1,
            labels=labels_b,
            attention_mask=attn_b,
            h_t=h_t_for_fwd1,
            bypass_noising=True,
        )
        logits1 = self._shift_logits(out1.logits)
        h_s1 = out1.h_s

        mask1 = (x_t1 == self.mask_token_id) & maskable
        unmask1 = self._select_unmask_for_strategy(base, logits1, mask1, block_size)

        ce1 = F.cross_entropy(
            logits1.transpose(1, 2),
            labels_b,
            reduction="none",
            ignore_index=-100,
        )
        L1 = (ce1 * mask1.float()).sum() / mask1.float().sum().clamp_min(1)

        x_t2 = torch.where(unmask1, x_clean, x_t1)
        doubled_x2 = torch.cat([x_t2, x_clean], dim=1)

        # Stop-grad ablation: detach the carry between forwards so L2's
        # gradient cannot flow through h_s1 back into forward 1. Mirrors
        # ``doublebackprop/loss.py:441`` (`h_next = h_s.detach() if
        # self.stop_grad_h_s else h_s`).
        h_s_for_carry = h_s1.detach() if self.stop_grad_h_s else h_s1
        h_t_for_fwd2 = self._prepare_h_t(h_s_for_carry)
        out2 = model(
            input_ids=doubled_x2,
            labels=labels_b,
            attention_mask=attn_b,
            h_t=h_t_for_fwd2,
            bypass_noising=True,
        )
        logits2 = self._shift_logits(out2.logits)
        h_s2 = out2.h_s

        mask2 = (x_t2 == self.mask_token_id) & maskable
        ce2 = F.cross_entropy(
            logits2.transpose(1, 2),
            labels_b,
            reduction="none",
            ignore_index=-100,
        )
        L2 = (ce2 * mask2.float()).sum() / mask2.float().sum().clamp_min(1)

        with torch.no_grad():
            unmask2 = self._select_unmask_for_strategy(base, logits2, mask2, block_size)
            x_input_next = torch.where(unmask2, x_clean, x_t2)
        sb.persist_after_step(
            x_input_next=x_input_next,
            h_s_next=h_s2,
            mask_token_id=self.mask_token_id,
        )

        total_maskable = maskable.sum().float().clamp_min(1)
        metrics = self._collect_metrics(
            L1=L1, L2=L2,
            mask1=mask1, mask2=mask2, unmask=unmask1,
            total_maskable=total_maskable,
            h_s1=h_s1, h_t_for_fwd2=h_t_for_fwd2,
            carry_delta_raw_norm=getattr(out2, "carry_delta_raw_norm", None),
            carry_delta_mask_norm=getattr(out2, "carry_delta_mask_norm", None),
            base=base,
        )
        metrics["bptt/unmask_strategy_decode_aligned"] = float(
            self.unmask_strategy == "decode_aligned"
        )
        metrics["bptt/inner_block_size"] = float(self.inner_block_size)
        metrics["bptt/buf_evicted"] = float(n_filled)
        mask_counts = sb.slot_mask_counts(self.mask_token_id)
        if mask_counts is not None:
            metrics["bptt/buf_mean_mask_count"] = mask_counts.float().mean().item()
            metrics["bptt/buf_mask_ratio"] = (
                mask_counts.float().mean().item()
                / total_maskable.item()
                * maskable.shape[0]
            )

        return L1 + L2, metrics, out2

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _collect_metrics(
        self,
        L1: torch.Tensor,
        L2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        unmask: torch.Tensor,
        total_maskable: torch.Tensor,
        h_s1: Optional[torch.Tensor],
        h_t_for_fwd2: Optional[torch.Tensor],
        carry_delta_raw_norm: Optional[torch.Tensor] = None,
        carry_delta_mask_norm: Optional[torch.Tensor] = None,
        base: Optional[nn.Module] = None,
    ) -> dict:
        n_maskable = total_maskable.item()
        n_mask1 = mask1.sum().item()
        n_mask2 = mask2.sum().item()
        n_unmask = unmask.sum().item()
        B = mask1.shape[0]

        m: dict = {
            "bptt/L1": L1.detach().item(),
            "bptt/L2": L2.detach().item(),
            "bptt/unmask_count": n_unmask / max(B, 1),
            "bptt/unmask_frac": n_unmask / max(n_mask1, 1),
            "bptt/mask_ratio_step1": n_mask1 / max(n_maskable, 1),
            "bptt/mask_ratio_step2": n_mask2 / max(n_maskable, 1),
        }

        if h_s1 is not None:
            m["bptt/h_s1_norm"] = h_s1.float().norm().item() / max(B, 1)
        if h_t_for_fwd2 is not None:
            m["bptt/h_t_inject_norm"] = (
                h_t_for_fwd2.float().norm().item() / max(B, 1)
            )
        if carry_delta_raw_norm is not None:
            m["bptt/carry_delta_raw_norm"] = carry_delta_raw_norm.detach().float().item()
        if carry_delta_mask_norm is not None:
            m["bptt/carry_delta_norm"] = carry_delta_mask_norm.detach().float().item()

        # ---- Critical: is the relay gate actually opening? ----
        # The relay injects ``delta = relay_layer_norm(h_t)`` with the
        # LayerNorm weight init to 0, so the gate is closed at the start of
        # training. If the LayerNorm weight norm stays at 0 it means the
        # relay never learned to contribute -- the most informative single
        # signal for "is RELAY actually being used".
        #
        # Use ``_safe_param_norm`` instead of raw ``.detach().norm()`` so the
        # number is correct under DeepSpeed ZeRO-3 -- partitioned parameters
        # have a 0-element ``.data`` stub on each rank, which would
        # otherwise report 0.
        if base is not None:
            backbone = getattr(base, "model", base)
            if (
                getattr(backbone, "use_relay", False)
                and hasattr(backbone, "relay_layer_norm")
            ):
                ln = backbone.relay_layer_norm
                m["bptt/relay_ln_weight_norm"] = _safe_param_norm(ln.weight)
                m["bptt/relay_ln_bias_norm"] = _safe_param_norm(ln.bias)

        m["bptt/relay_enabled"] = float(self.use_relay)
        m["bptt/stop_grad_h_s"] = float(self.stop_grad_h_s)
        return m
