"""Persistent across-call rollout buffer for the RELAY training loop.

A Fast-dLLM v2 port of ``relay/streaming_batch.py`` (the rollout buffer used
by the Sudoku side of the codebase, paper Algorithm 1 line 3 -- "if no slot
is unfinished, draw a fresh ``x0``"). Differences from the upstream Sudoku
implementation:

  * No ``segment_ids`` (Fast-dLLM v2 builds its own ``gen_mask`` from
    ``labels.shape`` inside the model).
  * No ``fixed`` mask -- we use ``labels != -100`` as the maskable mask
    everywhere. Prompt positions (``labels == -100``) act as "fixed".
  * Storage schema mirrors what the RELAY loss needs to advance a buffered
    slice across calls:
        x_clean        - (capacity, L)        original GT, immutable per slice
        x_input        - (capacity, L)        current partial-mask state
        labels         - (capacity, L)        -100 outside response, GT inside
        attention_mask - (capacity, L) opt    pad mask if provided
        h_s            - (capacity, L, D)     persistent relay state, fp32
        ready_to_evict - (capacity,)          True when no maskable masks left

Per-call lifecycle (driven by ``FastDLLMBlockBPTTLoss.forward``):

  1. ``evict_and_fill(batch)`` -- at cold start initializes storage from the
     incoming batch (capacity = batch_size) and seeds ``x_input`` by
     replacing all maskable positions with ``mask_token_id``; on subsequent
     calls, evicts ready slots and fills them with rows from the new batch
     (newly-filled slots get fresh all-mask ``x_input`` and zeroed ``h_s``).
  2. The loss runs the 2-step RELAY rollout on the buffered ``(x_input, h_s)``.
  3. ``persist_after_step(x_input_next, h_s_next)`` -- writes the
     post-step state back to storage and recomputes ``ready_to_evict``.

Buffer state is per-rank, never crosses process boundaries, never enters the
autograd graph (we always ``detach().clone()`` on writes), and is *not*
saved with the model state dict (cold-starts on resume -- fine for finetune).
"""

from typing import Any, Dict, Optional

import torch


class StreamingBatch:
    def __init__(self):
        self.capacity: Optional[int] = None
        self.seq_len: Optional[int] = None
        self._device: Optional[torch.device] = None

        self.storage: Optional[Dict[str, torch.Tensor]] = None
        self.ready_to_evict: Optional[torch.Tensor] = None

    # -------------------- lifecycle --------------------

    def reset(self) -> None:
        """Drop storage so the next ``evict_and_fill`` cold-starts.

        Used for validation: each val call gets an independent buffer,
        sidestepping the train buffer's fixed capacity and persistent state.
        """
        self.capacity = None
        self.seq_len = None
        self._device = None
        self.storage = None
        self.ready_to_evict = None

    def is_initialized(self) -> bool:
        return self.storage is not None

    # -------------------- evict + fill --------------------

    def evict_and_fill(
        self,
        batch: Dict[str, torch.Tensor],
        mask_token_id: int,
        d_model: int,
        h_dtype: torch.dtype = torch.float32,
    ) -> int:
        """Cold-start or evict-then-fill from a new batch.

        ``batch`` must contain ``x_clean`` and ``labels`` of shape ``(B, L)``,
        and may contain ``attention_mask`` of shape ``(B, L)``. ``x_input`` is
        derived inside this method (mask all ``labels != -100`` positions).

        Returns the number of slots filled (0 on a no-op call where no slot
        was ready to evict).
        """
        x_clean = batch["x_clean"]
        labels = batch["labels"]
        batch_size, seq_len = x_clean.shape

        if not self.is_initialized():
            # Cold start: capacity = incoming batch size.
            self.capacity = batch_size
            self.seq_len = seq_len
            self._device = x_clean.device
            self._allocate_storage(batch, d_model, h_dtype)
            self._fill_slots(
                slots=torch.arange(self.capacity, device=self._device),
                batch=batch,
                src_offset=0,
                n=self.capacity,
                mask_token_id=mask_token_id,
            )
            return self.capacity

        if batch_size != self.capacity or seq_len != self.seq_len:
            raise ValueError(
                f"StreamingBatch: incoming batch shape ({batch_size}, {seq_len})"
                f" must match the cold-start shape ({self.capacity}, {self.seq_len})."
            )

        evicted = self.ready_to_evict.nonzero(as_tuple=True)[0]
        n = min(int(evicted.numel()), batch_size)
        if n == 0:
            return 0
        self._fill_slots(
            slots=evicted[:n],
            batch=batch,
            src_offset=0,
            n=n,
            mask_token_id=mask_token_id,
        )
        return n

    # -------------------- post-step write-back --------------------

    @torch.no_grad()
    def persist_after_step(
        self,
        x_input_next: torch.Tensor,
        h_s_next: Optional[torch.Tensor],
        mask_token_id: int,
    ) -> None:
        """Write the post-RELAY state back. Detaches both tensors.

        When the relay is disabled (``use_relay=False``) the model's forward
        skips the relay LayerNorm branch and returns ``h_s = None``; in that
        case we leave the buffer's ``h_s`` slot at its zero-init value (which
        is what ``_prepare_h_t`` will see and short-circuit again next step).
        """
        self.storage["x_input"] = x_input_next.detach().clone()
        if h_s_next is not None:
            # Cast h_s to the buffer's dtype (fp32) so subsequent reads + the
            # injection-time ``h_t.to(inputs_embeds.dtype)`` stay numerically
            # stable across mixed-precision training.
            self.storage["h_s"] = (
                h_s_next.detach().to(self.storage["h_s"].dtype).clone()
            )

        labels = self.storage["labels"]
        x_input = self.storage["x_input"]
        maskable = labels != -100
        still_masked = (x_input == mask_token_id) & maskable
        self.ready_to_evict = ~still_masked.any(dim=-1)

    # -------------------- internals --------------------

    def _allocate_storage(
        self,
        batch: Dict[str, torch.Tensor],
        d_model: int,
        h_dtype: torch.dtype,
    ) -> None:
        cap, seq = self.capacity, self.seq_len
        device = self._device
        storage: Dict[str, torch.Tensor] = {
            "x_clean": torch.zeros((cap, seq), dtype=batch["x_clean"].dtype, device=device),
            "x_input": torch.zeros((cap, seq), dtype=batch["x_clean"].dtype, device=device),
            "labels": torch.zeros((cap, seq), dtype=batch["labels"].dtype, device=device),
            "h_s": torch.zeros((cap, seq, d_model), dtype=h_dtype, device=device),
        }
        if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
            storage["attention_mask"] = torch.zeros(
                (cap, seq), dtype=batch["attention_mask"].dtype, device=device
            )
        self.storage = storage
        self.ready_to_evict = torch.zeros(cap, dtype=torch.bool, device=device)

    def _fill_slots(
        self,
        slots: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        src_offset: int,
        n: int,
        mask_token_id: int,
    ) -> None:
        src = slice(src_offset, src_offset + n)
        x_clean_in = batch["x_clean"][src].to(self.storage["x_clean"].device)
        labels_in = batch["labels"][src].to(self.storage["labels"].device)

        self.storage["x_clean"][slots] = x_clean_in
        self.storage["labels"][slots] = labels_in
        # Fresh slice: mask all maskable positions, keep prompt as-is.
        maskable_in = labels_in != -100
        x_input_in = torch.where(
            maskable_in,
            torch.full_like(x_clean_in, mask_token_id),
            x_clean_in,
        )
        self.storage["x_input"][slots] = x_input_in
        # Fresh slice: zero relay state.
        self.storage["h_s"][slots] = 0
        if "attention_mask" in self.storage and "attention_mask" in batch:
            self.storage["attention_mask"][slots] = batch["attention_mask"][src].to(
                self.storage["attention_mask"].device
            )
        # Newly filled slots are not ready to evict.
        self.ready_to_evict[slots] = False

    # -------------------- diagnostics --------------------

    def slot_mask_counts(self, mask_token_id: int) -> Optional[torch.Tensor]:
        """For debugging: per-slot count of remaining maskable masked positions."""
        if not self.is_initialized():
            return None
        labels = self.storage["labels"]
        x_input = self.storage["x_input"]
        maskable = labels != -100
        still_masked = (x_input == mask_token_id) & maskable
        return still_masked.sum(dim=-1)

    # Helper: keep the API close to ``relay/streaming_batch.py`` for grep-ability.
    storage_get: Any = None  # noqa: E501

    def __repr__(self) -> str:  # pragma: no cover
        if not self.is_initialized():
            return "StreamingBatch(uninitialized)"
        return (
            f"StreamingBatch(capacity={self.capacity}, seq_len={self.seq_len}, "
            f"ready_to_evict={int(self.ready_to_evict.sum().item())})"
        )
