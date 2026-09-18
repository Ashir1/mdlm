"""A tensor buffer that acts like a streaming batch (the rollout buffer of Algorithm 1)."""

from typing import Dict, Optional, TypedDict, Union

import torch

from .types import MLMBatch


class BufferSample(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    targets: torch.Tensor
    fixed_positions_mask: torch.Tensor


class StreamingBatch:
    """Rollout buffer for ``RelayBPTTLoss``: dict of tensors ``(capacity, seq_len)``
    with per-row eviction state.

    Can be constructed without ``capacity`` / ``seq_len`` / ``device``; these are
    inferred from the first batch passed to :meth:`evict_and_fill` and used to
    initialize storage.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self._device = device

        # Storage: (capacity, seq_len), keys aligned with MLMBatch
        self.storage: Optional[Dict[str, torch.Tensor]] = None
        # Ready to evict: no masked mutable positions left for that slice
        self.ready_to_evict: Optional[torch.Tensor] = None

    def initialize_storage(self, batch: MLMBatch, device: Optional[torch.device] = None) -> None:
        if self.capacity is None or self.seq_len is None:
            raise RuntimeError("StreamingBatch capacity/seq_len must be set before initialize_storage")
        if device is None:
            for key in batch.keys():
                elem = batch[key]
                if isinstance(elem, torch.Tensor):
                    device = elem.device
                    break
        if device is None:
            raise ValueError("No device found")
        storage: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # (capacity, *rest) matches batch[:, ...] or batch[...] for 1D (B,) tensors; flex BlockMask is not stored here.
                storage[k] = torch.zeros(
                    (self.capacity,) + v.shape[1:], dtype=v.dtype, device=device
                )
        self.storage = storage
        self._device = device
        self.ready_to_evict = torch.zeros(
            self.capacity, dtype=torch.bool, device=device
        )

    def evict_and_fill(
        self,
        batch: MLMBatch,
        mask_token_id: Optional[Union[int, torch.Tensor]] = None,
        batch_shape_key: str = "input_ids",
    ) -> int:
        """Evict all slices that are ready and overwrite with examples from batch.

        If storage is not yet initialized (cold start), extracts capacity/seq_len/device
        from batch[batch_shape_key].shape, calls initialize_from_batch, and returns 0.
        Otherwise validates batch shape matches, then returns the number of slots evicted.
        """
        if batch_shape_key not in batch or not isinstance(batch[batch_shape_key], torch.Tensor):
            raise KeyError(f"StreamingBatch: batch must contain tensor key {batch_shape_key!r}")
        shape_tensor = batch[batch_shape_key]
        batch_size, seq_len = shape_tensor.shape[0], shape_tensor.shape[1]

        if self.storage is None:
            self.capacity = batch_size
            self.seq_len = seq_len
            self._device = shape_tensor.device
            self.initialize_from_batch(
                batch, device=self._device, mask_token_id=mask_token_id or 0
            )
            return 0
        if batch_size != self.capacity or seq_len != self.seq_len:
            raise ValueError(
                f"StreamingBatch: batch shape ({batch_size}, {seq_len}) must match "
                f"({self.capacity}, {self.seq_len})"
            )
        evicted = self.ready_to_evict.nonzero(as_tuple=True)[0]
        n = min(len(evicted), batch_size)
        if n == 0:
            return 0
        evicted_slots = evicted[:n]
        for key in self.storage:
            if key in batch and isinstance(batch[key], torch.Tensor):
                self.storage[key][evicted_slots] = batch[key][:n].detach().to(
                    self.storage[key].device
                )
        self.ready_to_evict[evicted_slots] = False
        return n

    def update_after_unmask(
        self,
        unmask: torch.Tensor,
        x_clean: torch.Tensor,
        mask_token_id: Union[int, torch.Tensor],
    ) -> None:
        """Teacher-force unmask positions in storage; then set ready_to_evict per slice."""
        if self.storage is None:
            raise RuntimeError("StreamingBatch storage not initialized")
        # In-place: fill unmasked positions with clean tokens
        self.storage["input_ids"][unmask] = x_clean[unmask].to(
            self.storage["input_ids"].device
        )
        input_ids = self.storage["input_ids"]
        fixed = self.storage["fixed"]
        if isinstance(mask_token_id, torch.Tensor):
            mask_token_id = mask_token_id.item()
        masked_mutable = (input_ids == mask_token_id) & ~fixed
        # Ready to evict when no masked mutable positions left in that slice
        self.ready_to_evict = ~(masked_mutable.any(dim=-1))

    def copy_batch_into_storage(self, batch: MLMBatch) -> None:
        """Copy full batch into storage (cold start). Assumes batch size == capacity."""
        if self.storage is None:
            raise RuntimeError("StreamingBatch storage not initialized")
        batch_size = batch["input_ids"].shape[0]
        for key in self.storage:
            if key in batch and isinstance(batch[key], torch.Tensor):
                self.storage[key][:batch_size].copy_(
                    batch[key].detach().to(self.storage[key].device)
                )

    def reset(self) -> None:
        """Discard storage so the next evict_and_fill re-initializes from scratch.

        Used for validation: each val call runs the unrolling loop on an independent
        batch without cross-batch state, and without the training buffer's fixed capacity
        constraint.
        """
        self.storage = None
        self.ready_to_evict = None
        self.capacity = None
        self.seq_len = None

    def initialize_from_batch(
        self,
        batch: MLMBatch,
        device: Optional[torch.device] = None,
        mask_token_id: Union[int, torch.Tensor] = 0,
    ) -> None:
        """Cold start: allocate storage, copy batch in, set ready_to_evict."""
        self.initialize_storage(batch, device)
        self.copy_batch_into_storage(batch)
        if isinstance(mask_token_id, torch.Tensor):
            mask_token_id = mask_token_id.item()
        masked_mutable = (
            (self.storage["input_ids"] == mask_token_id) & ~self.storage["fixed"]
        )
        self.ready_to_evict = ~masked_mutable.any(dim=-1)
