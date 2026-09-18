"""Rollout buffer for relay training (RELAY_PLAN.md, M3).

A row is a sequence in the middle of being unmasked over several
training calls: its clean tokens, its current partially masked tokens,
the relay state carried from the previous denoising step, the positions
that may be masked, and its position on the denoising schedule. Rows
with no masked mutable position left are replaced by rows of the next
incoming batch (the rest of that batch is discarded). The buffer lives
on one rank, never enters the autograd graph and is not saved with
checkpoints: a resumed run starts from a fresh buffer.
"""
import torch


class StreamingBatch:
  def __init__(self, mask_index, d_model):
    self.mask_index = mask_index
    self.d_model = d_model
    self.reset()

  def reset(self):
    self.x0 = None  # (capacity, length) clean tokens
    self.x = None  # (capacity, length) current tokens
    self.h = None  # (capacity, length, d_model) relay state
    self.mutable = None  # (capacity, length) positions that can be masked
    self.step = None  # (capacity,) position on the schedule
    self.done = None  # (capacity,) no masked mutable position left

  @property
  def initialized(self):
    return self.x0 is not None

  def masked(self):
    return (self.x == self.mask_index) & self.mutable

  def evict_and_fill(self, x0, mutable):
    """Replaces the finished rows with rows of the incoming batch.

    On a cold start every row is filled. A fresh row has all its
    mutable positions masked, a zero relay state and step 0. Rows are
    taken from the front of the batch; if the batch is smaller than
    the number of finished rows, the remaining finished rows wait for
    the next call.

    Returns:
      The indices of the filled rows.
    """
    if not self.initialized:
      capacity, length = x0.shape
      self.x0 = torch.zeros_like(x0)
      self.x = torch.zeros_like(x0)
      self.h = torch.zeros(capacity, length, self.d_model,
                           device=x0.device)
      self.mutable = torch.zeros_like(mutable)
      self.step = torch.zeros(capacity, dtype=torch.long,
                              device=x0.device)
      self.done = torch.ones(capacity, dtype=torch.bool,
                             device=x0.device)
    assert x0.shape[1] == self.x0.shape[1], (x0.shape, self.x0.shape)
    rows = self.done.nonzero(as_tuple=True)[0]
    rows = rows[:x0.shape[0]]
    n = rows.numel()
    self.x0[rows] = x0[:n]
    self.mutable[rows] = mutable[:n]
    self.x[rows] = torch.where(mutable[:n], self.mask_index, x0[:n])
    self.h[rows] = 0
    self.step[rows] = 0
    self.done[rows] = False
    return rows

  def update(self, x, h, step):
    """Stores the state after a rollout call."""
    self.x = x.detach().clone()
    if h is not None:
      self.h = h.detach().float().clone()
    self.step = step.detach().clone()
    self.done = ~self.masked().any(dim=-1)
