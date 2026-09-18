"""Relay rollout training (RELAY_PLAN.md, M3).

`RelayRolloutLoss` keeps a buffer of sequences that are being unmasked
step by step (`streaming_batch.StreamingBatch`) and advances every row
by `relay.num_rollout_steps` denoising steps per call:

  1. forward on (x, h): cross-entropy on the masked positions, h_next;
  2. reveal the ground truth at the positions the sampler would reveal
     at this step;
  3. forward on the revealed x with the live h_next, and so on.

The relay state stays attached between the steps (truncated backprop
through time; `relay.stop_grad_state` detaches it instead), and the
last state is detached and stored for the next call. Positions are
revealed with the schedule of `Diffusion._ddpm_update`: at step i each
masked position is revealed with probability (mc_t - mc_s) / mc_t, and
the final noise-removal step reveals everything that is left.
"""
import torch

import streaming_batch


class Schedule:
  """Reveal schedule of MDLM's ancestral sampler, indexed by step.

  Steps 0 .. num_steps - 1 walk the time grid of `Diffusion._sample`
  (t from 1 to eps); step num_steps is the noise-removal step.
  """

  def __init__(self, noise, num_steps, eps=1e-5):
    self.noise = noise
    self.num_steps = num_steps
    self.timesteps = torch.linspace(1, eps, num_steps + 1)
    self.dt = (1 - eps) / num_steps

  def t(self, step):
    return self.timesteps.to(step.device)[
      step.clamp(max=self.num_steps)]

  def sigma(self, step):
    return self.noise(self.t(step))[0]

  def reveal_probability(self, step):
    """Probability that a masked position is revealed at `step`.

    `Diffusion._ddpm_update` keeps a masked position with probability
    mc_s / mc_t, where mc = 1 - exp(-sigma) is the mask probability.
    """
    t = self.t(step)
    move_chance_t = 1 - torch.exp(- self.noise(t)[0])
    move_chance_s = 1 - torch.exp(- self.noise(t - self.dt)[0])
    p = (move_chance_t - move_chance_s) / move_chance_t
    return torch.where(step >= self.num_steps, torch.ones_like(p), p)

  def reveal(self, masked, step):
    """Samples the positions to reveal among `masked`."""
    p = self.reveal_probability(step)[:, None]
    return masked & (torch.rand(masked.shape, device=masked.device) < p)


class RelayRolloutLoss:
  """K-step rollout loss over a persistent buffer of sequences.

  `model` is the `diffusion.Diffusion` module; its `forward` supplies
  the SUBS log-probabilities and the relay state.
  """

  def __init__(self, model, config):
    self.model = model
    self.num_rollout_steps = config.num_rollout_steps
    self.stop_grad_state = config.stop_grad_state
    self.use_state = config.use_state
    self.cold_start_stagger = config.cold_start_stagger
    self.schedule = Schedule(model.noise, config.num_steps)
    self.buffers = {
      split: streaming_batch.StreamingBatch(
        mask_index=model.mask_index,
        d_model=model.config.model.hidden_size)
      for split in ('train', 'val')}

  def __call__(self, batch, train):
    """Advances the buffer by K denoising steps.

    Validation uses its own buffer, reset on every call, so that every
    validation batch is an independent rollout.

    Returns:
      loss: sum of the K masked-token cross-entropies.
      metrics: dict of detached diagnostics.
    """
    buffer = self.buffers['train' if train else 'val']
    if not train:
      buffer.reset()
    cold_start = not buffer.initialized
    x0 = batch['input_ids']
    if 'attention_mask' in batch:
      mutable = batch['attention_mask'].bool()
    else:
      mutable = torch.ones_like(x0, dtype=torch.bool)
    rows = buffer.evict_and_fill(x0, mutable)
    if cold_start and self.cold_start_stagger:
      self._stagger(buffer, rows)

    x0, mutable = buffer.x0, buffer.mutable
    x = buffer.x.clone()
    h = buffer.h.clone() if self.use_state else None
    step = buffer.step.clone()
    metrics = {'buffer_new_rows': float(rows.numel()),
               'buffer_step': step.float().mean()}
    losses = []
    for k in range(self.num_rollout_steps):
      masked = (x == buffer.mask_index) & mutable
      log_probs, h_next = self.model.forward(
        x, self.schedule.sigma(step), h_prev=h, return_state=True)
      nll = - torch.gather(log_probs, -1, x0[:, :, None]).squeeze(-1)
      loss = (nll * masked).sum() / masked.sum().clamp(min=1)
      losses.append(loss)
      reveal = self.schedule.reveal(masked, step)
      metrics[f'L{k + 1}'] = loss.detach()
      metrics[f'mask_ratio_{k + 1}'] = (
        masked.sum() / mutable.sum().clamp(min=1))
      metrics[f'reveal_frac_{k + 1}'] = (
        reveal.sum() / masked.sum().clamp(min=1))
      x = torch.where(reveal, x0, x)
      step = step + 1
      if self.use_state:
        metrics[f'h_norm_{k + 1}'] = h_next.detach().norm(dim=-1).mean()
        last = k == self.num_rollout_steps - 1
        h = h_next.detach() if last or self.stop_grad_state else h_next
    buffer.update(x, h, step)
    return torch.stack(losses).sum(), metrics

  def _stagger(self, buffer, rows):
    """Cold start: spreads the rows over the schedule.

    Each row starts at a random step with MDLM's corruption at that
    time (every mutable position masked with probability
    1 - exp(-sigma)) and a zero relay state. The buffer then covers
    all noise levels from the first call on, and rows finish at
    different times instead of all at once.
    """
    step = torch.randint(0, self.schedule.num_steps, (rows.numel(),),
                         device=rows.device)
    move_chance = 1 - torch.exp(- self.schedule.sigma(step))
    x0, mutable = buffer.x0[rows], buffer.mutable[rows]
    masked = mutable & (
      torch.rand(x0.shape, device=x0.device) < move_chance[:, None])
    buffer.x[rows] = torch.where(masked, buffer.mask_index, x0)
    buffer.step[rows] = step
