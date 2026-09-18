"""Tests for relay rollout training (RELAY_PLAN.md, M3).

GPU only: the DIT backbone needs flash_attn. Run on Modal from the
repository root:

  modal run modal_app.py --cmd "python -m pytest -x -q tests"
"""
import hydra
import pytest
import torch

import dataloader
import diffusion
import main  # noqa: F401  registers the OmegaConf resolvers
import streaming_batch

TINY = [
  'backbone=dit',
  'model=small',
  'model.hidden_size=64',
  'model.cond_dim=32',
  'model.n_blocks=2',
  'model.n_heads=4',
  'model.length=64',
  'data=openwebtext-split',
  'relay.enabled=true',
  'relay.num_steps=8',
  'sampling.steps=8',
  'loader.global_batch_size=4',
  'loader.batch_size=4',
  'loader.eval_batch_size=4',
  'trainer.devices=1',
]


def make_config(*overrides):
  with hydra.initialize(version_base=None, config_path='../configs'):
    return hydra.compose(config_name='config',
                         overrides=[*TINY, *overrides])


@pytest.fixture(scope='module')
def tokenizer():
  return dataloader.get_tokenizer(make_config())


def make_model(tokenizer, *overrides):
  model = diffusion.Diffusion(make_config(*overrides),
                              tokenizer=tokenizer).to('cuda')
  model.train()
  return model


def make_batch(model, batch_size=4):
  length = model.config.model.length
  return {
    'input_ids': torch.randint(0, model.tokenizer.vocab_size,
                               (batch_size, length), device='cuda'),
    'attention_mask': torch.ones(batch_size, length, device='cuda'),
  }


def test_schedule_matches_ddpm_update(tokenizer):
  """The explicit reveal rule has the marginal of `_ddpm_update`."""
  model = make_model(tokenizer).eval()
  schedule = model.relay_loss.schedule
  batch_size, length = 4, model.config.model.length
  x = model._sample_prior(batch_size, length).to('cuda')
  trials = 200
  torch.manual_seed(0)
  for i in [0, schedule.num_steps // 2, schedule.num_steps - 1]:
    step = torch.full((batch_size,), i, device='cuda')
    t = schedule.t(step)[:, None]
    revealed = 0.
    with torch.no_grad():
      for _ in range(trials):
        x_next = model._ddpm_update(x, t, schedule.dt)
        revealed += (x_next != model.mask_index).float().mean().item()
    expected = schedule.reveal_probability(step)[0].item()
    assert abs(revealed / trials - expected) < 0.01, (i, expected)
  step = torch.full((batch_size,), schedule.num_steps, device='cuda')
  assert (schedule.reveal_probability(step) == 1).all()


def test_buffer_lifecycle():
  mask_index = 99
  buffer = streaming_batch.StreamingBatch(mask_index=mask_index, d_model=8)
  x0 = torch.arange(12, device='cuda').view(3, 4)
  mutable = torch.ones(3, 4, dtype=torch.bool, device='cuda')
  mutable[0, 0] = False  # a fixed position
  rows = buffer.evict_and_fill(x0, mutable)
  assert rows.tolist() == [0, 1, 2]
  assert buffer.x[0].tolist() == [0, 99, 99, 99]
  assert (buffer.x[1:] == mask_index).all()
  assert (buffer.h == 0).all() and (buffer.step == 0).all()
  assert not buffer.done.any()

  # Row 1 finishes, the others advance.
  x = buffer.x.clone()
  x[1] = x0[1]
  x[2, 0] = x0[2, 0]
  h = torch.randn(3, 4, 8, device='cuda')
  buffer.update(x, h, buffer.step + 2)
  assert buffer.done.tolist() == [False, True, False]
  assert (buffer.step == 2).all()

  # Only the finished row is replaced, by the first row of the batch.
  new_x0 = x0 + 100
  rows = buffer.evict_and_fill(new_x0, torch.ones_like(mutable))
  assert rows.tolist() == [1]
  assert torch.equal(buffer.x0[1], new_x0[0])
  assert (buffer.x[1] == mask_index).all()
  assert (buffer.h[1] == 0).all() and buffer.step[1] == 0
  assert not buffer.done[1]
  assert torch.equal(buffer.h[0], h[0]) and buffer.step[0] == 2
  assert buffer.evict_and_fill(new_x0, mutable).numel() == 0


def open_gradient_paths(model):
  """Gives the zero-initialized DiT parameters random values.

  The DiT zero-initializes its output head and its adaLN gates, so an
  untrained model has a non-zero gradient only in the head; a
  pretrained checkpoint has neither property.
  """
  with torch.no_grad():
    for name, param in model.backbone.named_parameters():
      if 'adaLN_modulation' in name or 'output_layer.linear' in name:
        param.normal_(std=0.02)


def test_rollout_loss(tokenizer):
  model = make_model(tokenizer, 'relay.cold_start_stagger=false')
  open_gradient_paths(model)
  with torch.no_grad():
    model.backbone.relay_ln.weight.fill_(0.5)  # open the relay
  batch = make_batch(model)
  loss, metrics = model.relay_loss(batch, train=True)
  loss.backward()
  assert torch.isfinite(loss)
  assert metrics['L1'] > 0 and metrics['L2'] > 0
  assert metrics['buffer_new_rows'] == 4
  assert metrics['mask_ratio_1'] == 1
  assert 0 < metrics['mask_ratio_2'] < 1
  for name, param in model.backbone.named_parameters():
    assert param.grad is not None, name
    assert torch.isfinite(param.grad).all(), name
  assert model.backbone.relay_ln.weight.grad.abs().sum() > 0
  assert model.backbone.relay_ln.bias.grad.abs().sum() > 0

  buffer = model.relay_loss.buffers['train']
  assert (buffer.step == 2).all()
  assert not buffer.done.any()
  revealed = ~buffer.masked()
  assert revealed.any()
  assert torch.equal(buffer.x[revealed], buffer.x0[revealed])
  assert (buffer.h != 0).any()
  assert not buffer.h.requires_grad


def test_state_is_attached_unless_stop_grad(tokenizer):
  for stop_grad, expected in [(False, [False, True]),
                              (True, [False, False])]:
    model = make_model(tokenizer,
                       f'relay.stop_grad_state={str(stop_grad).lower()}',
                       'relay.cold_start_stagger=false')
    seen = []
    hook = model.backbone.relay_ln.register_forward_hook(
      lambda module, inputs, output: seen.append(inputs[0].requires_grad))
    model.relay_loss(make_batch(model), train=True)
    hook.remove()
    assert seen == expected, (stop_grad, seen)


def test_trajectory_finishes_and_refills(tokenizer):
  """A row is unmasked completely within the schedule and then replaced.

  With K = 2 and T = 8 the last Bernoulli step (t ~ eps) reveals
  almost everything, so a row usually finishes after 4 calls; the
  noise-removal step (call 5) only catches leftovers. One row per
  batch keeps the test deterministic.
  """
  model = make_model(tokenizer, 'relay.cold_start_stagger=false').eval()
  batch = make_batch(model, batch_size=1)
  buffer = model.relay_loss.buffers['train']
  with torch.no_grad():
    for calls in range(1, 6):
      model.relay_loss(batch, train=True)
      if buffer.done.all():
        break
  assert buffer.done.all()
  assert calls in (4, 5), calls
  assert torch.equal(buffer.x, buffer.x0)
  assert (buffer.step == 2 * calls).all()
  with torch.no_grad():
    _, metrics = model.relay_loss(batch, train=True)
  assert metrics['buffer_new_rows'] == 1
  assert (buffer.step == 2).all()


def test_cold_start_stagger(tokenizer):
  model = make_model(tokenizer).eval()
  batch = make_batch(model, batch_size=32)
  buffer = model.relay_loss.buffers['train']
  rows = buffer.evict_and_fill(batch['input_ids'],
                               batch['attention_mask'].bool())
  torch.manual_seed(0)
  model.relay_loss._stagger(buffer, rows)
  assert buffer.step.unique().numel() > 1
  assert buffer.step.max() < model.relay_loss.schedule.num_steps
  assert (buffer.h == 0).all()
  expected = 1 - torch.exp(- model.relay_loss.schedule.sigma(buffer.step))
  observed = buffer.masked().float().mean(dim=-1)
  assert (observed - expected).abs().max() < 0.3  # binomial noise, L = 64


def test_rollout_without_state(tokenizer):
  model = make_model(tokenizer, 'relay.use_state=false')
  assert not hasattr(model.backbone, 'relay_ln')
  loss, metrics = model.relay_loss(make_batch(model), train=True)
  loss.backward()
  assert torch.isfinite(loss)
  assert 'h_norm_1' not in metrics


def test_relay_sampler(tokenizer):
  """`relay_ddpm` with a closed relay is exactly `ddpm` without a
  cache; with an open relay the carried state changes the samples."""
  model = make_model(tokenizer, 'sampling.predictor=relay_ddpm').eval()
  open_gradient_paths(model)  # make the logits depend on the input
  num_steps = model.config.sampling.steps
  batch_size = model.config.loader.eval_batch_size
  length = model.config.model.length

  torch.manual_seed(0)
  closed = model._sample(num_steps=num_steps)
  assert closed.shape == (batch_size, length)
  assert (closed != model.mask_index).all()
  assert closed.max() < model.tokenizer.vocab_size
  assert model.nfe == num_steps + 1

  # No state at all consumes the same random numbers.
  model.config.relay.use_state = False
  torch.manual_seed(0)
  without_state = model._sample(num_steps=num_steps)
  assert torch.equal(closed, without_state)
  model.config.relay.use_state = True

  with torch.no_grad():
    model.backbone.relay_ln.weight.fill_(0.5)
  torch.manual_seed(0)
  opened = model._sample(num_steps=num_steps)
  assert opened.shape == closed.shape
  assert not torch.equal(opened, closed)


def test_training_and_validation_steps(tokenizer):
  model = make_model(tokenizer)
  batch = make_batch(model)
  loss = model.training_step(batch, 0)
  assert loss.requires_grad and torch.isfinite(loss)
  loss.backward()
  model.eval()
  with torch.no_grad():
    val_loss = model.validation_step(batch, 0)
  assert torch.isfinite(val_loss)
