"""Times the pieces of a training step on the current GPU.

  modal run modal_app.py --cmd "PYTHONPATH=. python scripts/profile_step.py \
    relay.enabled=true loader.batch_size=16 loader.global_batch_size=16 \
    data=wikitext2 data.cache_dir=/root/outputs/data loader.num_workers=4"

Overrides are Hydra overrides of configs/config.yaml. Prints, with CUDA
synchronization: the backbone forward, the full `Diffusion.forward`
(with the SUBS head), `training_step` + backward + AdamW + EMA, and the
dataloader time per batch.
"""
import gc
import itertools
import os
import sys
import time

import hydra
import lightning as L
import torch

import dataloader
import diffusion


class StepTimer(L.Callback):
  """Prints, every `every` training batches, the mean wall time inside
  a batch (training_step + backward + optimizer) and between batches
  (dataloader, logging, callbacks), GPU memory and allocator retries.

    python main.py ... +callbacks.step_timer._target_=scripts.profile_step.StepTimer
  """

  def __init__(self, every=25):
    self.every = every
    self.count = 0
    self.inside = 0.
    self.between = 0.
    self.start = None
    self.last_end = None

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    self.start = time.perf_counter()
    if self.last_end is not None:
      self.between += self.start - self.last_end

  def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                         batch_idx):
    torch.cuda.synchronize()
    self.last_end = time.perf_counter()
    self.inside += self.last_end - self.start
    self.count += 1
    if self.count % self.every == 0:
      stats = torch.cuda.memory_stats()
      print(f'batches {self.count - self.every + 1:4d}-{self.count:4d}: '
            f'inside {self.inside / self.every * 1e3:5.0f} ms, '
            f'between {self.between / self.every * 1e3:5.0f} ms, '
            f'reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB, '
            f'alloc_retries {stats["num_alloc_retries"]}, '
            f'gc {gc.get_count()}', flush=True)
      self.inside = self.between = 0.


def timed(fn, n=10, warmup=3):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(n):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - start) / n


def profile(overrides):
  import main  # noqa: F401  registers the OmegaConf resolvers
  with hydra.initialize(version_base=None, config_path='../configs'):
    config = hydra.compose(config_name='config', overrides=overrides)
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
  model.train()
  if model.ema:
    model.ema.move_shadow_params_to_device('cuda')  # as in on_train_start
  batch_size, length = config.loader.batch_size, config.model.length
  batch = {
    'input_ids': torch.randint(0, tokenizer.vocab_size,
                               (batch_size, length), device='cuda'),
    'attention_mask': torch.ones(batch_size, length, device='cuda'),
  }
  sigma = torch.zeros(batch_size, device='cuda')
  parameters = itertools.chain(model.backbone.parameters(),
                               model.noise.parameters())
  optimizer = torch.optim.AdamW(parameters, lr=1e-5)

  def backbone_forward():
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float32):
      model.backbone(batch['input_ids'], sigma)

  def full_forward():
    with torch.no_grad():
      model.forward(batch['input_ids'], sigma)

  def train_step():
    # Lightning's precision='bf16' autocasts training_step like this.
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
      loss = model.training_step(batch, 0)
    loss.backward()
    optimizer.step()
    if model.ema:
      model.ema.update(itertools.chain(model.backbone.parameters(),
                                       model.noise.parameters()))

  print(f'--- batch {batch_size}, relay={config.relay.enabled} ---')
  print(f'backbone forward: {timed(backbone_forward) * 1e3:7.1f} ms')
  print(f'full forward:     {timed(full_forward) * 1e3:7.1f} ms')
  print(f'training step:    {timed(train_step) * 1e3:7.1f} ms '
        '(training_step + backward + AdamW + EMA)')
  print(f'peak memory:      {torch.cuda.max_memory_allocated() / 2**30:7.1f} GiB')

  # PROFILE_STEPS=N: sustained training to expose slowdowns over time
  # (allocator retries, clock throttling).
  steps = int(os.environ.get('PROFILE_STEPS', '0'))
  if steps:
    torch.cuda.synchronize()
    last = time.perf_counter()
    for i in range(1, steps + 1):
      train_step()
      if i % 25 == 0:
        torch.cuda.synchronize()
        now = time.perf_counter()
        stats = torch.cuda.memory_stats()
        print(f'steps {i - 24:4d}-{i:4d}: {(now - last) / 25 * 1e3:6.0f} ms/step'
              f'  alloc_retries={stats["num_alloc_retries"]}'
              f'  reserved={torch.cuda.memory_reserved() / 2**30:.1f} GiB',
              flush=True)
        last = now

  train_loader, _ = dataloader.get_dataloaders(
    config, tokenizer, skip_valid=True)
  iterator = iter(train_loader)
  for _ in range(3):
    next(iterator)
  start = time.perf_counter()
  for _ in range(20):
    next(iterator)
  print(f'dataloader:       {(time.perf_counter() - start) / 20 * 1e3:7.1f} '
        'ms per batch')
  print(flush=True)


if __name__ == '__main__':
  profile(sys.argv[1:])
