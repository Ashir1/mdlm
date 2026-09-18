"""Tokenizes and caches the datasets of a config into `data.cache_dir`.

Makes the same `dataloader.get_dataset` calls as
`dataloader.get_dataloaders` would on first use, so that training jobs
find the cached `.dat` directories. Needs no GPU:

  MDLM_MODAL_GPU=none MDLM_MODAL_CPU=16 MDLM_MODAL_MEMORY_GB=64 \
  modal run --detach modal_app.py --cmd "PYTHONPATH=. python scripts/prepare_data.py \
    data=openwebtext-split data.cache_dir=/root/outputs/data model.length=1024"
"""
import sys

import hydra

import dataloader


def prepare(overrides):
  import main  # noqa: F401  registers the OmegaConf resolvers
  with hydra.initialize(version_base=None, config_path='../configs'):
    config = hydra.compose(config_name='config', overrides=overrides)
  tokenizer = dataloader.get_tokenizer(config)
  train = dataloader.get_dataset(
    config.data.train,
    tokenizer,
    mode='train',
    wrap=config.data.wrap,
    cache_dir=config.data.cache_dir,
    block_size=config.model.length)
  print(f'{config.data.train}: {len(train)} training blocks', flush=True)
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  valid = dataloader.get_dataset(
    config.data.valid,
    tokenizer,
    wrap=config.data.wrap,
    mode=validation_split,
    cache_dir=config.data.cache_dir,
    block_size=config.model.length,
    streaming=False)
  print(f'{config.data.valid}: {len(valid)} validation blocks', flush=True)


if __name__ == '__main__':
  prepare(sys.argv[1:])
