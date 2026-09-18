import copy
import itertools
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Forward calls per sample batch:', model.nfe)
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _parity_check(config, logger, tokenizer):
  """Checks the local backbone against the Hugging Face reference.

  Loads `training.init_from_pretrained` into `backbone=dit` and
  compares its predictions with the reference implementation of the
  same checkpoint (`backbone=hf_dit`, loaded from
  `eval.checkpoint_path`, which defaults to the same repo). With the
  relay enabled it also checks that a freshly initialized relay
  contributes exactly nothing, and that the EMA tracks the loaded
  weights.
  """
  logger.info('Starting parity check.')
  assert config.backbone == 'dit', (
    'parity_check compares backbone=dit against hf_dit.')
  pretrained = config.training.init_from_pretrained
  assert pretrained, (
    'Set training.init_from_pretrained, e.g. kuleshov-group/mdlm-owt.')
  reference_path = config.eval.checkpoint_path or pretrained

  model = diffusion.Diffusion(config, tokenizer=tokenizer)
  missing, _ = model.load_pretrained_backbone(pretrained)
  logger.info(f'Loaded {pretrained}; keys left at init: {missing}')
  model = model.to('cuda').eval()

  reference_config = copy.deepcopy(config)
  reference_config.backbone = 'hf_dit'
  reference_config.eval.checkpoint_path = reference_path
  reference_config.relay.enabled = False
  reference = diffusion.Diffusion(
    reference_config, tokenizer=tokenizer).to('cuda').eval()

  torch.manual_seed(config.seed)
  batch_size, length = 2, config.model.length
  x0 = torch.randint(0, tokenizer.vocab_size, (batch_size, length),
                     device='cuda')
  masked = torch.rand(batch_size, length, device='cuda') < 0.5
  x = torch.where(masked, model.mask_index, x0)
  sigma = torch.zeros(batch_size, device='cuda')
  with torch.no_grad():
    reference_output = reference.forward(x, sigma)
    output = model.forward(x, sigma)
  max_diff = (output - reference_output).abs().max().item()
  logger.info(f'dit vs hf_dit log-probs: max |diff| = {max_diff:.3e}')
  torch.testing.assert_close(output, reference_output,
                             atol=1e-4, rtol=1e-4)

  if model.relay_enabled:
    h_prev = torch.randn(batch_size, length, config.model.hidden_size,
                         device='cuda')
    with torch.no_grad():
      output_relay, h_next = model.forward(
        x, sigma, h_prev=h_prev, return_state=True)
    max_diff = (output_relay - output).abs().max().item()
    logger.info('zero-init relay with a random state: '
                f'max |diff| = {max_diff:.3e}, '
                f'h_next shape {tuple(h_next.shape)}')
    assert max_diff <= 1e-6, 'relay is not a no-op at initialization'
    assert h_next.shape == (batch_size, length,
                            config.model.hidden_size)
    assert torch.isfinite(h_next).all()

  if model.ema is not None:
    params = [p for p in itertools.chain(model.backbone.parameters(),
                                         model.noise.parameters())
              if p.requires_grad]
    assert len(params) == len(model.ema.shadow_params)
    assert all(torch.equal(s.to(p.device), p)
               for s, p in zip(model.ema.shadow_params, params)), (
      'EMA shadow parameters do not match the loaded weights.')
    logger.info('EMA shadow parameters match the loaded weights.')
  logger.info('Parity check passed.')


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  if config.get('wandb', None) is not None:
    trainer_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  else:
    # Without W&B (`wandb=null`), keep the metrics in <run dir>/csv_logs.
    trainer_logger = L.pytorch.loggers.CSVLogger(
      os.getcwd(), name='csv_logs')

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)
  if config.training.init_from_pretrained:
    if ckpt_path is None:
      logger.info('Initializing backbone from '
                  f'{config.training.init_from_pretrained}.')
      model.load_pretrained_backbone(
        config.training.init_from_pretrained)
    else:
      logger.info(f'Resuming from {ckpt_path}; '
                  'ignoring training.init_from_pretrained.')

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=trainer_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  elif config.mode == 'parity_check':
    _parity_check(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()