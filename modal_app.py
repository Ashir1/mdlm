"""Runs `main.py` (or any command) on a Modal GPU.

Usage, from the repository root:

  modal run modal_app.py --args "mode=parity_check backbone=dit \
    model=small data=openwebtext-split model.length=1024 \
    relay.enabled=true training.init_from_pretrained=kuleshov-group/mdlm-owt \
    hydra.run.dir=/root/outputs/parity_check"

  modal run modal_app.py --cmd "python -m pytest -x -q tests"

`--args` is passed to `python main.py` verbatim; `--cmd` runs a shell
command in the repository instead. The Hugging Face cache and
`/root/outputs` live on persistent volumes, so checkpoints, tokenizers
and Hydra run directories survive across runs. Set MDLM_MODAL_GPU
(default A10G; `none` for a CPU-only job) to pick the GPU (flash-attn
needs an Ampere or newer card), MDLM_MODAL_CPU / MDLM_MODAL_MEMORY_GB
(defaults 4 / 16) for the reserved cores and memory. Long jobs:
`modal run --detach ...` keeps running if the client disconnects.
"""
import os
import shutil
import subprocess

import modal

REMOTE_ROOT = '/root/mdlm'
HF_CACHE = '/root/.cache/huggingface'
OUTPUTS = '/root/outputs'
GPU = os.environ.get('MDLM_MODAL_GPU', 'A10G')
if GPU.lower() in ('', 'none'):
  GPU = None  # CPU-only job, e.g. dataset preparation
# Reserved CPU cores / memory are billed; training needs little of
# either, dataset tokenization (one process per core) wants more.
CPU = int(os.environ.get('MDLM_MODAL_CPU', '4'))
MEMORY_GB = int(os.environ.get('MDLM_MODAL_MEMORY_GB', '16'))

# Prebuilt CUDA extensions matching torch 2.2 / CUDA 12 / Python 3.10
# (the versions pinned in requirements.yaml).
_WHEEL = '+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
CUDA_WHEELS = [
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/'
  f'flash_attn-2.5.6{_WHEEL}',
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/'
  f'v1.1.3.post1/causal_conv1d-1.1.3.post1{_WHEEL}',
  'https://github.com/state-spaces/mamba/releases/download/v1.1.4/'
  f'mamba_ssm-1.1.4{_WHEEL}',
]

image = (
  modal.Image.debian_slim(python_version='3.10')
  .apt_install('git')
  .pip_install('torch==2.2.2', 'torchvision==0.17.2',
               index_url='https://download.pytorch.org/whl/cu121')
  .pip_install(
    'numpy<2',
    'setuptools<80',  # lightning 2.2.1 imports pkg_resources
    'lightning==2.2.1',
    'hydra-core==1.3.2',
    'omegaconf==2.3.0',
    'transformers==4.38.2',
    'datasets==2.18.0',
    'einops==0.7.0',
    'timm==0.9.16',
    'torchmetrics',
    'safetensors',
    'fsspec==2024.2.0',
    'rich==13.7.1',
    'requests',
    'wandb',
    'pytest')
  # --no-deps: their runtime deps (torch, einops, triton, packaging,
  # transformers) are pinned above; mamba_ssm's metadata would
  # otherwise pull a newer transformers.
  .pip_install(*CUDA_WHEELS, extra_options='--no-deps')
  .env({'HF_HOME': HF_CACHE,
        'HYDRA_FULL_ERROR': '1',
        'PYTHONUNBUFFERED': '1'})
  .add_local_dir(
    os.path.dirname(os.path.abspath(__file__)),
    remote_path=REMOTE_ROOT,
    ignore=['.git/**', 'relay_prev_implementations/**', 'outputs/**',
            'watch_folder/**', '**/__pycache__/**', '*.ckpt',
            '*.safetensors']))

app = modal.App('mdlm-relay')
hf_cache = modal.Volume.from_name('mdlm-relay-hf-cache',
                                  create_if_missing=True)
outputs = modal.Volume.from_name('mdlm-relay-outputs',
                                 create_if_missing=True)


@app.function(image=image,
              gpu=GPU,
              cpu=CPU,
              memory=MEMORY_GB * 1024,
              timeout=24 * 60 * 60,
              volumes={HF_CACHE: hf_cache, OUTPUTS: outputs})
def run(command: str) -> None:
  if shutil.which('nvidia-smi'):  # absent in CPU-only containers
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                    '--format=csv'], check=False)
  try:
    subprocess.run(command, shell=True, cwd=REMOTE_ROOT, check=True)
  finally:
    hf_cache.commit()
    outputs.commit()


@app.local_entrypoint()
def main(args: str = '', cmd: str = '') -> None:
  assert bool(args) != bool(cmd), 'pass exactly one of --args and --cmd'
  run.remote(cmd or f'python main.py {args}')
