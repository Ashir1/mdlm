# relay / sudoku

Sudoku Extreme experiments from the *Learned Relay Representations* paper
(Table 1, paper §4.1). Code paths kept here are the ones that produce the
reported numbers: Mask-uniform CE, Rollout-buffer-only, Relay (BPTT), and
Relay-sg.

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd relay/sudoku
```

Submodules:

- `xlm-core` — the XLM training harness (Hydra + PyTorch Lightning) used by
  every experiment YAML.
- `slurm_scripts` — SLURM submission helpers used by the `submit_*.sh` wrappers.

To pull updates and keep submodules in sync:

```bash
git pull --recurse-submodules
git submodule update --remote xlm-core
```

### Python environment

Create a virtual environment and install everything in editable mode:

```bash
python -m venv .venv_relay
source .venv_relay/bin/activate
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -e .
```

Then create a `.env` file at the `relay/sudoku/` root (used by Hydra +
`xlm-core`):

```bash
WANDB_ENTITY=<your-wandb-entity>
WANDB_PROJECT=BPTT-sudoku
DATA_DIR=data
HF_HOME=hf_home
HF_DATASETS_CACHE=hf_datasets_cache
LOG_DIR=logs
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=.
HYDRA_FULL_ERROR=1
OC_CAUSE=1
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
```

`xlm_models.json` at this directory tells Hydra where to find the `relay/`
config tree — leave it as is.

## Data

Training and evaluation use a curated dataset that joins the Sudoku
Extreme puzzles from [`sapientinc/sudoku-extreme`](https://huggingface.co/datasets/sapientinc/sudoku-extreme)
with the deduction-step trajectories from
[`timvink/sudoku-solver`](https://huggingface.co/datasets/timvink/sudoku-solver).
The exact derived dataset that produces the Table 1 numbers will be
released on the Hub upon paper acceptance; the Hub identifier is withheld
during the double-blind review period.

To use the dataset in this anonymized release, set the environment
variable `RELAY_SUDOKU_HF_DATASET` to either (a) your own re-upload of the
joined dataset, or (b) a local path to a `datasets`-loadable directory:

```bash
export RELAY_SUDOKU_HF_DATASET=<your-hf-user>/sudoku-extreme-deduction
xlm "job_type=prepare_data" "job_name=sudoku_extreme_prepare_data" "experiment=sudoku_extreme_mlm_uniform"
```

Hydra configs (`relay/configs/datasets/sudoku_extreme_*.yaml`) read this
variable via `oc.env` and fall back to the `<anonymous-hf-id>/...`
placeholder if it is unset, which will cause the data loader to fail
loudly at first use.

## Smoke test (no SLURM)

A single-GPU sanity run to confirm the loop is wired up before launching the
full 300k training:

```bash
export PROJECT_ROOT="$PWD"
python -m xlm.train \
  experiment=sudoku_extreme_relay_bptt \
  trainer.max_steps=200 \
  trainer.val_check_interval=100 \
  trainer.limit_val_batches=2 \
  per_device_batch_size=8 \
  global_batch_size=8 \
  loggers.wandb=null
```

Substitute `experiment=sudoku_extreme_mlm_uniform` for the baseline.

## Reproducing Table 1

```bash
# Dry-run first to inspect the SLURM submission for your cluster.
DO=print ./submit_sudoku_300k_sweep.sh

# Submit the eight reported runs (single seed; ~36h on 80GB GPUs for relay).
./submit_sudoku_300k_sweep.sh

# Multi-seed sweep (paper averages over seeds 1, 2, 3 -> 24 runs).
./submit_sudoku_300k_seeds_sweep.sh
```

Cluster-specific knobs (`SLURM_RESERVATION`, `SLURM_CONSTRAIN_*`, partition,
wall-clock) are environment variables; see [`SUDOKU_COMMANDS.md`](SUDOKU_COMMANDS.md)
for the full table and the per-objective Hydra overrides.

The Table 1 numbers (exact-match accuracy, token accuracy, mean NFE,
legal rate) are the validation-set metrics logged every
`val_check_interval` to W&B; see the metric-name list in
[`SUDOKU_COMMANDS.md`](SUDOKU_COMMANDS.md).

## Layout

| Path                                | Contents                                                                                  |
|-------------------------------------|-------------------------------------------------------------------------------------------|
| `relay/`                            | The Sudoku training package (model, loss, predictor, datamodule, metrics, sudoku tools). |
| `relay/configs/`                    | Hydra configs (`experiment/`, `model/`, `model_type/`, `datamodule/`, `metrics/`, ...).   |
| `xlm-core/` (submodule)             | The XLM training harness used by every experiment.                                        |
| `slurm_scripts/` (submodule)        | SLURM submission helpers used by the `submit_*.sh` wrappers.                              |
| `submit_sudoku_300k_sweep.sh`       | Submits the 8 reported Sudoku runs (single seed).                                         |
| `submit_sudoku_300k_seeds_sweep.sh` | Same 8 ablations × N seeds.                                                              |
| `SUDOKU_COMMANDS.md`                | Per-objective Hydra overrides + sweep environment variables.                              |

## W&B

Reported runs live under `https://wandb.ai/<your-wandb-entity>/BPTT-sudoku`
(the authors' entity is withheld for double-blind review). The relevant
filters are `+tags.sweep`, `+tags.objective`, `+tags.embed_tying`,
`+tags.seed`. Set `WANDB_ENTITY` in `.env` to point logging at your own
entity, or pass `loggers.wandb=null` on the Hydra command line to disable
logging entirely.
