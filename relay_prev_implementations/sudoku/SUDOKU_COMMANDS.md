# Sudoku Extreme 300k training commands

All commands train `sudoku_extreme` for 300k steps and validate every 5k steps.
The four objectives map directly to Table 1 of the paper:

| W&B `objective` tag | What it is                                     | Hydra overrides                                                                                                |
|---------------------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `mlm_uniform`       | Mask-uniform CE baseline                       | (uses `sudoku_extreme_mlm_uniform.yaml` directly)                                                              |
| `rollout`           | Rollout-buffer-only (no relay carry)           | `loss.with_relay=false  loss.stop_grad_h_s=true  predictor.with_relay=false  model=rotary_transformer_xtiny` |
| `relay_sg`          | Relay rollout with stop-gradient on `h`        | `loss.with_relay=true   loss.stop_grad_h_s=true   predictor.with_relay=true`                                   |
| `relay`             | Relay rollout with truncated BPTT (`T=2`)      | `loss.with_relay=true   loss.stop_grad_h_s=false  loss.num_steps=2  predictor.with_relay=true`                 |

Each objective is run in two weight-tying conditions (`+tags.embed_tying=tied`
adds `++model.tie_embeddings=true`; `untied` is the default), giving the eight
runs of Table 1.

## Batch submission (recommended)

```bash
DO=print  ./submit_sudoku_300k_sweep.sh        # dry-run, prints sbatch arguments
./submit_sudoku_300k_sweep.sh                  # submit the eight runs (single seed)
```

To run the same eight ablations across multiple seeds (paper uses seeds 1, 2, 3):

```bash
DO=print ./submit_sudoku_300k_seeds_sweep.sh   # dry-run all 24 jobs
./submit_sudoku_300k_seeds_sweep.sh            # submit all 24 jobs
```

Both scripts read cluster-specific knobs from environment variables:

| Variable                | Default              | Purpose                                                            |
|-------------------------|----------------------|--------------------------------------------------------------------|
| `DO`                    | `submit`             | `print` for dry-run; `submit` actually `sbatch`'es                  |
| `SLURM_RESERVATION`     | (unset)              | If set, adds `++slurm.reservation=$VAL`                             |
| `SLURM_CONSTRAIN_MLM`   | `vram40,bf16`        | GPU constraint for the MLM-uniform job                              |
| `SLURM_CONSTRAIN_RELAY` | `vram80,bf16`        | GPU constraint for the rollout / relay_sg / relay jobs              |
| `TRAIN_MAX_STEPS`       | `300000`             | Total training steps                                                |
| `TRAIN_VAL_INTERVAL`    | `5000`               | Validation interval                                                 |
| `BATCH_SIZE`            | `512`                | Per-device batch size                                               |
| `SLURM_TIME`            | `36:00:00`           | Wall-clock limit                                                    |
| `TAG_SWEEP`             | `sudoku_extreme_300k`| W&B `sweep=…` tag for filtering                                     |
| `SEEDS`                 | `1 2 3`              | (seeds sweep only) whitespace-separated seed list                   |
| `ONLY_OBJECTIVES`       | (all four)           | (seeds sweep only) filter to a subset                               |
| `ONLY_TYINGS`           | (`untied tied`)      | (seeds sweep only) filter weight-tying conditions                   |
| `SKIP_EXISTING_LOGDIR`  | `0`                  | (seeds sweep only) skip jobs whose `logs/<job_name>/` already exists|

## Validation metrics (logged every `val_check_interval`)

- `val/prediction/exact_match` — fraction of boards predicted exactly correctly.
- `val/prediction/token_accuracy` — fraction of masked cells filled correctly.
- `val/prediction/rollout_steps` — mean number of forward evaluations (NFE) until the predictor terminates.
- `val/prediction/legal_rate` — fraction of boards whose final 81-cell prediction is fully filled AND has no Sudoku constraint violation. Distinguishes "wrong but plausible" from "completely broken"; flags divergence in the very first validation epoch.
- `val_sweep/t_{0p05,0p10,0p15,0p20,0p25}/{exact_match, token_accuracy, rollout_steps, legal_rate}` — same metrics computed at five inference confidence thresholds.
- The same suite is also reported under `test/prediction/*`.

## Quick local smoke test (no SLURM, single GPU)

The same code runs without the SLURM wrapper. Reduce steps to confirm the
loop is wired up before launching the full 300k:

```bash
cd relay/sudoku
source .venv_relay/bin/activate
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

Replace `experiment=sudoku_extreme_relay_bptt` with `sudoku_extreme_mlm_uniform`
to smoke-test the baseline. See `submit_sudoku_300k_sweep.sh` for the full set
of `loss.*` / `predictor.*` overrides that toggle between objectives.

## Paper Table 1 numbers

After all 24 (8 × 3 seeds) runs finish, the per-step Table 1 numbers are
the validation-set metrics logged to W&B (or to local TensorBoard logs).
The relevant columns map to the metric keys logged every
`val_check_interval`:

- **Exact-match accuracy** → `val/prediction/exact_match`
- **Token accuracy** → `val/prediction/token_accuracy`
- **Mean NFE** → `val/prediction/rollout_steps`
- **Legal-rate** (Sudoku constraint check, used to flag divergence) →
  `val/prediction/legal_rate`

Group by the `+tags.{objective, embed_tying, seed}` tags to recover the
eight rows × three seeds; average over seeds.
