#!/usr/bin/env bash
# Relaunch the eight sudoku_extreme 300k ablations across multiple random seeds.
#
# Each (objective, embed_tying, seed) tuple becomes a distinct job:
#   job_name=sudoku_extreme_<objective>_300k_<tying>_seed${SEED}
# This guarantees a unique log directory and W&B run id (when use_job_name_as_id=false).
#
# Usage (from repo root):
#   ./submit_sudoku_300k_seeds_sweep.sh                  # submits 8 * 3 = 24 jobs (seeds 1 2 3)
#   DO=print ./submit_sudoku_300k_seeds_sweep.sh         # dry-run
#   SEEDS="2 3" ./submit_sudoku_300k_seeds_sweep.sh      # only seeds 2, 3 (16 jobs)
#   ONLY_OBJECTIVES="relay mlm_uniform" \
#       ./submit_sudoku_300k_seeds_sweep.sh              # filter to a subset of objectives
#   ONLY_TYINGS="untied" ./submit_sudoku_300k_seeds_sweep.sh
#
# Notes:
#   * Set SKIP_EXISTING_LOGDIR=1 to skip jobs whose log dir already exists.
#   * Cluster-specific knobs (SLURM_RESERVATION, SLURM_CONSTRAIN_*, partition) are
#     read from environment variables; defaults are unset / portable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DO="${DO:-submit}"
SEEDS="${SEEDS:-1 2 3}"
SKIP_EXISTING_LOGDIR="${SKIP_EXISTING_LOGDIR:-0}"

BATCH_SIZE="${BATCH_SIZE:-512}"
TRAIN_COMPILE="${TRAIN_COMPILE:-false}"
TRAIN_PRECISION="${TRAIN_PRECISION:-bf16-mixed}"
HARDWARE="${HARDWARE:-1_node_1_gpu}"
SLURM_TIME="${SLURM_TIME:-36:00:00}"
USE_JOB_NAME_AS_ID="${USE_JOB_NAME_AS_ID:-false}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-300000}"
TRAIN_VAL_INTERVAL="${TRAIN_VAL_INTERVAL:-5000}"
SLURM_CONSTRAIN_MLM="${SLURM_CONSTRAIN_MLM:-vram40,bf16}"
SLURM_CONSTRAIN_RELAY="${SLURM_CONSTRAIN_RELAY:-vram80,bf16}"
TAG_SWEEP="${TAG_SWEEP:-sudoku_extreme_300k_seeds}"

ONLY_OBJECTIVES="${ONLY_OBJECTIVES:-}"
ONLY_TYINGS="${ONLY_TYINGS:-}"

LOG_DIR_ROOT="${LOG_DIR_ROOT:-logs}"

# 8 ablation rows, one per (objective, tying). Each row is "|"-separated:
#   objective_tag | embed_tying | base_job_name | experiment | slurm_constrain | extra_inner_args
runs=(
  "mlm_uniform|untied|sudoku_extreme_mlm_uniform_300k_untied|sudoku_extreme_mlm_uniform|${SLURM_CONSTRAIN_MLM}|"
  "rollout|untied|sudoku_extreme_rollout_300k_untied|sudoku_extreme_relay_bptt|${SLURM_CONSTRAIN_RELAY}|model=rotary_transformer_xtiny loss.with_relay=false loss.stop_grad_h_s=true predictor.with_relay=false"
  "relay_sg|untied|sudoku_extreme_relay_sg_300k_untied|sudoku_extreme_relay_bptt|${SLURM_CONSTRAIN_RELAY}|loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true"
  "relay|untied|sudoku_extreme_relay_bptt_steps2_300k_untied|sudoku_extreme_relay_bptt|${SLURM_CONSTRAIN_RELAY}|loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true"
  "mlm_uniform|tied|sudoku_extreme_mlm_uniform_300k_tied|sudoku_extreme_mlm_uniform|${SLURM_CONSTRAIN_MLM}|++model.tie_embeddings=true"
  "rollout|tied|sudoku_extreme_rollout_300k_tied|sudoku_extreme_relay_bptt|${SLURM_CONSTRAIN_RELAY}|model=rotary_transformer_xtiny ++model.tie_embeddings=true loss.with_relay=false loss.stop_grad_h_s=true predictor.with_relay=false"
  "relay_sg|tied|sudoku_extreme_relay_sg_300k_tied|sudoku_extreme_relay_bptt|${SLURM_CONSTRAIN_RELAY}|++model.tie_embeddings=true loss.with_relay=true loss.stop_grad_h_s=true predictor.with_relay=true"
  "relay|tied|sudoku_extreme_relay_bptt_steps2_300k_tied|sudoku_extreme_relay_bptt|${SLURM_CONSTRAIN_RELAY}|++model.tie_embeddings=true loss.with_relay=true loss.stop_grad_h_s=false loss.num_steps=2 predictor.with_relay=true"
)

_in_filter() {
  local needle="$1" haystack="$2"
  if [[ -z "${haystack// }" ]]; then
    return 0
  fi
  for tok in ${haystack}; do
    if [[ "${tok}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

_submit_one() {
  local job_name=$1
  local experiment=$2
  local slurm_constrain=$3
  local embed_tying=$4
  local objective=$5
  local seed=$6
  shift 6

  if [[ "${SKIP_EXISTING_LOGDIR}" == "1" && -d "${LOG_DIR_ROOT}/${job_name}" ]]; then
    echo "[skip] log dir exists: ${LOG_DIR_ROOT}/${job_name}"
    return 0
  fi

  local extra_outer=()
  if [[ -n "${SLURM_RESERVATION}" ]]; then
    extra_outer+=("++slurm.reservation=${SLURM_RESERVATION}")
  fi

  python slurm_scripts/submit_train.py \
    "do=${DO}" \
    "job_name=${job_name}" \
    "train.experiment=${experiment}" \
    "train.batch_size=${BATCH_SIZE}" \
    "train.compile=${TRAIN_COMPILE}" \
    "train.precision=${TRAIN_PRECISION}" \
    "hardware=${HARDWARE}" \
    "slurm.constrain=\"${slurm_constrain}\"" \
    "slurm.time=${SLURM_TIME}" \
    "use_job_name_as_id=${USE_JOB_NAME_AS_ID}" \
    "${extra_outer[@]}" \
    "---" \
    "seed=${seed}" \
    "trainer.max_steps=${TRAIN_MAX_STEPS}" \
    "trainer.val_check_interval=${TRAIN_VAL_INTERVAL}" \
    "+tags.sweep=${TAG_SWEEP}" \
    "+tags.embed_tying=${embed_tying}" \
    "+tags.objective=${objective}" \
    "+tags.seed=${seed}" \
    "$@"
}

submitted=0
skipped=0
for seed in ${SEEDS}; do
  for entry in "${runs[@]}"; do
    IFS='|' read -r objective tying base_job experiment constrain extra <<< "${entry}"
    if ! _in_filter "${objective}" "${ONLY_OBJECTIVES}"; then
      skipped=$((skipped + 1)); continue
    fi
    if ! _in_filter "${tying}" "${ONLY_TYINGS}"; then
      skipped=$((skipped + 1)); continue
    fi
    job_name="${base_job}_seed${seed}"
    extra_args=()
    if [[ -n "${extra// }" ]]; then
      read -r -a extra_args <<< "${extra}"
    fi
    submitted=$((submitted + 1))
    echo "[${submitted}] ${job_name}  (seed=${seed} obj=${objective} tying=${tying})"
    _submit_one "${job_name}" "${experiment}" "${constrain}" "${tying}" "${objective}" "${seed}" "${extra_args[@]}"
  done
done

echo ""
echo "Done. submitted=${submitted}  skipped=${skipped}  (DO=${DO}, SEEDS='${SEEDS}')"
