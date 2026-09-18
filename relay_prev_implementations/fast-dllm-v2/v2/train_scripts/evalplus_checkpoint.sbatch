#!/bin/bash
#
# EvalPlus checkpoint evaluation for Magicoder training runs. Intended to be
# submitted by EvalPlusOnSaveCallback after a Trainer checkpoint is written.
#
# Required env:
#   MODEL_PATH=/path/to/checkpoint-N
#   TRAIN_RUN_NAME=<training run name>
#
# Optional env:
#   CHECKPOINT_STEP=<int>
#   EVALPLUS_DATASETS="humaneval mbpp"
#   EVALPLUS_SEEDS="0"
#   EVALPLUS_THRESHOLD=0.85
#   EVALPLUS_USE_CARRY=0|1
#   EVALPLUS_BLOCK_CACHE=auto|on|off
#   EVALPLUS_OUT_DIR=evalplus_results/on_save/<run>/<checkpoint-N>
#   EVALPLUS_FORCE=0|1
#   EVALPLUS_NO_WANDB=0|1
#
#SBATCH --job-name=evalplus_checkpoint
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint=vram80,bf16,ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x/%j.out
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=$USER

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [ ! -f "${REPO_ROOT}/scripts/generate_evalplus_jsonl.py" ]; then
    echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like Fast-dLLM/v2." >&2
    exit 1
fi
cd "${REPO_ROOT}"

CONDA_ENV=${EVALPLUS_CONDA_ENV:-relay}
CONDA_ROOT=${CONDA_ROOT:-${HOME}/miniconda3}
mkdir -p "${REPO_ROOT}/logs/${SLURM_JOB_NAME:-evalplus_checkpoint}"

if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load cuda/12.1 2>/dev/null \
        || module load cuda/12.4 2>/dev/null \
        || module load cuda/12.6 2>/dev/null \
        || module load cuda 2>/dev/null \
        || true
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TRAIN_RUN_NAME:?TRAIN_RUN_NAME is required}"

checkpoint_label=$(basename "${MODEL_PATH}")
checkpoint_step=${CHECKPOINT_STEP:-}
evalplus_out_dir=${EVALPLUS_OUT_DIR:-"${REPO_ROOT}/evalplus_results/on_save/${TRAIN_RUN_NAME}/${checkpoint_label}"}

cmd=(python scripts/run_evalplus_checkpoint.py
    --model_path "${MODEL_PATH}"
    --run_name "${TRAIN_RUN_NAME}"
    --datasets "${EVALPLUS_DATASETS:-humaneval mbpp}"
    --seeds "${EVALPLUS_SEEDS:-0}"
    --threshold "${EVALPLUS_THRESHOLD:-0.85}"
    --batch_size "${EVALPLUS_BATCH_SIZE:-4}"
    --max_new_tokens "${EVALPLUS_MAX_NEW_TOKENS:-512}"
    --bd_size "${EVALPLUS_BD_SIZE:-32}"
    --small_block_size "${EVALPLUS_SMALL_BLOCK_SIZE:-8}"
    --block_cache_mode "${EVALPLUS_BLOCK_CACHE:-auto}"
    --out_dir "${evalplus_out_dir}"
    --wandb_project "${EVALPLUS_WANDB_PROJECT:-${WANDB_PROJECT:-}}"
    --wandb_entity "${EVALPLUS_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
    --wandb_group "${EVALPLUS_WANDB_GROUP:-${TRAIN_RUN_NAME}}"
    --wandb_name "${EVALPLUS_WANDB_NAME:-evalplus__${TRAIN_RUN_NAME}__${checkpoint_label}}"
    --wandb_tags "${EVALPLUS_WANDB_TAGS:-${WANDB_TAGS:-}}"
)

if [ -n "${checkpoint_step}" ]; then
    cmd+=(--checkpoint_step "${checkpoint_step}")
fi
if [ "${EVALPLUS_USE_CARRY:-0}" = "1" ]; then
    cmd+=(--use_carry)
fi
if [ "${EVALPLUS_FORCE:-0}" = "1" ]; then
    cmd+=(--force)
fi
if [ "${EVALPLUS_NO_WANDB:-0}" = "1" ]; then
    cmd+=(--no_wandb)
fi
if [ "${EVALPLUS_DRY_RUN:-0}" = "1" ]; then
    cmd+=(--dry_run)
fi

echo "===== EVALPLUS CHECKPOINT JOB ====="
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID:-<not in slurm>}"
echo "MODEL_PATH         = ${MODEL_PATH}"
echo "TRAIN_RUN_NAME     = ${TRAIN_RUN_NAME}"
echo "CHECKPOINT_STEP    = ${checkpoint_step:-<none>}"
echo "EVALPLUS_OUT_DIR   = ${evalplus_out_dir}"
echo "EVALPLUS_DATASETS  = ${EVALPLUS_DATASETS:-humaneval mbpp}"
echo "EVALPLUS_SEEDS     = ${EVALPLUS_SEEDS:-0}"
echo "EVALPLUS_THRESHOLD = ${EVALPLUS_THRESHOLD:-0.85}"
echo "EVALPLUS_USE_CARRY = ${EVALPLUS_USE_CARRY:-0}"
echo "EVALPLUS_BLOCK_CACHE = ${EVALPLUS_BLOCK_CACHE:-auto}"
echo "===== COMMAND ====="
printf '%q ' "${cmd[@]}"
echo
echo "==================="
"${cmd[@]}"
