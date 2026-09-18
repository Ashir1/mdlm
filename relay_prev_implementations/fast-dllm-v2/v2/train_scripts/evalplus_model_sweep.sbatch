#!/bin/bash
#
# Sweep EvalPlus over one model family: a Hub/local model path, or every
# checkpoint-* under a training output directory.
#
# Required env:
#   MODEL_LABEL=<short label used in output filenames>
#   MODEL_PATH=<Hub id or local model path>       # for one model
#     or
#   MODEL_ROOT=<training output dir>              # for checkpoint sweep
#
#SBATCH --job-name=evalplus_model_sweep
#SBATCH --partition=gpu,gpu-preempt,superpod-a100
#SBATCH --constraint=vram80,bf16,ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=$USER

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "${REPO_ROOT}"
mkdir -p logs evalplus_results/magicoder_full_sweep

CONDA_ENV=${EVALPLUS_CONDA_ENV:-relay}
CONDA_ROOT=${CONDA_ROOT:-${HOME}/miniconda3}
if [ "${EVALPLUS_SKIP_CONDA:-0}" != "1" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1

: "${MODEL_LABEL:?MODEL_LABEL is required}"

EVALPLUS_DATASETS=${EVALPLUS_DATASETS:-"humaneval mbpp"}
EVALPLUS_SEEDS=${EVALPLUS_SEEDS:-"0 1 2"}
EVALPLUS_THRESHOLD=${EVALPLUS_THRESHOLD:-0.85}
EVALPLUS_USE_CARRY=${EVALPLUS_USE_CARRY:-0}
EVALPLUS_BLOCK_CACHE=${EVALPLUS_BLOCK_CACHE:-auto}
EVALPLUS_FORCE=${EVALPLUS_FORCE:-0}
EVALPLUS_NO_WANDB=${EVALPLUS_NO_WANDB:-0}
EVALPLUS_DRY_RUN=${EVALPLUS_DRY_RUN:-0}
ONLY_CHECKPOINTS=${ONLY_CHECKPOINTS:-0}

list_model_paths() {
    local d="$1" p bn n
    local -a cks=()
    if [ -d "$d" ]; then
        mapfile -t cks < <(
            find "$d" -mindepth 1 -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | while IFS= read -r p; do
                bn=$(basename "$p")
                n="${bn#checkpoint-}"
                if [[ "$n" =~ ^[0-9]+$ ]]; then
                    printf '%d\t%s\n' "$n" "$p"
                fi
            done | sort -n -k1,1 | cut -f2-
        )
    fi
    if ((${#cks[@]} > 0)); then
        printf '%s\n' "${cks[@]}"
    elif [ "${ONLY_CHECKPOINTS}" != "1" ] && [ -n "${MODEL_PATH:-}" ]; then
        printf '%s\n' "${MODEL_PATH}"
    elif [ "${ONLY_CHECKPOINTS}" != "1" ] && [ -n "${MODEL_ROOT:-}" ] && [ -f "${MODEL_ROOT}/config.json" ]; then
        printf '%s\n' "${MODEL_ROOT}"
    fi
}

if [ -n "${MODEL_ROOT:-}" ]; then
    mapfile -t model_paths < <(list_model_paths "${MODEL_ROOT}")
else
    : "${MODEL_PATH:?MODEL_PATH or MODEL_ROOT is required}"
    model_paths=("${MODEL_PATH}")
fi

if ((${#model_paths[@]} == 0)); then
    echo "ERROR: no model paths found for MODEL_LABEL=${MODEL_LABEL}" >&2
    echo "MODEL_ROOT=${MODEL_ROOT:-<unset>} MODEL_PATH=${MODEL_PATH:-<unset>} ONLY_CHECKPOINTS=${ONLY_CHECKPOINTS}" >&2
    exit 1
fi

echo "===== EVALPLUS MODEL SWEEP ====="
echo "MODEL_LABEL          = ${MODEL_LABEL}"
echo "MODEL_ROOT           = ${MODEL_ROOT:-<unset>}"
echo "MODEL_PATH           = ${MODEL_PATH:-<unset>}"
echo "EVALPLUS_DATASETS    = ${EVALPLUS_DATASETS}"
echo "EVALPLUS_SEEDS       = ${EVALPLUS_SEEDS}"
echo "EVALPLUS_THRESHOLD   = ${EVALPLUS_THRESHOLD}"
echo "EVALPLUS_USE_CARRY   = ${EVALPLUS_USE_CARRY}"
echo "EVALPLUS_BLOCK_CACHE = ${EVALPLUS_BLOCK_CACHE}"
echo "paths:"
printf '  %s\n' "${model_paths[@]}"
echo "==============================="

for model_path in "${model_paths[@]}"; do
    checkpoint_label=$(basename "${model_path}")
    out_dir="${REPO_ROOT}/evalplus_results/magicoder_full_sweep/${MODEL_LABEL}/${checkpoint_label}"
    cmd=(python scripts/run_evalplus_checkpoint.py
        --model_path "${model_path}"
        --run_name "${MODEL_LABEL}"
        --datasets "${EVALPLUS_DATASETS}"
        --seeds "${EVALPLUS_SEEDS}"
        --threshold "${EVALPLUS_THRESHOLD}"
        --block_cache_mode "${EVALPLUS_BLOCK_CACHE}"
        --out_dir "${out_dir}"
        --wandb_project "${EVALPLUS_WANDB_PROJECT:-${WANDB_PROJECT:-huggingface}}"
        --wandb_entity "${EVALPLUS_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
        --wandb_group "${EVALPLUS_WANDB_GROUP:-magicoder_full_evalplus_sweep}"
        --wandb_name "evalplus__${MODEL_LABEL}__${checkpoint_label}"
        --wandb_tags "evalplus,magicoder_oss,full,${MODEL_LABEL}"
    )
    if [ "${EVALPLUS_USE_CARRY}" = "1" ]; then
        cmd+=(--use_carry)
    fi
    if [ "${EVALPLUS_FORCE}" = "1" ]; then
        cmd+=(--force)
    fi
    if [ "${EVALPLUS_NO_WANDB}" = "1" ]; then
        cmd+=(--no_wandb)
    fi
    if [ "${EVALPLUS_DRY_RUN}" = "1" ]; then
        cmd+=(--dry_run)
    fi
    echo "===== RUN ${MODEL_LABEL}/${checkpoint_label} ====="
    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
done
