#!/bin/bash
#
# Submit Math500/GSM8K/HumanEval+/MBPP+ eval jobs for selected checkpoints.
#
# This is a lightweight Slurm wrapper that runs after a training job and then
# calls scripts/submit_eval.py once per task/checkpoint. It does not run evals
# itself; each eval gets its own one-GPU Slurm job from submit_eval.py.
#
# Required/typical overrides:
#   MODEL_ROOT=output_models/opencode_openmath_60k_1p5B/...
#   USE_CARRY=0|1
#
#SBATCH --job-name=submit_mix_eval_sweep
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=$USER

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT=${SLURM_SUBMIT_DIR}
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-relay}
CONDA_ROOT=${CONDA_ROOT:-${HOME}/miniconda3}
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

model_root=${MODEL_ROOT:-output_models/opencode_openmath_60k_1p5B/vanilla_nopack2048_bs2x16x2_ep3_lr5e-6}
use_carry=${USE_CARRY:-0}
threshold=${THRESHOLD:-0.85}
tasks_csv=${TASKS:-math500,gsm8k,humaneval_plus_evalplus,mbpp_plus_evalplus}
checkpoints_csv=${CHECKPOINTS:-300,600,900,1200,1800,final}
output_root=${EVAL_OUTPUT_ROOT:-eval_results2}
wandb_tags=${WANDB_TAGS_EXTRA:-opencode_openmath_60k,mixed_sweep}
evalplus_tokenizer=${EVALPLUS_TOKENIZER:-}

if [ ! -d "${model_root}" ]; then
    echo "ERROR: MODEL_ROOT does not exist: ${model_root}" >&2
    exit 1
fi

IFS=',' read -r -a tasks <<< "${tasks_csv}"
IFS=',' read -r -a checkpoints <<< "${checkpoints_csv}"

carry_args=()
if [ "${use_carry}" = "1" ]; then
    carry_args=(--use_carry)
fi

echo "MODEL_ROOT=${model_root}"
echo "USE_CARRY=${use_carry}"
echo "TASKS=${tasks_csv}"
echo "CHECKPOINTS=${checkpoints_csv}"

for ck in "${checkpoints[@]}"; do
    ck=$(echo "${ck}" | xargs)
    if [ "${ck}" = "final" ]; then
        model_path="${model_root}"
    else
        model_path="${model_root}/checkpoint-${ck}"
    fi
    if [ ! -f "${model_path}/config.json" ]; then
        echo "Skipping missing checkpoint/model: ${model_path}"
        continue
    fi

    for task in "${tasks[@]}"; do
        task=$(echo "${task}" | xargs)
        cmd=(
            python scripts/submit_eval.py
            --task "${task}"
            --model_path "${model_path}"
            --threshold "${threshold}"
            --output_root "${output_root}"
            --wandb_tags "${wandb_tags}"
            "${carry_args[@]}"
        )
        if [[ "${task}" == humaneval_plus_evalplus || "${task}" == mbpp_plus_evalplus ]]; then
            if [ -n "${evalplus_tokenizer}" ]; then
                cmd+=(--evalplus_tokenizer "${evalplus_tokenizer}")
            else
                cmd+=(--evalplus_tokenizer "${model_path}")
            fi
        fi
        echo "Submitting: ${cmd[*]}"
        "${cmd[@]}"
    done
done
