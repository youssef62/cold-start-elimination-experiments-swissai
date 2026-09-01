#!/bin/bash
# Build the artifact via `servekit prepare` (shards the checkpoint and
# warms/persists the JIT cache).
#
#SBATCH --job-name=servekit-jit-prepare
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
SERVEKIT_DIR="${SERVEKIT_DIR:-/iopsstor/scratch/cscs/yboughizane/cold-start-elimination/servekit-main}"
EDF="${SLURM_SUBMIT_DIR}/../servekit-eval/scripts/sglang.toml"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded-fa3}"
MODEL_PATH=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

echo "job=${SLURM_JOB_ID} node=$(hostname) arm=prepare artifact=${ARTIFACT_ROOT}"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" bash -c "
set -euo pipefail
unset BASH_ENV
pip install -e '${SERVEKIT_DIR}' --no-build-isolation --no-deps -q
servekit launch --servekit-artifact-path '${ARTIFACT_ROOT}' --only-prepare -- \
  python -m sglang.launch_server \
    --model-path '${MODEL_PATH}' \
    --served-model-name '${SERVED_MODEL_NAME}' \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port 8080 \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --random-seed 42 \
    --weight-loader-disable-mmap \
    --attention-backend flashinfer \
    --load-format sharded_state
"

echo "prepared ${ARTIFACT_ROOT}; JIT cache files:"
find "${ARTIFACT_ROOT}/servekit-caches" -type f | wc -l
