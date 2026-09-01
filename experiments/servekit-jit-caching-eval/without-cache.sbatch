#!/bin/bash
# Other half of the JIT cache A/B (see with-cache.sbatch). Renames the shared
# artifact's servekit-caches/ dir away so servekit finds nothing to stage and
# the engine JITs from cold, then renames it back -- even on failure -- so
# other experiments sharing this artifact are unaffected.
#
# FlashInfer backend, matching with-cache.sbatch.
#
#   sbatch with-cache.sbatch
#   # note the node it landed on (squeue -j <id>), then:
#   sbatch --exclude=<that node> without-cache.sbatch
#
# Submit serially, not with --dependency: that hands back the same node.
#
#SBATCH --job-name=servekit-jit-no-cache
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
SERVEKIT_DIR="${SERVEKIT_DIR:-/iopsstor/scratch/cscs/yboughizane/cold-start-elimination/servekit-main}"
EDF="${SLURM_SUBMIT_DIR}/../servekit-eval/scripts/sglang.toml"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded-fa3}"
CACHE_DIR="${ARTIFACT_ROOT}/servekit-caches"
CACHE_DIR_AWAY="${ARTIFACT_ROOT}/servekit-caches.disabled"

MODEL_PATH=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
READY_TIMEOUT=1200
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${SLURM_JOB_ID}"
OUT="${RESULTS_DIR}/no-cache.json"
mkdir -p "${RESULTS_DIR}"

echo "job=${SLURM_JOB_ID} node=$(hostname) arm=no-cache"

mv "${CACHE_DIR}" "${CACHE_DIR_AWAY}"
# Restores the artifact's cache dir even on failure: scancel/timeout/errors
# all skip a plain final line, but Slurm sends SIGTERM before SIGKILL, giving
# this trap a chance to run either way.
trap 'mv "${CACHE_DIR_AWAY}" "${CACHE_DIR}" 2>/dev/null || true' EXIT

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" bash -c "
set -euo pipefail
unset BASH_ENV
pip install -e '${SERVEKIT_DIR}' --no-build-isolation --no-deps -q
servekit launch --servekit-artifact-path '${ARTIFACT_ROOT}' --out '${OUT}' --timeout ${READY_TIMEOUT} --overlap -- \
  python -m sglang.launch_server \
    --model-path '${MODEL_PATH}' \
    --served-model-name '${SERVED_MODEL_NAME}' \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port 8080 \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --random-seed 42 \
    --attention-backend flashinfer \
    --load-format sharded_state &
SRV=\$!
for i in \$(seq 1 ${READY_TIMEOUT}); do
  [ -f '${OUT}' ] && break
  sleep 1
done
kill \${SRV} 2>/dev/null || true
wait \${SRV} 2>/dev/null || true
"

echo
jq '{total_duration_s, phases}' "${OUT}"
