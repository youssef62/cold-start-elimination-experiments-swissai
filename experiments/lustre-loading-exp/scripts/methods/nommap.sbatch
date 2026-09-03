#!/bin/bash
# Phase 1.3 - end-to-end cold-start baseline for Llama-3.1-70B, NATIVE layout,
# WITHOUT mmap (--weight-loader-disable-mmap -> explicit pread instead of
# demand-paged mmap). Captures engine-reported `weight_loading` + total
# cold-start via servekit profile.
# Directly submittable:  sbatch experiments/lustre-loading-exp/scripts/methods/nommap.sbatch
# (mmap variant: mmap.sbatch)
#
# Do NOT run while a capstor-heavy job (dd sweep) is active - cross-node
# contention on the shared capstor OSTs skews weight_loading.
#
#SBATCH --job-name=e2e-baseline-nommap
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=experiments/lustre-loading-exp/results/methods/nommap/%x-%j.out

set -euo pipefail

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-meta-llama/Llama-3.1-70B-Instruct}"
EDF="${EDF:-${SLURM_SUBMIT_DIR}/edf/llama-3.1-70b-sglang.toml}"
SERVEKIT_DIR="${SERVEKIT_DIR:-/iopsstor/scratch/cscs/yboughizane/cold-start-elimination/servekit-main}"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
RESULTS_DIR="${RESULTS_DIR:-${SLURM_SUBMIT_DIR}/experiments/lustre-loading-exp/results/methods/nommap}"
PROFILE_OUT="${RESULTS_DIR}/phase1_3_e2e-nommap-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=nommap model=${MODEL}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
# the only proof the cgroup affinity limit took effect: under --exclusive sacct
# reports the whole node regardless of --cpus-per-task
echo "cpus_per_task=${SLURM_CPUS_PER_TASK} nproc=\$(nproc)"
servekit profile --out "${PROFILE_OUT}" \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --weight-loader-disable-mmap \
    --enable-metrics &
PROF=\$!

# servekit's bench moved out of profile into its own command: profile emits the
# report the moment ready is detected, bench merges into it while the server is
# still up.
servekit bench --url "http://127.0.0.1:${SGLANG_PORT}" --into "${PROFILE_OUT}" \
  --wait-ready 1800 --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true
kill \${PROF} 2>/dev/null || true
wait \${PROF} 2>/dev/null || true
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (nommap) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
