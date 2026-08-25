#!/bin/bash
# The one offline step: write the sharded_state artifact the two servekit arms
# load. --only-prepare stops after writing it, so no measured run pays for it.
#
# All three models already have an artifact under ARTIFACTS_ROOT, so this is only
# needed if one of them stops loading. The artifact is keyed by the parallel and
# quant config, so this job's tp/pp/ep/dtype must match the serving arms' exactly
# or they re-dump at load time and the timing is meaningless. Both read them from
# models.sh.
#
#   MODEL=llama70b ./submit.sh prepare
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

set -euo pipefail

SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
MODEL="${MODEL:?set MODEL, e.g. MODEL=llama70b}"
source "${SLURM_SUBMIT_DIR}/scripts/models.sh"

EDF="${SLURM_SUBMIT_DIR}/scripts/${EDF_NAME}"
RUNDIR="${SLURM_SUBMIT_DIR}/${RESULTS_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "${RUNDIR}" "${ARTIFACT_ROOT}"

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
DIST_FLAGS=""
if [ "${NNODES}" -gt 1 ]; then
  DIST_FLAGS="--nnodes ${NNODES} --node-rank \${SLURM_PROCID} --dist-init-addr ${HEAD_NODE}:${DIST_PORT} --dist-timeout 3600"
fi

echo "model=${MODEL} arm=prepare (sharded_state artifact)"
echo "tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE} nodes=${NNODES}"
echo "source: ${MODEL_PATH}"
echo "artifact: ${ARTIFACT_ROOT}"
echo "started=$(date --iso-8601=seconds)"

# No backticks anywhere below, not even in comments: they would be command
# substituted on the submit host while this heredoc is built.
INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
export PYTHONPATH="${SLURM_SUBMIT_DIR}/patches"

servekit launch --servekit-artifact-path "${ARTIFACT_ROOT}" --only-prepare \
  --timeout ${READY_TIMEOUT} \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    ${PARALLEL_FLAGS} ${DIST_FLAGS} \
    --host 0.0.0.0 \
    --port ${SERVE_PORT} \
    --trust-remote-code \
    --context-length ${MAX_MODEL_LEN} \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --max-running-requests ${MAX_RUNNING_REQUESTS} \
    --random-seed ${RANDOM_SEED} 2>&1 | tee "${RUNDIR}/node\${SLURM_PROCID}.log"
EOF
)

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo
echo "=== artifact ==="
du -sh "${ARTIFACT_ROOT}" 2>/dev/null || true
echo "finished=$(date --iso-8601=seconds)"
