#!/bin/bash
# One arm of the loader sweep for one model: cold start measured end to end, then
# benched, then checked against that model's gold logprobs.
#
# Every arm of every model runs this same script. MODEL picks the checkpoint and
# its parallelism, ARM picks the loader, and nothing else differs -- copies of
# this heredoc are exactly the drift models.sh exists to prevent.
#
#   ARM=mmap            stock sglang loader off Lustre (the baseline)
#   ARM=nommap          --weight-loader-disable-mmap
#   ARM=fst             --load-format fastsafetensors
#   ARM=servekit        servekit launch: sharded_state artifact staged to /dev/shm
#   ARM=servekit-overlap  same, engine started alongside the stage
#
# VERIFY_MODE=record makes this run write the gold reference instead of checking
# against it; every other run checks. Run it once per model, on an mmap arm.
#
#   MODEL=llama70b ./submit.sh mmap
#
# --time and --nodes come from submit.sh: they are per-model and an #SBATCH line
# cannot read models.sh.
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

set -euo pipefail

# /users is a symlink to /iopsstor and is not mounted in the container, so the
# submit dir must be resolved before it crosses into one.
SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
MODEL="${MODEL:?set MODEL, e.g. MODEL=llama70b}"
source "${SLURM_SUBMIT_DIR}/scripts/models.sh"

ARM="${ARM:?set ARM, e.g. ARM=mmap}"
EDF="${SLURM_SUBMIT_DIR}/scripts/${EDF_NAME}"
RUNDIR="${SLURM_SUBMIT_DIR}/${RESULTS_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT="${RUNDIR}/run.json"
# servekit suffixes the report with .node<rank> only when the command carries
# --node-rank, which the single-node models do not, so the head's report is
# plain run.json there and bench --into would miss it.
if [ "${NNODES}" -gt 1 ]; then OUT_HEAD="${RUNDIR}/run.node0.json"; else OUT_HEAD="${OUT}"; fi
GOLD_PATH="${SLURM_SUBMIT_DIR}/${RESULTS_DIR}/${GOLD}"
mkdir -p "${RUNDIR}"

PIP_EXTRA=""
LOAD_FLAG=""
WRAPPER="servekit profile --out ${OUT} --timeout ${READY_TIMEOUT}"
case "${ARM}" in
  mmap)    ;;
  # Without mmap each loader thread does safetensors.torch.load(f.read()): a whole
  # shard as bytes plus the tensors built from it, so peak host memory is roughly
  # 2x shard x threads x 4 ranks per node against 515 GB. glm4.7's 7.2 GB shards
  # blow that at the engine default of 8 and the OS kills a rank at shard 1 of 93;
  # models.sh caps it there and leaves the smaller two at the default.
  nommap)  LOAD_FLAG="--weight-loader-disable-mmap"
           if [ -n "${NOMMAP_THREADS:-}" ]; then
             LOAD_FLAG="${LOAD_FLAG} --model-loader-extra-config '{\"num_threads\": ${NOMMAP_THREADS}}'"
           fi ;;
  # SERVEKIT_FST_DEVICE_FIX: see patches/sitecustomize.py -- sglang picks the
  # fastsafetensors device off the global rank, which is out of range on any
  # multi-node run.
  fst)     LOAD_FLAG="--load-format fastsafetensors"
           PIP_EXTRA="pip install fastsafetensors -q
export SERVEKIT_FST_DEVICE_FIX=1" ;;
  # --load-format sharded_state is redundant -- servekit appends it itself once
  # it picks the artifact -- but it has to be in OUR command: launch() decides
  # per-node sliced staging from wants_sharded_state(command), reading the
  # command as passed, before it appends anything. Without it every node of a
  # multi-node run stages the whole artifact instead of its own pipeline stage.
  servekit)
           LOAD_FLAG="--load-format sharded_state"
           WRAPPER="servekit launch --servekit-artifact-path ${ARTIFACT_ROOT} --out ${OUT} --timeout ${READY_TIMEOUT}" ;;
  servekit-overlap)
           LOAD_FLAG="--load-format sharded_state"
           WRAPPER="servekit launch --servekit-artifact-path ${ARTIFACT_ROOT} --out ${OUT} --timeout ${READY_TIMEOUT} --overlap" ;;
  *) echo "unknown ARM: ${ARM}" >&2; exit 1 ;;
esac

case "${ARM}" in
  servekit*)
    if [ -z "$(ls -A "${ARTIFACT_ROOT}" 2>/dev/null || true)" ]; then
      echo "ERROR: no artifact under ${ARTIFACT_ROOT}; run MODEL=${MODEL} ./submit.sh prepare first" >&2
      exit 1
    fi ;;
esac

VERIFY_MODE="${VERIFY_MODE:-check}"
if [ "${VERIFY_MODE}" = "record" ]; then
  VERIFY_ARGS="--record ${GOLD_PATH}"
else
  if [ ! -f "${GOLD_PATH}" ]; then
    echo "ERROR: no gold at ${GOLD_PATH}; run VERIFY_MODE=record MODEL=${MODEL} ./submit.sh mmap first" >&2
    exit 1
  fi
  VERIFY_ARGS="--reference ${GOLD_PATH} --out ${RUNDIR}/verify.json"
fi

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
DIST_FLAGS=""
if [ "${NNODES}" -gt 1 ]; then
  DIST_FLAGS="--nnodes ${NNODES} --node-rank \${SLURM_PROCID} --dist-init-addr ${HEAD_NODE}:${DIST_PORT} --dist-timeout 3600"
fi

echo "model=${MODEL} arm=${ARM}  verify=${VERIFY_MODE}"
echo "tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE} nodes=${NNODES} seed=${RANDOM_SEED}"
echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"
echo "model path: ${MODEL_PATH}"
echo "rundir: ${RUNDIR}"
echo "started=$(date --iso-8601=seconds)"

# No backticks anywhere below, not even in comments: they would be command
# substituted on the submit host while this heredoc is built.
INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
# The image sets BASH_ENV, which makes every one of the stager's subshells run
# nvidia-smi first: 0.41 GB/s instead of 16.89.
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
${PIP_EXTRA}
export PYTHONPATH="${SLURM_SUBMIT_DIR}/patches"
df -h /dev/shm

${WRAPPER} \
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
    --random-seed ${RANDOM_SEED} \
    --enable-metrics ${ATTN_BACKEND_FLAG} ${LOAD_FLAG} &
SRV=\$!

# Only the head binds a port, so only the head benches and verifies. Killing it
# drops the pipeline P2P connections and the workers follow.
if [ "\${SLURM_PROCID}" = "0" ]; then
  servekit bench --url "http://127.0.0.1:${SERVE_PORT}" --into "${OUT_HEAD}" \
    --wait-ready ${READY_TIMEOUT} \
    --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true
  servekit verify --url "http://127.0.0.1:${SERVE_PORT}" ${VERIFY_ARGS} \
    --wait-ready ${READY_TIMEOUT} --timeout 600 || true
  kill \${SRV} 2>/dev/null || true
fi
wait \${SRV} 2>/dev/null || true
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
  bash -c "${INNER_CMD}" || true

echo
echo "=== summary ==="
/usr/bin/python3 "${SLURM_SUBMIT_DIR}/scripts/summarize_run.py" "${RUNDIR}" || true
echo "finished=$(date --iso-8601=seconds)"
