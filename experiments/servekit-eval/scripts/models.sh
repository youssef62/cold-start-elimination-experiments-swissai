# Every arm of every model sources this, so the only intended difference between
# two arms is the load flag, and between two models the block below. Anything
# that lives above the case cannot drift between arms OR between models.

SERVEKIT_DIR="${SERVEKIT_DIR:-/iopsstor/scratch/cscs/yboughizane/cold-start-elimination/servekit-main}"
EDF_NAME=sglang.toml

# One engine seed everywhere. servekit verify on main has no --seed, so this is
# what keeps sampling and MoE routing tie-breaks comparable across runs.
RANDOM_SEED=42

DIST_PORT=20000
SERVE_PORT=8080
# 32768 for all three, though Llama allows 131072 and Apertus 65536: the KV
# allocation is one of the phases we report, so it has to mean the same thing
# in every table.
MAX_MODEL_LEN=32768
MEM_FRACTION_STATIC=0.85
MAX_RUNNING_REQUESTS=256
READY_TIMEOUT=9000

MODELS_ROOT=/capstor/store/cscs/swissai/infra01/hf_models/models
ARTIFACTS_ROOT=/capstor/store/cscs/swissai/infra01/cold-start-experiments

# Read here (not just below with RESULTS_DIR) because llama70b's NOMMAP_THREADS
# default already needs it.
CLUSTER="${CLUSTER:-bristen}"

# NOMMAP_THREADS caps --weight-loader-disable-mmap's read pool (DEFAULT_NUM_THREADS
# is 8 in sglang's loader). Bristen runs fine uncapped. On clariden, llama70b's
# real weight_loader_disable_mmap default OOM-killed rank 1 mid weight-loading
# (v0.5.10 didn't have this problem there, so it looks like a v0.5.16 loader
# regression rather than a hardware ceiling); num_threads=1 avoided it but
# nearly 3x slower than num_threads=4, which also passed clean. glm4.7's 7.2 GB
# shards need the same cap for the analogous memory reason.
case "${MODEL:?set MODEL, e.g. MODEL=llama70b}" in
  apertus8b)
    MODEL_PATH="${MODELS_ROOT}/swiss-ai/Apertus-8B-Instruct-2509"
    SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
    ARTIFACT_ROOT="${ARTIFACTS_ROOT}/apertus8b-tp4-sharded"
    NNODES=1 TP_SIZE=4 PP_SIZE=1 EP_SIZE=1
    NOMMAP_THREADS="${NOMMAP_THREADS:-}"
    TIME_LIMIT=00:45:00 ;;
  llama70b)
    MODEL_PATH="${MODELS_ROOT}/meta-llama/Llama-3.1-70B-Instruct"
    SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
    ARTIFACT_ROOT="${ARTIFACTS_ROOT}/llama70b-tp4-sharded"
    NNODES=1 TP_SIZE=4 PP_SIZE=1 EP_SIZE=1
    [ "${CLUSTER}" = clariden ] && NOMMAP_THREADS="${NOMMAP_THREADS:-4}" || NOMMAP_THREADS="${NOMMAP_THREADS:-}"
    TIME_LIMIT=01:30:00 ;;
  # 92 layers split 23 per stage across PP=4; 160 routed experts split 40 per EP rank.
  glm4.7)
    MODEL_PATH="${MODELS_ROOT}/zai-org/GLM-4.7"
    SERVED_MODEL_NAME=zai-org/GLM-4.7
    ARTIFACT_ROOT="${ARTIFACTS_ROOT}/glm4.7-tp4pp4-sharded"
    NNODES=4 TP_SIZE=4 PP_SIZE=4 EP_SIZE=4
    # 4 (bristen's working value) still OOM-killed every non-head rank on
    # clariden; servekit-eval-glm4.7's own probing found num_threads=3 also
    # OOM's there and num_threads=2 is the highest that reaches ready.
    if [ "${CLUSTER}" = clariden ]; then NOMMAP_THREADS="${NOMMAP_THREADS:-2}"; else NOMMAP_THREADS="${NOMMAP_THREADS:-4}"; fi
    # clariden's debug-qos caps a job at 90 node-minutes total; 4 nodes leaves
    # 22 min each. Below what glm4.7's slowest bristen arms took (mmap ~20 min,
    # fst ~115 min) but debug's queue is otherwise much faster than normal's;
    # a slow arm timing out here just means resubmitting that one arm alone.
    if [ "${CLUSTER}" = clariden ]; then TIME_LIMIT=00:22:00; else TIME_LIMIT=00:50:00; fi ;;
  *) echo "unknown MODEL: ${MODEL} (apertus8b|llama70b|glm4.7)" >&2; return 1 2>/dev/null || exit 1 ;;
esac

GOLD="${VERIFY_GOLD:-${MODEL}-gold.json}"

# Dense single-node models run the plain TP command rather than carrying PP=1 and
# EP=1 flags the multi-node model needs. The dist flags are added by serve.sbatch,
# which is where SLURM_PROCID exists.
PARALLEL_FLAGS="--tensor-parallel-size ${TP_SIZE}"
if [ "${PP_SIZE}" -gt 1 ]; then
  PARALLEL_FLAGS="${PARALLEL_FLAGS} --pipeline-parallel-size ${PP_SIZE}"
fi
if [ "${EP_SIZE}" -gt 1 ]; then
  PARALLEL_FLAGS="${PARALLEL_FLAGS} --expert-parallel-size ${EP_SIZE}"
fi

# Results live under results/<cluster>-<model>: the same model on bristen and on
# clariden are different measurements, not two runs of one.
RESULTS_DIR="results/${CLUSTER}-${MODEL}"

# Slurm account names differ per cluster; the #SBATCH lines can't read this, so
# submit.sh passes it on the command line to override them. clariden's debug
# partition allows up to 1:30:00, which covers every arm's TIME_LIMIT above.
#
# clariden's driver/CUDA stack can't load the FA3 kernel sglang picks by
# default on Hopper (sgl_kernel raises ImportError: cannot import name
# 'flash_ops'), well after the weight-loading phase this sweep measures.
# flashinfer is the working fallback; pinned for every arm so a cluster's
# arms stay comparable to each other.
# MEM_MB=0 leaves --mem out of the sbatch call (bristen's default already
# grants a full exclusive node); clariden needs an explicit value under its
# cluster-wide MaxMemPerNode=850000, just below the node's actual 870000.
#
# clariden's debug-qos caps a job at MaxTRESMinsPerJob=node=90 (1.5 node-hours
# total); glm4.7's TIME_LIMIT above is already capped to fit that at 4 nodes.
ATTN_BACKEND_FLAG=""
MEM_MB=0
case "${CLUSTER}" in
  bristen)  ACCOUNT=a-infra02 PARTITION=normal ;;
  clariden) ACCOUNT=infra02   PARTITION=debug   MEM_MB=850000
            ATTN_BACKEND_FLAG="--attention-backend flashinfer" ;;
esac
