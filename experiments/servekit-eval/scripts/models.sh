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

# NOMMAP_THREADS caps --weight-loader-disable-mmap's read pool. Peak host memory
# there is roughly 2 x shard x threads x 4 ranks per node, against 515 GB: each
# worker does safetensors.torch.load(f.read()), holding a whole shard as bytes
# and the tensors built from it. Only glm4.7's 7.2 GB shards need the cap.
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
    NOMMAP_THREADS="${NOMMAP_THREADS:-}"
    TIME_LIMIT=01:30:00 ;;
  # 92 layers split 23 per stage across PP=4; 160 routed experts split 40 per EP rank.
  glm4.7)
    MODEL_PATH="${MODELS_ROOT}/zai-org/GLM-4.7"
    SERVED_MODEL_NAME=zai-org/GLM-4.7
    ARTIFACT_ROOT="${ARTIFACTS_ROOT}/glm4.7-tp4pp4-sharded"
    NNODES=4 TP_SIZE=4 PP_SIZE=4 EP_SIZE=4
    NOMMAP_THREADS="${NOMMAP_THREADS:-4}"
    TIME_LIMIT=00:50:00 ;;
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
CLUSTER="${CLUSTER:-bristen}"
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
ATTN_BACKEND_FLAG=""
case "${CLUSTER}" in
  bristen)  ACCOUNT=a-infra02 PARTITION=normal ;;
  clariden) ACCOUNT=infra02   PARTITION=debug
            ATTN_BACKEND_FLAG="--attention-backend flashinfer" ;;
esac
