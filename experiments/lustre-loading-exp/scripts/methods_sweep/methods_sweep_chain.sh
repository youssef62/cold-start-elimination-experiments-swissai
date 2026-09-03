#!/bin/bash
# One uniform sweep of every loader arm: exactly one run each, back to back,
# fastest arm first. The per-phase numbers in SUMMARY.md were collected weeks
# apart with 1-5 runs per arm; capstor swings 2-6x over that span, so those rows
# are not strictly comparable. This produces a single internally-consistent set.
#
# Order is the reverse of the SUMMARY table, so the slow, high-variance mmap arm
# runs last and a partial sweep still yields the interesting rows.
#
# FRESH NODE PER RUN. `--exclusive` grants sole use of a node, not a DIFFERENT
# node -- SLURM hands the same one straight back to the next job, where 141 GB of
# page cache would quietly feed every subsequent arm. So: submit one job, wait,
# learn its node, add it to a growing --exclude list.
#
# Run from the repo root:
#   bash experiments/lustre-loading-exp/scripts/methods_sweep/methods_sweep_chain.sh
#
# Seed the exclude list with nodes already contaminated:
#   EXCLUDE_NODES=nid002281,nid002293 bash .../methods_sweep_chain.sh
#
# MODEL selects the checkpoint: llama70b (default) or apertus8b.
#   MODEL=apertus8b bash .../methods_sweep_chain.sh
set -euo pipefail

# Physical path, not the /users/<user>/scratch symlink: the container cannot
# write through it, and SLURM_SUBMIT_DIR (which every sbatch derives its EDF,
# servekit and results paths from) inherits whichever one we submit from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"

S="experiments/lustre-loading-exp/scripts"
MODEL="${MODEL:-llama70b}"
RES="${RES:-experiments/lustre-loading-exp/results/methods_sweep}"
[ "$MODEL" != llama70b ] && RES="$RES-$MODEL"
mkdir -p "$RES"
RES_ABS="$(cd "$RES" && pwd -P)"

case "$MODEL" in
  llama70b)
    NATIVE_SRC=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
    SHARDED_SRC=/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded
    SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
    EDF="$(pwd -P)/edf/llama-3.1-70b-sglang.toml"
    SHM_DEST=/dev/shm/llama70b
    ;;
  apertus8b)
    NATIVE_SRC=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509
    SHARDED_SRC=/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus8b-tp4-sharded
    SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
    EDF="$(pwd -P)/edf/apertus-8b-sglang.toml"
    SHM_DEST=/dev/shm/apertus8b
    ;;
  *) echo "unknown MODEL=$MODEL (want llama70b or apertus8b)" >&2; exit 1 ;;
esac
COMMON="SERVED_MODEL_NAME=$SERVED_MODEL_NAME,EDF=$EDF,SHM_DEST=$SHM_DEST"

USED="${EXCLUDE_NODES:-}"

run() {  # run <tag> <script> [extra exports, comma-separated]
  local tag="$1" script="$2" extra="${3:-}" jid node state
  local exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")
  local partition_arg=()
  [ -n "${PARTITION:-}" ] && partition_arg=(--partition="$PARTITION")
  local account_arg=()
  [ -n "${ACCOUNT:-}" ] && account_arg=(--account="$ACCOUNT")
  local cpus_arg=()
  [ -n "${CPUS_PER_TASK:-}" ] && cpus_arg=(--cpus-per-task="$CPUS_PER_TASK")

  jid="$(sbatch --parsable "${exclude_arg[@]}" "${partition_arg[@]}" "${account_arg[@]}" "${cpus_arg[@]}" \
        --job-name="ms-$tag" \
        --output="$RES/ms-$tag-%j.out" \
        --export="ALL,RESULTS_DIR=$RES_ABS${extra:+,$extra}" \
        "$S/$script")"
  printf '%-12s -> %-8s exclude=[%s]\n' "$tag" "$jid" "${USED:-none}"

  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 15
  done

  node="$(sacct -j "$jid" -X -n -o NodeList | head -1 | xargs)"
  state="$(sacct -j "$jid" -X -n -o State | head -1 | xargs)"
  echo "    -> $tag finished on $node ($state)"

  if [ -n "$node" ] && [ "$node" != "None" ]; then
    USED="${USED:+$USED,}$node"
  fi
}

run overlap    methods/overlap.sbatch      "$COMMON,SHARDED_SRC=$SHARDED_SRC"
run preshard   methods/preshard.sbatch        "$COMMON,SHARDED_SRC=$SHARDED_SRC"
run shm_mmap   methods/shm_mmap.sbatch               "$COMMON,SRC=$NATIVE_SRC,TAG=shm_mmap"
run fst        methods/fastsafetensors.sbatch "$COMMON,MODEL=$NATIVE_SRC"
run nommap     methods/nommap.sbatch      "$COMMON,MODEL=$NATIVE_SRC"
run mmap       methods/mmap.sbatch        "$COMMON,MODEL=$NATIVE_SRC"

echo
echo "nodes used (must all be distinct): $USED"
echo "summarize: python3 $S/methods_sweep/summarize_methods_sweep.py $RES"
