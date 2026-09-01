#!/bin/bash
# Same question as nommap_cpu_sweep.sh, asked of the servekit overlap arm.
#
# The overlap arm hides a /dev/shm stage inside SGLang's startup window. Both
# halves are CPU-sensitive and they contend with each other: the sliced stage is
# CPU-bound, and it now shares the allocation with process_startup and
# tp_worker_spawn. Shrinking --cpus-per-task therefore stretches the stage while
# the window it has to fit inside stays roughly fixed, so below some CPU count
# the stage stops beating the loader and the run goes INVALID -- SGLang reads
# half-staged bytes. Finding that cliff is the point of this sweep.
#
# SLICES stays at its default (64) for every point, so CPU count is the only
# variable. Tuning slices per CPU count would measure a different thing.
#
#   bash experiments/cpu-sweep/scripts/overlap_cpu_sweep.sh
#   CPUS_LIST="128 64" EXCLUDE_NODES=nid002284 bash .../overlap_cpu_sweep.sh
set -euo pipefail

# Physical path, not the /users/<user>/scratch symlink: the container cannot
# write through it, and every path the sbatch derives comes from SLURM_SUBMIT_DIR.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

ARM=experiments/lustre-loading-exp/scripts/methods/overlap.sbatch
CPUS_LIST="${CPUS_LIST:-128 64 32 16}"
RES="${RES:-experiments/cpu-sweep/results/bristen-$(date +%F)-overlap}"
mkdir -p "$RES"
RES_ABS="$(cd "$RES" && pwd -P)"

SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="$(pwd -P)/examples/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"

USED="${EXCLUDE_NODES:-}"

for n in $CPUS_LIST; do
  mkdir -p "$RES_ABS/cpu$n"
  exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")

  jid="$(sbatch --parsable "${exclude_arg[@]}" \
        --partition="${PARTITION:-normal}" \
        --account="${ACCOUNT:-infra02}" \
        --cpus-per-task="$n" \
        --job-name="cpu$n-overlap" \
        --output="$RES/cpu$n-overlap-%j.out" \
        --export="ALL,RESULTS_DIR=$RES_ABS/cpu$n,SHARDED_SRC=$SHARDED_SRC,SERVED_MODEL_NAME=$SERVED_MODEL_NAME,EDF=$EDF" \
        "$ARM")"
  printf '%-6s -> %-8s exclude=[%s]\n' "cpu$n" "$jid" "${USED:-none}"

  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 15
  done

  node="$(sacct -j "$jid" -X -n -o NodeList | head -1 | xargs)"
  state="$(sacct -j "$jid" -X -n -o State | head -1 | xargs)"
  echo "    -> cpu$n finished on $node ($state)"
  grep -h '^cpus_per_task=' "$RES/cpu$n-overlap-$jid.out" 2>/dev/null || true
  grep -hE '^  (stage \(overlapped\)|weight_loading|TOTAL COLD START)|VALID|INVALID' \
    "$RES/cpu$n-overlap-$jid.out" 2>/dev/null || true

  if [ -n "$node" ] && [ "$node" != "None" ]; then
    USED="${USED:+$USED,}$node"
  fi
done

echo
echo "nodes used (must all be distinct): $USED"
echo "summarize: bash experiments/cpu-sweep/scripts/summarize_overlap_cpu_sweep.sh $RES"
