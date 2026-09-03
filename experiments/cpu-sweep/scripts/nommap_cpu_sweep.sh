#!/bin/bash
# How much of the no-mmap loader's speed comes from having a whole node?
#
# `--weight-loader-disable-mmap` issues explicit preads from a thread pool
# instead of letting the kernel demand-page an mmap, so unlike the default
# loader its throughput is bounded by the CPUs the task may run on. Every
# number in lustre-loading-exp was taken at --cpus-per-task=128; this sweeps
# the same arm at 16/32/64/128 to show where that 12x actually comes from.
#
# Run from anywhere:
#   bash experiments/cpu-sweep/scripts/nommap_cpu_sweep.sh
#
# FRESH NODE PER RUN: nommap still leaves ~141 GB in page cache and SLURM hands
# --exclusive nodes straight back, so each run grows a --exclude list.
#
#   CPUS_LIST="128 64" EXCLUDE_NODES=nid002284 bash .../nommap_cpu_sweep.sh
set -euo pipefail

# Physical path, not the /users/<user>/scratch symlink: the container cannot
# write through it, and every path the sbatch derives comes from SLURM_SUBMIT_DIR.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

ARM=experiments/lustre-loading-exp/scripts/methods/nommap.sbatch
CPUS_LIST="${CPUS_LIST:-128 64 32 16}"
RES="${RES:-experiments/cpu-sweep/results/bristen-$(date +%F)}"
mkdir -p "$RES"
RES_ABS="$(cd "$RES" && pwd -P)"

MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="$(pwd -P)/edf/llama-3.1-70b-sglang.toml"

USED="${EXCLUDE_NODES:-}"

for n in $CPUS_LIST; do
  mkdir -p "$RES_ABS/cpu$n"
  exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")

  jid="$(sbatch --parsable "${exclude_arg[@]}" \
        --partition="${PARTITION:-normal}" \
        --account="${ACCOUNT:-infra02}" \
        --cpus-per-task="$n" \
        --job-name="cpu$n-nommap" \
        --output="$RES/cpu$n-nommap-%j.out" \
        --export="ALL,RESULTS_DIR=$RES_ABS/cpu$n,MODEL=$MODEL,SERVED_MODEL_NAME=$SERVED_MODEL_NAME,EDF=$EDF" \
        "$ARM")"
  printf '%-6s -> %-8s exclude=[%s]\n' "cpu$n" "$jid" "${USED:-none}"

  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 15
  done

  node="$(sacct -j "$jid" -X -n -o NodeList | head -1 | xargs)"
  state="$(sacct -j "$jid" -X -n -o State | head -1 | xargs)"
  echo "    -> cpu$n finished on $node ($state)"
  grep -h '^cpus_per_task=' "$RES/cpu$n-nommap-$jid.out" 2>/dev/null || true

  if [ -n "$node" ] && [ "$node" != "None" ]; then
    USED="${USED:+$USED,}$node"
  fi
done

echo
echo "nodes used (must all be distinct): $USED"
echo "summarize: bash experiments/cpu-sweep/scripts/summarize_cpu_sweep.sh $RES"
