#!/bin/bash
# Loader-comparison table across the CPU-count arms of one sweep dir.
# Usage: summarize_cpu_sweep.sh [results/bristen-<date>]   (default: newest)
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

RES="${1:-$(ls -d experiments/cpu-sweep/results/bristen-* | sort | tail -1)}"
PHASE_STATS=experiments/lustre-loading-exp/scripts/analysis/phase_stats.py

args=()
nodes=()
for d in $(ls -d "$RES"/cpu*/ | sed "s|.*cpu||; s|/$||" | sort -n); do
  jsons="$(ls "$RES/cpu$d"/*-profile.json 2>/dev/null | paste -sd, -)"
  [ -z "$jsons" ] && { echo "cpu$d: no profile JSON, skipping" >&2; continue; }
  args+=("cpu$d=$jsons")
  nodes+=("cpu$d: $(ls "$RES/cpu$d"/*-profile.json | sed 's/.*-\(nid[0-9]*\)-profile.json/\1/' | paste -sd' ' -)")
done

{
  echo "# nommap CPU sensitivity — $(basename "$RES")"
  echo
  echo "Llama-3.1-70B-Instruct, TP4, \`--weight-loader-disable-mmap\`, weights read"
  echo "straight off capstor. One run per CPU count, fresh node per run."
  echo
  printf '%s\n' "${nodes[@]/#/- }"
  echo
  "${PYTHON:-python3}" "$PHASE_STATS" --compare "${args[@]}"
  echo
  echo "Speedup is relative to the first row (cpu16)."
} | tee "$RES/results.md"
