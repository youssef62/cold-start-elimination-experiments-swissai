#!/bin/bash
# Per-CPU-count table for the overlap arm.
#
# phase_stats.py --compare is not enough here: its "total" is servekit's, which
# starts when launch_server starts and so EXCLUDES the stage. For overlap the
# honest cold start is ready_at - stage_start. This also reports the validity
# gate, since an overlap run that lost the race is not a slow result, it is a
# discarded one.
#
# Usage: summarize_overlap_cpu_sweep.sh [results/bristen-<date>-overlap]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

RES="${1:-$(ls -d experiments/cpu-sweep/results/bristen-*-overlap | sort | tail -1)}"

PY_BIN="$(command -v python3 || true)"
"${PY_BIN:-/usr/bin/python3}" -c '' 2>/dev/null || PY_BIN=/usr/bin/python3

{
  echo "# servekit overlap, CPU sensitivity — $(basename "$RES")"
  echo
  echo "Llama-3.1-70B-Instruct, TP4, TP-presharded checkpoint staged to /dev/shm"
  echo "concurrently with SGLang startup. One run per CPU count, fresh node per run,"
  echo "\`SLICES\` fixed at its default so CPU count is the only variable."
  echo
  "${PY_BIN}" "$(dirname "${BASH_SOURCE[0]}")/overlap_table.py" "$RES"
} | tee "$RES/results.md"
