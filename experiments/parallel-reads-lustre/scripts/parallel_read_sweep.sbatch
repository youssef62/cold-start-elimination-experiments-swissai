#!/bin/bash
# Does reading a single file with more parallel readers increase throughput?
#
# Each Lustre OST serves one RPC at a time per in-flight slot, so a single
# `dd` only ever keeps one request outstanding. Splitting the same file into
# disjoint byte ranges and reading them concurrently keeps more requests in
# flight and should raise aggregate throughput, up to the OST/network limit.
#
#   sbatch experiments/parallel-reads-lustre/scripts/parallel_read_sweep.sbatch
#
#SBATCH --job-name=parallel-read-sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=experiments/parallel-reads-lustre/results/%x-%j.out

set -uo pipefail

FILE="${FILE:?set FILE to a path on Lustre, e.g. one .safetensors shard}"
TOTAL_MB=1024
BS=16M
TOTAL_BLOCKS=$(( TOTAL_MB / 16 ))   # blocks of size BS; must divide evenly by every reader count below

echo "job=${SLURM_JOB_ID} node=$(hostname) file=${FILE} started=$(date --iso-8601=seconds)"
echo

printf "%8s %12s\n" "readers" "MB/s"
for n in 1 2 4 8 16 32; do
  per=$(( TOTAL_BLOCKS / n ))
  t0=$(date +%s.%N)
  for ((i=0; i<n; i++)); do
    dd if="${FILE}" of=/dev/null bs="${BS}" skip=$((i*per)) count="${per}" iflag=direct status=none &
  done
  wait
  t1=$(date +%s.%N)
  awk -v s="$t0" -v e="$t1" -v n="$n" -v mb="$TOTAL_MB" \
    'BEGIN{printf "%8s %12.1f\n", n, mb/(e-s)}'
done

echo
echo "finished=$(date --iso-8601=seconds)"
