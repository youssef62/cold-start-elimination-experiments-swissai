#!/bin/bash
# Does reading a single file with more parallel readers increase throughput?
#
# Each Lustre OST serves one RPC at a time per in-flight slot, so a single
# `dd` only ever keeps one request outstanding. Splitting the same file into
# disjoint byte ranges and reading them concurrently keeps more requests in
# flight and should raise aggregate throughput, up to the OST/network limit.
#
#   FILE=<path> sbatch experiments/parallel-reads-lustre/scripts/parallel_read_sweep.sbatch
#
#SBATCH --job-name=parallel-read-sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=experiments/parallel-reads-lustre/results/%x-%j.out

set -uo pipefail

FILE="${FILE:?set FILE to a path on Lustre, e.g. one .safetensors shard}"
TOTAL_MB="${TOTAL_MB:-4096}"
BS_MB=16
READERS="${READERS:-1 2 4 8 16 32 64}"
REPS="${REPS:-3}"
TOTAL_BLOCKS=$(( TOTAL_MB / BS_MB ))   # must divide evenly by every reader count

echo "job=${SLURM_JOB_ID} node=$(hostname) file=${FILE} total_mb=${TOTAL_MB} reps=${REPS} started=$(date --iso-8601=seconds)"
echo "stripe: $(lfs getstripe -c -i "${FILE}" | tr '\n' ' ')"
echo

printf "%8s %12s\n" "readers" "MB/s"
for n in ${READERS}; do
  per=$(( TOTAL_BLOCKS / n ))
  for ((rep=0; rep<REPS; rep++)); do
    t0=$(date +%s.%N)
    for ((i=0; i<n; i++)); do
      dd if="${FILE}" of=/dev/null bs="${BS_MB}M" skip=$((i*per)) count="${per}" iflag=direct status=none &
    done
    wait
    t1=$(date +%s.%N)
    awk -v s="$t0" -v e="$t1" -v n="$n" -v mb="$TOTAL_MB" \
      'BEGIN{printf "%8s %12.1f\n", n, mb/(e-s)}'
  done
done

echo
echo "finished=$(date --iso-8601=seconds)"
