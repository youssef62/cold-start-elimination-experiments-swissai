# Parallel reads on a single Lustre file

Does splitting one file into disjoint byte ranges and reading them with more
concurrent processes increase throughput? Each range is read with
`dd iflag=direct` (bypasses the page cache) so every reader measures a genuine
disk read, not a page-cache hit.

```
FILE=/path/to/some/large/file sbatch scripts/parallel_read_sweep.sbatch
python scripts/plot.py results/parallel-read-sweep-<jobid>.out results/sweep.png
```

Sweeps reader counts `READERS` (default 1/2/4/8/16/32/64) over a fixed
`TOTAL_MB` window (default 4096) of `FILE`, `bs=16M`, `REPS` repetitions each,
and reports aggregate MB/s per reader count. `TOTAL_MB / 16` must divide
evenly by every reader count. Pick a single-striped `FILE` (check with
`lfs getstripe -c`) to measure one OST.
