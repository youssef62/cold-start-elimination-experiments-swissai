# CPU sensitivity of the no-mmap weight loader

`--weight-loader-disable-mmap` makes SGLang read safetensors with explicit
`pread` calls from a thread pool instead of letting the kernel demand-page an
mmap. That moves the read work into the job's own CPUs, so unlike the default
mmap path its throughput is bounded by `--cpus-per-task`. Every number in
[`../lustre-loading-exp`](../lustre-loading-exp) was collected on a full
128-core node, which leaves open how much of that arm's 12x speedup survives on
a smaller allocation.

This sweeps the same arm — `Llama-3.1-70B-Instruct`, TP4, weights read straight
off capstor — at 16, 32, 64 and 128 CPUs.

## Results

See `results/bristen-<date>/results.md`.

## Method

One run per CPU count, **fresh node per run**: `nommap` leaves ~141 GB of the
checkpoint in page cache and SLURM hands an `--exclusive` node straight back to
the next job, so each run adds its node to a growing `--exclude` list.

`--exclusive` is still required (the run needs all 4 GPUs and no co-tenant on
the NIC), which means `sacct AllocCPUS` reports 128 for every run regardless of
`--cpus-per-task`. The cgroup affinity limit is instead confirmed by the
`cpus_per_task=N nproc=N` line each run logs from inside the container.

Caveat: n=1 per point. At 128 CPUs, `nommap` weight_loading measured 47.2s ±
2.0s across three dates
([`../lustre-loading-exp/results/phase_stats.md`](../lustre-loading-exp/results/phase_stats.md)),
so differences of a few seconds between adjacent points are noise; only large
gaps are meaningful.

## Reproducing

```
bash experiments/cpu-sweep/scripts/nommap_cpu_sweep.sh
bash experiments/cpu-sweep/scripts/summarize_cpu_sweep.sh
```

`nommap_cpu_sweep.sh` submits
[`../lustre-loading-exp/scripts/methods/nommap.sbatch`](../lustre-loading-exp/scripts/methods/nommap.sbatch)
once per CPU count with `RESULTS_DIR` pointed at `results/<date>/cpu<N>/`, waits
for each job, and excludes every node already used. Overridable:
`CPUS_LIST`, `EXCLUDE_NODES`, `PARTITION`, `ACCOUNT`, `RES`.

`summarize_cpu_sweep.sh` feeds the per-CPU profile JSONs to
[`../lustre-loading-exp/scripts/analysis/phase_stats.py`](../lustre-loading-exp/scripts/analysis/phase_stats.py)
`--compare` and writes `results.md`.
