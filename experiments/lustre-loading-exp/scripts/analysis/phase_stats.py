#!/usr/bin/env python3
"""Per-phase min/max/mean/std across repeated runs of the same arm.

Usage:
  phase_stats.py [--markdown] <profile.json> [<profile.json> ...]
  phase_stats.py --compare <label>=<json>,<json>,... [<label>=<json>,...]
    Loader-comparison table (weight_loading + total, mean/stddev/min/max,
    speedup vs the first label). One column pair per arm, one row per arm.
"""
import json
import statistics
import sys

# Kept in sync with the blog's phase table; phases not listed here print with
# an empty explanation instead of failing.
EXPLANATIONS = {
    "process_startup": "Process launch, mostly Python `import`s",
    "tp_worker_spawn": "Spawning the tensor-parallel worker processes",
    "torch_distributed_init": "Initializing the NCCL / torch distributed process group",
    "unknown": "",
    "weight_loading": "Reading the model weights from storage and copying them to GPU memory",
    "kv_cache_alloc": "Allocating the KV cache",
    "cuda_graph_capture": "Capturing Decode CUDA graphs. In practice, this is mostly JIT compilation happening during the graph capture's forward passes.",
    "piecewise_cuda_graph_capture": "Capturing piecewise CUDA graphs (cuda graphs for prefill)",
    "http_bind": "Binding the HTTP server socket",
    "warmup_request(JIT)": "Warmup request that triggers remaining JIT kernel compilation",
}


def stats_rows(runs):
    phase_names = list(dict.fromkeys(p["name"] for r in runs for p in r["phases"]))
    rows = []
    for name in phase_names:
        vals = [p["duration_s"] for r in runs for p in r["phases"] if p["name"] == name]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append((name, min(vals), max(vals), mean, std, len(vals)))
    totals = [r["total_duration_s"] for r in runs]
    mean = statistics.mean(totals)
    std = statistics.stdev(totals) if len(totals) > 1 else 0.0
    rows.append(("total", min(totals), max(totals), mean, std, len(totals)))
    return rows


def print_table(rows, n_runs):
    hdr = f"{'phase':<32}{'min':>9}{'max':>9}{'mean':>9}{'stddev':>9}  n"
    print(hdr)
    print("-" * len(hdr))
    for name, lo, hi, mean, std, n in rows[:-1]:
        print(f"{name:<32}{lo:>9.2f}{hi:>9.2f}{mean:>9.2f}{std:>9.2f}  {n}/{n_runs}")
    print("-" * len(hdr))
    name, lo, hi, mean, std, n = rows[-1]
    print(f"{name:<32}{lo:>9.2f}{hi:>9.2f}{mean:>9.2f}{std:>9.2f}  {n}/{n_runs}")


def print_markdown(rows, n_runs):
    print("| phase | mean_s | stddev_s | min_s | max_s | explanation |")
    print("|---|---|---|---|---|---|")
    for name, lo, hi, mean, std, n in rows[:-1]:
        bold = "**" if name == "weight_loading" else ""
        print(f"| {name} | {bold}{mean:.2f}{bold} | {std:.2f} | {lo:.2f} | {hi:.2f} | {EXPLANATIONS.get(name, '')} |")
    name, lo, hi, mean, std, n = rows[-1]
    print(f"| **{name}** | **{mean:.2f}** | {std:.2f} | {lo:.2f} | {hi:.2f} | |")


def _phase_stat(runs, name):
    vals = [p["duration_s"] for r in runs for p in r["phases"] if p["name"] == name]
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std, min(vals), max(vals)


def print_compare(groups):
    print("| config | weight_loading mean_s (stddev) | min-max | speedup | total mean_s (stddev) | min-max |")
    print("|---|---|---|---|---|---|")
    baseline_mean = None
    for label, paths in groups:
        runs = [json.load(open(p)) for p in paths]
        w_mean, w_std, w_min, w_max = _phase_stat(runs, "weight_loading")
        totals = [r["total_duration_s"] for r in runs]
        t_mean = statistics.mean(totals)
        t_std = statistics.stdev(totals) if len(totals) > 1 else 0.0
        if baseline_mean is None:
            baseline_mean = w_mean
        speedup = baseline_mean / w_mean
        print(f"| {label} | {w_mean:.1f} ({w_std:.1f}) | {w_min:.1f}-{w_max:.1f} | "
              f"{speedup:.1f}x | {t_mean:.1f} ({t_std:.1f}) | {min(totals):.1f}-{max(totals):.1f} |")


def main():
    args = sys.argv[1:]
    if "--compare" in args:
        groups = []
        for a in args:
            if a == "--compare":
                continue
            label, paths = a.split("=", 1)
            groups.append((label, paths.split(",")))
        print_compare(groups)
        return 0

    markdown = "--markdown" in args
    paths = [a for a in args if a != "--markdown"]
    if len(paths) < 2:
        print("need >=2 profile JSONs to compute statistics", file=sys.stderr)
        return 1

    runs = [json.load(open(p)) for p in paths]
    rows = stats_rows(runs)
    if markdown:
        print_markdown(rows, len(runs))
    else:
        print_table(rows, len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
