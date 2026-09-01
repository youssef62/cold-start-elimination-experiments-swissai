#!/usr/bin/env python3
"""Plot readers vs MB/s from a parallel_read_sweep.sbatch .out file.

Usage: plot.py <sweep.out> [out_png]
"""
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.linewidth": 1.0,
})

out_path = sys.argv[1]
png_path = sys.argv[2] if len(sys.argv) > 2 else "sweep.png"

runs = {}
with open(out_path) as f:
    for line in f:
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s*$", line)
        if m:
            runs.setdefault(int(m.group(1)), []).append(float(m.group(2)))

readers = sorted(runs)
mbps = [statistics.median(runs[r]) for r in readers]
lo = [mbps[i] - min(runs[r]) for i, r in enumerate(readers)]
hi = [max(runs[r]) - mbps[i] for i, r in enumerate(readers)]

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.errorbar(readers, mbps, yerr=[lo, hi], marker="o", linewidth=2, markersize=7,
            color="#1f4e8c", capsize=3, elinewidth=1.2)
ax.set_xscale("log", base=2)
ax.set_xticks(readers)
ax.set_xticklabels(readers)
ax.set_xlim(readers[0] / 1.15, readers[-1] * 1.15)
ax.margins(y=0.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("concurrent readers")
ax.set_ylabel("MB/s")
ax.set_title("Read throughput vs. concurrent readers (single OST)")
fig.tight_layout(pad=0.6)
fig.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"wrote {png_path}")
