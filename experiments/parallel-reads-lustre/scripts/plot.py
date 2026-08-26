#!/usr/bin/env python3
"""Plot readers vs MB/s from a parallel_read_sweep.sbatch .out file.

Usage: plot.py <sweep.out> [out_png]
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_path = sys.argv[1]
png_path = sys.argv[2] if len(sys.argv) > 2 else "sweep.png"

readers, mbps = [], []
with open(out_path) as f:
    for line in f:
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s*$", line)
        if m:
            readers.append(int(m.group(1)))
            mbps.append(float(m.group(2)))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(readers, mbps, marker="o")
ax.set_xscale("log", base=2)
ax.set_xticks(readers)
ax.set_xticklabels(readers)
ax.set_xlabel("concurrent readers")
ax.set_ylabel("MB/s")
ax.set_title("Parallel reads on a single Lustre file")
fig.tight_layout()
fig.savefig(png_path, dpi=150)
print(f"wrote {png_path}")
