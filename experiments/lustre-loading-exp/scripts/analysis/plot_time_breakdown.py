#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.linewidth": 1.0,
})

phases = [
    ("process_startup", 28.44, "#2a78d6"),
    ("tp_worker_spawn", 16.59, "#eb6834"),
    ("torch_distributed_init", 3.39, "#1baf7a"),
    ("unknown", 2.03, "#b3b1a8"),
    ("weight_loading", 453.74, "#eda100"),
    ("cuda_graph_capture", 29.26, "#e87ba4"),
    ("piecewise_cuda_graph_capture", 79.08, "#008300"),
    ("http_bind", 1.66, "#4a3aa7"),
    ("warmup_request(JIT)", 15.20, "#e34948"),
]

total = sum(d for _, d, _ in phases)

fig, ax = plt.subplots(figsize=(9, 4.2))

# only inline-label the one segment wide enough to hold its name comfortably;
# everything else is carried by the legend below, to avoid any clipped text.
INLINE_LABEL_MIN_PCT = 30

left = 0.0
for name, dur, color in phases:
    ax.barh(0, dur, left=left, height=0.6, color=color,
            edgecolor="white", linewidth=1.2)
    pct = dur / total * 100
    if pct >= INLINE_LABEL_MIN_PCT:
        ax.text(left + dur / 2, 0, f"{name}  ({dur:g}s, {pct:.0f}%)", ha="center", va="center",
                 fontsize=10, color="#1a1a19", clip_on=False)
    left += dur

ax.set_xlim(0, total)
ax.set_ylim(-0.6, 0.6)
ax.set_yticks([])
ax.set_xlabel("time (s)", labelpad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_title(f"Baseline cold start time breakdown, total {total:.1f}s", pad=14)

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in phases]
legend_labels = [f"{n} ({d:g}s, {d/total*100:.1f}%)" for n, d, _ in phases]
ax.legend(legend_handles, legend_labels, loc="upper center",
          bbox_to_anchor=(0.5, -0.85), ncol=3, frameon=False, fontsize=8.5,
          columnspacing=1.2, handletextpad=0.6, labelspacing=1.1)

fig.subplots_adjust(bottom=0.55, top=0.82)
fig.savefig("time_breakdown.png", dpi=200)
print("wrote time_breakdown.png")
