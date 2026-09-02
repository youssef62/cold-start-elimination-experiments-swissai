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

# --- shared bar geometry (keep identical to plot_overlap_breakdown.py) ---
# scale = ROW_IN / LANE_DATA inches per data unit; bar thickness in inches is
# BAR_H * scale, so both figures render bars at the same thickness.
LANE_DATA = 0.9
BAR_H = 0.34
ROW_IN = 0.85
SCALE = ROW_IN / LANE_DATA

YLIM = 0.26

FIG_W = 9.0
M_TOP, M_LEFT, M_RIGHT, M_BOT = 0.5, 0.18, 0.15, 0.10
GAP, LEGEND_IN = 0.12, 0.78

axes_h = 2 * YLIM * SCALE
fig_h = M_TOP + axes_h + GAP + LEGEND_IN + M_BOT
fig = plt.figure(figsize=(FIG_W, fig_h))
ax = fig.add_axes([
    M_LEFT / FIG_W,
    (M_BOT + LEGEND_IN + GAP) / fig_h,
    (FIG_W - M_LEFT - M_RIGHT) / FIG_W,
    axes_h / fig_h,
])

# only inline-label the one segment wide enough to hold its name comfortably;
# everything else is carried by the legend below, to avoid any clipped text.
INLINE_LABEL_MIN_PCT = 30

left = 0.0
for name, dur, color in phases:
    ax.barh(0, dur, left=left, height=BAR_H, color=color,
            edgecolor="white", linewidth=1.2)
    pct = dur / total * 100
    if pct >= INLINE_LABEL_MIN_PCT:
        ax.text(left + dur / 2, 0, f"{name}  ({dur:g}s, {pct:.0f}%)", ha="center", va="center",
                 fontsize=10, color="#1a1a19", clip_on=False)
    left += dur

ax.set_xlim(0, total)
ax.set_ylim(-YLIM, YLIM)
ax.set_yticks([])
ax.set_xticks([])
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_title(f"Baseline cold start time breakdown, total {total:.1f}s", pad=12)

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in phases]
legend_labels = [f"{n} ({d:g}s, {d/total*100:.1f}%)" for n, d, _ in phases]
fig.legend(legend_handles, legend_labels, loc="lower center",
           bbox_to_anchor=(0.5, M_BOT / fig_h), ncol=3, frameon=False, fontsize=8.5,
           columnspacing=1.2, handletextpad=0.6, labelspacing=0.9)

fig.savefig("time_breakdown.png", dpi=200)
print("wrote time_breakdown.png")
