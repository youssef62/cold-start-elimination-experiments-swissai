#!/usr/bin/env python3
"""Horizontal breakdown for the /dev/shm staging + presharded + overlap arm.

The top bar is the server cold start (phases, from launch to ready). The bar
underneath is the /dev/shm staging, which runs concurrently and finishes while
the engine is still in process_startup, so it never shows up on the critical
path. Data: results/methods_sweep/bristen-2026-08-31-cpu128, job 81610.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.linewidth": 1.0,
})

# server phases (profile.json), in order
phases = [
    ("process_startup", 31.75, "#2a78d6"),
    ("tp_worker_spawn", 16.49, "#eb6834"),
    ("torch_distributed_init", 3.08, "#1baf7a"),
    ("unknown", 2.00, "#b3b1a8"),
    ("weight_loading", 9.33, "#eda100"),
    ("cuda_graph_capture", 28.05, "#e87ba4"),
    ("piecewise_cuda_graph_capture", 79.47, "#008300"),
    ("http_bind", 1.67, "#4a3aa7"),
    ("warmup_request(JIT)", 15.23, "#e34948"),
]

# staging: stage_start_epoch .. stage_end_epoch (timing.txt); the server's
# launch is ~0.13s after stage_start, so we treat both as starting at 0.
STAGE_DUR = 1788167642.325703786 - 1788167626.089764473  # 16.24s

total = sum(d for _, d, _ in phases)

# --- shared bar geometry (keep identical to plot_time_breakdown.py) ---
# scale = ROW_IN / LANE_DATA inches per data unit; bar thickness in inches is
# BAR_H * scale, so both figures render bars at the same thickness.
LANE_DATA = 0.9
BAR_H = 0.34
ROW_IN = 0.85
SCALE = ROW_IN / LANE_DATA

Y_MAIN, Y_STAGE = 0.30, -0.30
YLIM = 0.62

FIG_W = 9.0
M_TOP, M_LEFT, M_RIGHT, M_BOT = 0.5, 1.5, 0.15, 0.10
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

left = 0.0
for name, dur, color in phases:
    ax.barh(Y_MAIN, dur, left=left, height=BAR_H, color=color,
            edgecolor="white", linewidth=1.2)
    pct = dur / total * 100
    if pct >= 30:
        ax.text(left + dur / 2, Y_MAIN, f"{name}  ({dur:g}s, {pct:.0f}%)",
                ha="center", va="center", fontsize=10, color="#1a1a19", clip_on=False)
    left += dur

ax.barh(Y_STAGE, STAGE_DUR, height=BAR_H, color="#00a0b0",
        edgecolor="white", linewidth=1.2)
ax.text(STAGE_DUR + total * 0.012, Y_STAGE,
        f"staging to /dev/shm  ({STAGE_DUR:.1f}s, hidden under process_startup)",
        ha="left", va="center", fontsize=10, color="#1a1a19", clip_on=False)

ax.set_xlim(0, total)
ax.set_ylim(-YLIM, YLIM)
ax.set_yticks([Y_MAIN, Y_STAGE])
ax.set_yticklabels(["engine start", "staging"])
ax.set_xticks([])
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_title(f"/dev/shm staging + presharded + overlap, cold start {total:.1f}s", pad=12)

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in phases]
legend_labels = [f"{n} ({d:g}s, {d/total*100:.1f}%)" for n, d, _ in phases]
fig.legend(legend_handles, legend_labels, loc="lower center",
           bbox_to_anchor=(0.5, M_BOT / fig_h), ncol=3, frameon=False, fontsize=8.5,
           columnspacing=1.2, handletextpad=0.6, labelspacing=0.9)

fig.savefig("overlap_breakdown.png", dpi=200)
print("wrote overlap_breakdown.png")
