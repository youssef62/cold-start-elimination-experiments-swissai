#!/usr/bin/env python3
"""Emit the methods-sweep comparison table from one results dir.

summarize_e2e.py already covers phases, throughput and the cross-run
correctness check; it has no concept of staging time, which three of the six
arms need and which is where their cold start actually differs. This folds the
stage in and prints the presentation table.

Runs are clustered by the calendar date of each profile's started_at (not
the job ID, which was sorting stale runs ahead of fresh ones -- "77297" >
"3102776" lexicographically -- and not file mtime, which moving/copying a
result file can bump). Within a date, the arm's newest run wins, chosen by
started_at instead of filename so job-ID width no longer matters.

Usage: summarize_methods_sweep.py [results_dir] [date, e.g. 2026-08-17]
"""
import datetime
import glob
import json
import os
import re
import sys

# prefix of the profile filename -> (row label, how the stage is accounted)
ARMS = [
    ("phase1_3_e2e-mmap-",            "mmap — SGLang's default on Lustre", None),
    ("phase1_3_e2e-nommap-",          "nommap",                            None),
    ("phase1_3_e2e-fastsafetensors-", "fastsafetensors (upstream)",        None),
    ("phase4_sliced-",                "/dev/shm staging + mmap",           "serial"),
    ("phase6-preshard-shm-",          "/dev/shm + TP-presharded",          "serial"),
    ("phase7-overlap-",               "/dev/shm + presharded + overlap",   "overlapped"),
]

# column headings for the per-phase table, where the full labels do not fit
SHORT = {
    "mmap — SGLang's default on Lustre": "mmap",
    "fastsafetensors (upstream)": "fst",
    "/dev/shm staging + mmap": "shm+mmap",
    "/dev/shm + TP-presharded": "shm+preshard",
    "/dev/shm + presharded + overlap": "+overlap",
}


def stage_seconds(profile_path, mode):
    """-> (stage_s, cold_start_extra_s). Overlapped stages cost nothing extra."""
    base = profile_path[: -len("-profile.json")]
    if mode == "overlapped":
        t = dict(re.findall(r"(\w+)=([\d.\-]+)", open(base + "-timing.txt").read()))
        stage = float(t["stage_end_epoch"]) - float(t["stage_start_epoch"])
        return stage, None  # total is measured from stage start instead
    m = re.search(r"stage_wall_s=([\d.]+)", open(base + "-stage.txt").read())
    stage = float(m.group(1))
    return stage, stage


def run_date(profile_path):
    started = json.load(open(profile_path))["started_at"]
    return datetime.date.fromtimestamp(started), started


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    want_date = sys.argv[2] if len(sys.argv) > 2 else None

    all_hits = []
    for prefix, label, mode in ARMS:
        for path in glob.glob(os.path.join(d, prefix + "*-profile.json")):
            date, started = run_date(path)
            all_hits.append((prefix, label, mode, path, date, started))

    if want_date:
        target = datetime.date.fromisoformat(want_date)
    elif all_hits:
        target = max(date for *_, date, _ in all_hits)
    else:
        target = None
    if target:
        print(f"clustering on date: {target}", file=sys.stderr)

    rows = []
    for prefix, label, mode in ARMS:
        hits = sorted(
            (h for h in all_hits if h[0] == prefix and h[4] == target),
            key=lambda h: h[5],
        )
        if not hits:
            rows.append((label, None))
            continue
        if len(hits) > 1:
            print(f"warning: {len(hits)} runs for {label!r} on {target}, using the newest",
                  file=sys.stderr)
        path = hits[-1][3]
        p = json.load(open(path))
        ph = {x["name"]: x["duration_s"] for x in p["phases"]}
        thr = (p.get("benchmark") or {}).get("throughput") or {}
        stage = extra = None
        if mode:
            stage, extra = stage_seconds(path, mode)
        if mode == "overlapped":
            base = path[: -len("-profile.json")]
            t = dict(re.findall(r"(\w+)=([\d.\-]+)", open(base + "-timing.txt").read()))
            total = p["ready_at"] - float(t["stage_start_epoch"])
        else:
            total = p["total_duration_s"] + (extra or 0.0)
        rows.append((label, {
            "node": re.search(r"-(nid\d+)-profile\.json$", path).group(1),
            "job": re.search(r"-(\d+)-nid", path).group(1),
            "weight": ph.get("weight_loading", float("nan")),
            "stage": stage,
            "mode": mode,
            "total": total,
            "graph": ph.get("cuda_graph_capture", 0)
                     + ph.get("piecewise_cuda_graph_capture", 0),
            "tok_s": thr.get("output_tok_per_s"),
            "ok": f"{thr.get('completed')}/{thr.get('requests')}",
            "errors": thr.get("errors"),
            "phases": [(x["name"], x["duration_s"]) for x in p["phases"]],
            "servekit_total": p["total_duration_s"],
        }))

    # mmap is the baseline every speedup is quoted against.
    base = next((r["weight"] for l, r in rows if r and l.startswith("mmap")), None)

    print("| config | weight_loading (s) | speedup | total cold start (s) |")
    print("|---|---|---|---|")
    for label, r in rows:
        if r is None:
            print(f"| {label} | — | — | *not run* |")
            continue
        # An overlapped stage costs nothing on the critical path, so it is not
        # shown -- it is already absent from the total.
        w = f"{r['weight']:.1f}"
        # Overlapped staging costs nothing on the critical path, so it is
        # excluded from the speedup; serial staging is on the critical path
        # and must be counted alongside weight_loading.
        cost = r["weight"] + ((r["stage"] or 0.0) if r["mode"] != "overlapped" else 0.0)
        if r["stage"] is not None and r["mode"] != "overlapped":
            w += f" + {r['stage']:.1f} (stage)"
        speed = "—" if base is None else f"{base / cost:.1f}×"
        print(f"| {label} | {w} | {speed} | {r['total']:.1f} |")

    print()
    print("| config | node | job | graph capture (s) | tok/s | completed | errors |")
    print("|---|---|---|---|---|---|---|")
    for label, r in rows:
        if r is None:
            continue
        print(f"| {label} | {r['node']} | {r['job']} | {r['graph']:.1f} | "
              f"{r['tok_s']} | {r['ok']} | {r['errors']} |")

    # Per-phase breakdown of the default loader: where cold start actually goes
    # for a deployment that has changed nothing.
    mmap = next((r for l, r in rows if r and l.startswith("mmap")), None)
    if mmap:
        total = mmap["servekit_total"]
        print()
        print("| phase | time(s) | % |")
        print("|---|---|---|")
        for name, d in mmap["phases"]:
            print(f"| {name} | {d:.1f} | {100 * d / total:.0f}% |")

    nodes = [r["node"] for _, r in rows if r]
    if len(nodes) != len(set(nodes)):
        print("\n!! NODE REUSED across arms — page cache may have warmed a run",
              file=sys.stderr)
    bad = [l for l, r in rows if r and (r["errors"] or r["ok"].startswith("0"))]
    if bad:
        print(f"\n!! arms with bench errors, timings void: {bad}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
