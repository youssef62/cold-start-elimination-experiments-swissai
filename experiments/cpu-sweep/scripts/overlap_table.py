#!/usr/bin/env python3
"""Rows of the overlap CPU sweep: stage cost, loader cost, true cold start, verdict.

`total cold start` is ready_at - stage_start, not the servekit total: the stage
is part of the cold start even though it runs before launch_server is timed.
"""
import glob
import json
import os
import re
import sys


def load(cpudir):
    prof = glob.glob(os.path.join(cpudir, "*-profile.json"))
    timing = glob.glob(os.path.join(cpudir, "*-timing.txt"))
    if not prof or not timing:
        return None
    d = json.load(open(prof[0]))
    t = dict(re.findall(r"(\w+)=([\d.\-]+)", open(timing[0]).read()))
    phases = [(p["name"], p["duration_s"]) for p in d["phases"]]

    start = float(t["stage_start_epoch"])
    end = float(t["stage_end_epoch"]) if "stage_end_epoch" in t else None

    # deadline the stage had to beat: every phase ahead of weight_loading
    elapsed = 0.0
    for name, dur in phases:
        if name == "weight_loading":
            break
        elapsed += dur
    wl_start = d["started_at"] + elapsed

    stage_log = glob.glob(os.path.join(cpudir, "*-stage.txt"))
    gbps = "?"
    if stage_log:
        m = re.search(r"stage_GBps=([\d.]+)", open(stage_log[0]).read())
        if m:
            gbps = m.group(1)

    if end is None:
        verdict, slack, stage_wall = "INVALID (stage never finished)", None, None
    else:
        stage_wall = end - start
        slack = wl_start - end
        rc = t.get("stage_rc", "?")
        if rc != "0":
            verdict = "INVALID (stage rc=%s)" % rc
        elif slack > 0:
            verdict = "VALID (+%.1fs slack)" % slack
        else:
            verdict = "**INVALID** (loader beat stage by %.1fs)" % -slack

    b = (d.get("benchmark") or {}).get("throughput", {}) or {}
    return dict(
        stage=stage_wall,
        gbps=gbps,
        wl=dict(phases).get("weight_loading"),
        servekit_total=d["total_duration_s"],
        cold=d["ready_at"] - start,
        verdict=verdict,
        errors=b.get("errors"),
        tok=b.get("output_tok_per_s"),
        node=os.path.basename(prof[0]).split("-")[-2],
    )


def main(res):
    dirs = sorted(glob.glob(os.path.join(res, "cpu*", "")),
                  key=lambda p: int(re.search(r"cpu(\d+)", p).group(1)))
    print("| CPUs | stage (s) | stage GB/s | weight_loading (s) | total cold start (s) | overlap gate | bench errors | node |")
    print("|---|---|---|---|---|---|---|---|")
    for d in dirs:
        n = re.search(r"cpu(\d+)", d).group(1)
        r = load(d)
        if r is None:
            print("| %s | — | — | — | — | no profile/timing (run did not reach ready) | — | — |" % n)
            continue
        fmt = lambda v: "—" if v is None else "%.1f" % v
        print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            n, fmt(r["stage"]), r["gbps"], fmt(r["wl"]), fmt(r["cold"]),
            r["verdict"], "—" if r["errors"] is None else r["errors"], r["node"]))
    print()
    print("`total cold start` = `ready_at - stage_start`, so it includes the stage.")
    print("A run whose gate says INVALID read partially-staged weights and must be")
    print("discarded, not compared: check `bench errors` for corroboration.")


if __name__ == "__main__":
    main(sys.argv[1])
