"""results.md for the sweep: one row per arm, newest run of each.

Compares head to head across arms -- a worker stops its clock at the dummy
health check, the head at the real ready line, so the two are not the same
measurement. An arm whose engine never reported ready has no timing at all; it
is listed with why rather than dropped, because "does not run" is a result.
"""
import json
import re
import sys
from pathlib import Path

# mmap2 is the determinism gate, not a loader: it reruns the baseline to show the
# verify tolerance measures the loader rather than engine noise. Its row would be
# a second copy of the baseline here, so the table leaves it out; read its verdict
# with summarize_run.py on its own run directory.
ARMS = ["mmap", "nommap", "fst", "servekit", "servekit-overlap"]
LABELS = {
    "mmap": "default loader (mmap)",
    "nommap": "--weight-loader-disable-mmap",
    "fst": "--load-format fastsafetensors",
    "servekit": "servekit (shm, no overlap)",
    "servekit-overlap": "servekit (shm, overlap)",
}


def newest(results, prefix, arm):
    # "<prefix>-servekit-*" also matches <prefix>-servekit-overlap-<id>, so the
    # suffix has to be exactly a job id rather than merely start with one.
    runs = []
    for d in results.glob("%s-%s-*" % (prefix, arm)):
        head, _, tail = d.name.rpartition("-")
        if d.is_dir() and head == "%s-%s" % (prefix, arm) and tail.isdigit():
            runs.append(d)
    # The job id is the tie-break, not mtime: a rerun of an older arm still wins.
    return max(runs, key=lambda d: int(d.name.rsplit("-", 1)[1]), default=None)


def phase(report, name):
    for p in report.get("phases") or []:
        if p["name"] == name:
            return p["duration_s"]
    return None


def label(row):
    return LABELS[row["arm"]] + row.get("label_suffix", "")


def fmt(v, suffix=""):
    return "-" if v is None else "%s%s" % (v, suffix)


def cost_of(row):
    """What the arm spent getting weights onto the GPUs: stage + load.

    The overlap arm reports no stage phase because it hides staging behind
    interpreter start, so there its cost is the load alone -- which is the point
    of the arm, not an omission.
    """
    weights = row.get("weights")
    if weights is None:
        return None
    return weights + (row.get("stage") or 0.0)


def collect(results, prefix):
    """One row per arm that has a run, newest run of each.

    Shared with summarize_models.py, which pivots the same rows into a
    models-as-columns table -- two readings of one parse, not two parsers.
    """
    results = Path(results)
    rows = []
    for arm in ARMS:
        run = newest(results, prefix, arm)
        if run is None:
            continue
        # Multi-node runs write run.node0.json for the head, single-node run.json.
        head_path = run / "run.node0.json"
        if not head_path.is_file():
            head_path = run / "run.json"
        head = json.loads(head_path.read_text()) if head_path.is_file() else None
        # A report exists even for a run that died mid-load; ready_at is what says
        # the server actually came up, and total_duration_s is 0.0 without it.
        row = {"arm": arm, "run": run.name, "ready": bool(head and head.get("ready_at"))}
        if row["ready"]:
            bench = head.get("benchmark") or {}
            t = bench.get("throughput") or bench
            # The nommap arm carries its thread cap in the command; it changes what
            # the row means, so it belongs in the label rather than a footnote.
            threads = re.search(r'"num_threads":\s*(\d+)', head.get("command") or "")
            if threads:
                row["label_suffix"] = " (num_threads=%s)" % threads.group(1)
            row.update({
                "total": head.get("total_duration_s"),
                "weights": phase(head, "weight_loading"),
                "stage": phase(head, "stage"),
                "graph": phase(head, "cuda_graph_capture"),
                "tps": t.get("output_tok_per_s"),
                "completed": t.get("completed"),
                "errors": t.get("errors"),
            })
        verify = run / "verify.json"
        if verify.is_file():
            v = json.loads(verify.read_text())
            row["verify"] = "PASS" if v["passed"] else "FAIL"
            row["worst"] = v["worst_token_delta"]
        elif arm == "mmap":
            row["verify"] = "recorded the gold"
        rows.append(row)
    return rows


def main(results, prefix):
    rows = collect(results, prefix)
    base = next((r for r in rows if r["arm"] == "mmap"), None)

    print("| config | weight_loading (s) | speedup | total cold start (s) |")
    print("|---|---|---|---|")
    for r in rows:
        if not r["ready"]:
            print("| %s | did not reach ready | - | - |" % label(r))
            continue
        # Staging is what the arm pays to make the load fast, so it belongs in the
        # loader's cost and in the ratio. Without it a staged arm is compared on
        # only the half of its work that got faster.
        cost = cost_of(r)
        shown = fmt(r.get("weights"))
        if r.get("stage"):
            shown = "%s (stage) + %s = %.2f" % (r["stage"], r["weights"], cost)
        speedup = "-"
        if base and cost_of(base) and cost:
            speedup = "%.2fx" % (cost_of(base) / cost)
        print("| %s | %s | %s | %s |" % (label(r), shown, speedup, fmt(r.get("total"))))

    print()
    print("| config | run | cuda graph capture (s) | tok/s | completed | errors |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s | %s | %s | %s |" % (
            label(r), r["run"], fmt(r.get("graph")), fmt(r.get("tps")),
            fmt(r.get("completed")), fmt(r.get("errors"))))

    print()
    print("| config | verify vs gold | worst token delta |")
    print("|---|---|---|")
    for r in rows:
        print("| %s | %s | %s |" % (
            label(r), r.get("verify", "-"), fmt(r.get("worst"))))


if __name__ == "__main__":
    # usage: summarize_sweep.py results/bristen-<model> <model>
    main(sys.argv[1], sys.argv[2])
