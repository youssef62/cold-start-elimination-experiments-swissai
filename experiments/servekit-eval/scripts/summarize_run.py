"""Print one run's cold start, phases, throughput and verify verdict."""
import json
import sys
from pathlib import Path


def main(rundir):
    rundir = Path(rundir)
    # Multi-node runs write one report per node, single-node runs write run.json.
    reports = sorted(rundir.glob("run.node*.json")) or sorted(rundir.glob("run.json"))
    if not reports:
        print("  no report -- the engine never reported ready")
        return

    print("  %-6s %8s %s" % ("node", "total_s", "phases"))
    for path in reports:
        d = json.loads(path.read_text())
        phases = " ".join("%s=%.1f" % (p["name"], p["duration_s"]) for p in d.get("phases") or [])
        rank = d.get("node_rank")
        print("  %-6s %8.1f %s" % ("0" if rank is None else rank,
                                   d.get("total_duration_s") or 0.0, phases))

    head = json.loads(reports[0].read_text())
    bench = head.get("benchmark") or {}
    t = bench.get("throughput") or bench
    if t:
        print("  throughput: %s tok/s (%s/%s ok, errors=%s)" % (
            t.get("output_tok_per_s"), t.get("completed"), t.get("requests"), t.get("errors")))

    v = rundir / "verify.json"
    if v.is_file():
        d = json.loads(v.read_text())
        print("  verify: %s  worst_token_delta=%s worst_nll_delta=%s" % (
            "PASS" if d.get("passed") else "FAIL",
            d.get("worst_token_delta"), d.get("worst_nll_delta")))
        for f in d.get("failures") or []:
            print("    %s" % f)
    else:
        print("  verify: no verify.json (recording arm, or the check never ran)")


if __name__ == "__main__":
    main(sys.argv[1])
