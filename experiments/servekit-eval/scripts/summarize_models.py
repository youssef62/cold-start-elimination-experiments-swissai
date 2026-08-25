"""One table across models: loaders as rows, models as columns.

summarize_sweep.py answers "which loader wins for this model". This answers
"does the win hold as the model grows", which is the whole reason the sweep runs
three of them. Speedups are always against that column's own mmap baseline, so a
column is self-contained and columns stay comparable to each other.

    summarize_models.py results/bristen-apertus8b results/bristen-llama70b ...

Each directory is one column, labelled with the model name its jobs are prefixed
with -- the part of the directory name after the cluster.
"""
import sys
from pathlib import Path

from summarize_sweep import ARMS, LABELS, collect, cost_of


def model_of(path):
    # results/bristen-apertus8b -> apertus8b, which is also the job name prefix.
    return Path(path).name.split("-", 1)[1]


def cell(row, base, value, ratio_of):
    """`value(row)` with its speedup against the column's baseline, or "-".

    Any qualifier the per-model summary would have put in the row label goes in
    the cell instead: here the label is shared across columns, and a cap that
    applies to one model only (glm4.7's nommap thread cap) has to travel with
    the number it changes or the row reads as one configuration.
    """
    if row is None or not row.get("ready"):
        return "-"
    v = value(row)
    if v is None:
        return "-"
    # " (num_threads=4)" -> "num_threads=4"
    note = row.get("label_suffix", "").strip().strip("()")
    b = ratio_of(base) if base is not None else None
    r = ratio_of(row)
    parts = []
    if b and r:
        parts.append("%.1fx" % (b / r))
    if note:
        parts.append(note)
    return "%s (%s)" % (v, ", ".join(parts)) if parts else "%s" % v


def table(title, columns, value, ratio_of):
    print("### %s" % title)
    print()
    print("| loader | %s |" % " | ".join(m for m, _ in columns))
    print("|---%s|" % ("|---" * len(columns)))
    for arm in ARMS:
        cells = []
        for _, rows in columns:
            row = rows.get(arm)
            base = rows.get("mmap")
            cells.append(cell(row, base, value, ratio_of))
        # An arm no model ran is noise, not a result.
        if all(c == "-" for c in cells):
            continue
        print("| %s | %s |" % (LABELS[arm], " | ".join(cells)))
    print()


def weight_cost(row):
    """Stage plus load: what the arm spent getting weights onto the GPUs."""
    c = cost_of(row)
    if c is None:
        return None
    if row.get("stage"):
        return "%.1f + %.1f" % (row["stage"], row["weights"])
    return "%.1f" % c


def main(dirs):
    columns = []
    for d in dirs:
        model = model_of(d)
        columns.append((model, dict((r["arm"], r) for r in collect(d, model))))

    table("Weight loading (s), stage + load", columns, weight_cost, cost_of)
    table("Total cold start (s)", columns,
          lambda r: "%.1f" % r["total"] if r.get("total") else None,
          lambda r: r.get("total"))

    print("### Correctness and throughput")
    print()
    print("| loader | %s |" % " | ".join(m for m, _ in columns))
    print("|---%s|" % ("|---" * len(columns)))
    for arm in ARMS:
        cells = []
        for _, rows in columns:
            row = rows.get(arm)
            if row is None:
                cells.append("-")
            else:
                v = row.get("verify", "-")
                tps = row.get("tps")
                cells.append("%s, %s tok/s" % (v, tps) if tps else v)
        if all(c == "-" for c in cells):
            continue
        print("| %s | %s |" % (LABELS[arm], " | ".join(cells)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: summarize_models.py <results/cluster-model> [...]")
    main(sys.argv[1:])
