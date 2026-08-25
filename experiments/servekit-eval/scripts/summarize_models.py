"""One table across models: loaders as rows, models as columns.

summarize_sweep.py answers "which loader wins for this model". This answers
"does the win hold as the model grows", which is the whole reason the sweep runs
three of them.

    summarize_models.py results/bristen-apertus8b results/bristen-llama70b ...

Each directory is one column, labelled with the model's full checkpoint name via
MODEL_NAMES; model_of() gets the slug (the part of the directory name after the
cluster, also the job name prefix) that collect() needs to find its runs.
"""
import sys
from pathlib import Path

from summarize_sweep import ARMS, LABELS, collect, cost_of

MODEL_NAMES = {
    "apertus8b": "Apertus-8B-Instruct-2509",
    "llama70b": "Llama-3.1-70B-Instruct",
    "glm4.7": "GLM-4.7",
}


def model_of(path):
    # results/bristen-apertus8b -> apertus8b, which is also the job name prefix.
    return Path(path).name.split("-", 1)[1]


def cell(row, value):
    """`value(row)`, or "-".

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
    return "%s (%s)" % (v, note) if note else "%s" % v


def table(title, columns, value):
    print("### %s" % title)
    print()
    print("| loader | %s |" % " | ".join(m for m, _ in columns))
    print("|---%s|" % ("|---" * len(columns)))
    for arm in ARMS:
        cells = []
        for _, rows in columns:
            cells.append(cell(rows.get(arm), value))
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
        display = MODEL_NAMES.get(model, model)
        columns.append((display, dict((r["arm"], r) for r in collect(d, model))))

    table("Weight loading (s), stage + load", columns, weight_cost)
    table("Total cold start (s)", columns,
          lambda r: "%.1f" % r["total"] if r.get("total") else None)

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
