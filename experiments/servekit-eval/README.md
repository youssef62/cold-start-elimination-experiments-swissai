# servekit loader sweep: Apertus-8B, Llama-3.1-70B, GLM-4.7

Does the cold-start result hold across model sizes, or is it a large-MoE story? Same six
loader arms on three checkpoints spanning 16 GB to 668 GB, dense and MoE, one node and
four. Every arm runs the same engine command from `scripts/models.sh`; the only intended
difference between two arms is how weights are loaded, and between two models the block
in the `case`.

| `MODEL` | checkpoint | size | parallelism | nodes |
|---|---|---|---|---|
| `apertus8b` | `swiss-ai/Apertus-8B-Instruct-2509` | 16 GB, 4 shards | TP4 | 1 |
| `llama70b` | `meta-llama/Llama-3.1-70B-Instruct` | 132 GB, 30 shards | TP4 | 1 |
| `glm4.7` | `zai-org/GLM-4.7` | 668 GB, 93 shards | TP4 PP4 EP4 | 4 |

Arms: `mmap` (stock loader, the baseline) · `mmap2` (same loader again, the determinism
gate) · `nommap` (`--weight-loader-disable-mmap`) · `fst` (`--load-format
fastsafetensors`) · `servekit` (sharded_state artifact staged to `/dev/shm`) ·
`servekit-overlap` (same, engine started alongside the stage).

## Running

```
./sweep.sh                                  # all three models, every arm, in order
MODELS="llama70b glm4.7" ./sweep.sh
MODEL=apertus8b ARMS="mmap servekit" ./sweep.sh
MODEL=llama70b ./submit.sh mmap             # one arm on its own
```

`sweep.sh` records each model's gold logprobs on its first `mmap` arm and adds every
job's nodes to a growing `--exclude`, because page cache survives between jobs: an arm
landing on a node an earlier arm already read the checkpoint on would measure a warm
read, not a cold start.

Then, per model:

```
/usr/bin/python3 scripts/summarize_sweep.py results/bristen-<model> <model> \
  > results/bristen-<model>/results.md
```

and once across all three -- loaders as rows, models as columns, each column's
speedups against its own mmap baseline:

```
/usr/bin/python3 scripts/summarize_models.py results/bristen-* > results/results.md
```

## Reading the output

Each job writes `results/bristen-<model>/<model>-<arm>-<jobid>/` with one
`run.node*.json` per node and a `verify.json`, and prints `summarize_run.py` at the end
of its `.out`, so a single `.out` says whether the arm reached ready, where its time
went, what it served and whether it served the right thing.

Two things gate the rest:

- **`mmap2` must PASS with worst token delta 0.0.** It is two cold starts of the *same*
  loader on different nodes. Without it a fast arm's PASS would only prove the tolerance
  is wider than engine noise.
- **Throughput must be flat across arms within a model.** These arms change how weights
  load, not what is served. Moving tok/s means an arm changed something it should not
  have.

## Notes

- Artifacts for the servekit arms are prepared once and reused
  (`cold-start-experiments/{apertus8b-tp4,llama70b-tp4,glm4.7-tp4pp4}-sharded`). They are
  keyed to tp/pp/dtype: `servekit` checks the manifest against the command and falls back
  to the engine loader if they disagree, so a mismatch shows up as a slow arm, not an
  error. `MODEL=<m> ./submit.sh prepare` rewrites one.
- `nommap` peak host memory is roughly `2 x shard x threads x 4 ranks` against 515 GB per
  node — each worker holds a whole shard as bytes *and* the tensors built from it. Only
  glm4.7's 7.2 GB shards need the cap (`NOMMAP_THREADS=4` in `models.sh`); the other two
  run at the engine default. `summarize_sweep.py` puts the thread count in the row label
  so a capped row is never silently compared against an uncapped one.
- `patches/sitecustomize.py` carries two upstream workarounds: sglang's hardcoded 480 s
  post-load barrier, and its fastsafetensors device index, which is picked off the
  *global* rank and so is out of range on any multi-node run (the `fst` arm sets
  `SERVEKIT_FST_DEVICE_FIX=1`; no other arm inherits it).

## Known gap

glm4.7 has no `mmap2` row. Bristen had 16 of its ~32 GPU nodes drained while the sweep
ran, and a second 4-node cold start of the baseline loader could not be scheduled; the
arm was dropped rather than run warm, since a warm rerun measures page cache instead of
the loader. So glm4.7's fast arms rest on a determinism gate established on apertus8b and
llama70b, both of which passed at exactly 0.0 rather than merely under tolerance. Running
`MODEL=glm4.7 ARMS=mmap2 ./sweep.sh` when the cluster recovers fills it in.
