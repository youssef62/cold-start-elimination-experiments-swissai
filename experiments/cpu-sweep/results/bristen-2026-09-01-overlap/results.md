# servekit overlap, CPU sensitivity — bristen-2026-09-01-overlap

Llama-3.1-70B-Instruct, TP4, TP-presharded checkpoint staged to /dev/shm
concurrently with SGLang startup. One run per CPU count, fresh node per run,
`SLICES` fixed at its default so CPU count is the only variable.

| CPUs | stage (s) | stage GB/s | weight_loading (s) | total cold start (s) | overlap gate | bench errors | node |
|---|---|---|---|---|---|---|---|
| 16 | 29.2 | 5.12 | 12.2 | 227.9 | VALID (+54.8s slack) | 0 | nid002280 |
| 32 | 34.1 | 4.24 | 11.2 | 222.2 | VALID (+49.2s slack) | — | nid002313 |
| 64 | 21.5 | 6.82 | 11.2 | 191.0 | VALID (+34.2s slack) | 0 | nid002292 |
| 128 | 15.2 | 9.79 | 9.3 | 184.2 | VALID (+35.4s slack) | — | nid002285 |

`total cold start` = `ready_at - stage_start`, so it includes the stage.
A run whose gate says INVALID read partially-staged weights and must be
discarded, not compared: check `bench errors` for corroboration.
