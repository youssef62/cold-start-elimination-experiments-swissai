# nommap CPU sensitivity — bristen-2026-08-31

Llama-3.1-70B-Instruct, TP4, `--weight-loader-disable-mmap`, weights read
straight off capstor. One run per CPU count, fresh node per run.

- cpu16: nid002296
- cpu32: nid002285
- cpu64: nid002293
- cpu128: nid002284

| config | weight_loading mean_s (stddev) | min-max | speedup | total mean_s (stddev) | min-max |
|---|---|---|---|---|---|
| cpu16 | 192.2 (0.0) | 192.2-192.2 | 1.0x | 373.1 (0.0) | 373.1-373.1 |
| cpu32 | 189.6 (0.0) | 189.6-189.6 | 1.0x | 376.9 (0.0) | 376.9-376.9 |
| cpu64 | 102.9 (0.0) | 102.9-102.9 | 1.9x | 292.9 (0.0) | 292.9-292.9 |
| cpu128 | 48.9 (0.0) | 48.9-48.9 | 3.9x | 216.6 (0.0) | 216.6-216.6 |

Speedup is relative to the first row (cpu16).
