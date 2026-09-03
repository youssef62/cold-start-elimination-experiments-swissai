| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader (mmap) | 78.04 | 1.00x | 187.33 |
| --weight-loader-disable-mmap | 9.52 | 8.20x | 120.18 |
| --load-format fastsafetensors | 17.93 | 4.35x | 130.34 |
| servekit (shm, no overlap) | 3.32 (stage) + 2.03 = 5.35 | 14.59x | 115.36 |
| servekit (shm, overlap) | 2.14 | 36.47x | 114.38 |

| config | run | cuda graph capture (s) | tok/s | completed | errors |
|---|---|---|---|---|---|
| default loader (mmap) | apertus8b-mmap-80983 | 4.83 | 1991.0 | 64 | 0 |
| --weight-loader-disable-mmap | apertus8b-nommap-80987 | 4.89 | 1975.0 | 64 | 0 |
| --load-format fastsafetensors | apertus8b-fst-80986 | 4.97 | 1986.7 | 64 | 0 |
| servekit (shm, no overlap) | apertus8b-servekit-80985 | 4.83 | 1988.1 | 64 | 0 |
| servekit (shm, overlap) | apertus8b-servekit-overlap-80984 | 4.9 | 1984.4 | 64 | 0 |

| config | verify vs gold | worst token delta |
|---|---|---|
| default loader (mmap) | recorded the gold | - |
| --weight-loader-disable-mmap | PASS | 0.0 |
| --load-format fastsafetensors | PASS | 0.0 |
| servekit (shm, no overlap) | PASS | 0.0 |
| servekit (shm, overlap) | PASS | 0.0 |
