| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader (mmap) | 827.89 | 1.00x | 1024.67 |
| --weight-loader-disable-mmap (num_threads=4) | 294.92 | 2.81x | 433.85 |
| --load-format fastsafetensors | 143.75 | 5.76x | 265.16 |
| servekit (shm, no overlap) | 10.55 (stage) + 16.39 = 26.94 | 30.73x | 98.02 |
| servekit (shm, overlap) | 16.19 | 51.14x | 89.54 |

| config | run | cuda graph capture (s) | tok/s | completed | errors |
|---|---|---|---|---|---|
| default loader (mmap) | glm4.7-mmap-80995 | 48.07 | 215.8 | 64 | 0 |
| --weight-loader-disable-mmap (num_threads=4) | glm4.7-nommap-81009 | 46.75 | 223.4 | 64 | 0 |
| --load-format fastsafetensors | glm4.7-fst-81001 | 47.39 | 228.2 | 64 | 0 |
| servekit (shm, no overlap) | glm4.7-servekit-80997 | 12.72 | 213.8 | 64 | 0 |
| servekit (shm, overlap) | glm4.7-servekit-overlap-80996 | 12.81 | 223.7 | 64 | 0 |

| config | verify vs gold | worst token delta |
|---|---|---|
| default loader (mmap) | recorded the gold | - |
| --weight-loader-disable-mmap (num_threads=4) | PASS | 0.0 |
| --load-format fastsafetensors | PASS | 0.0 |
| servekit (shm, no overlap) | PASS | 0.0 |
| servekit (shm, overlap) | PASS | 0.0 |
