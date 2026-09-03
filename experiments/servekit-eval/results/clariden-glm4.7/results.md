| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader (mmap) | 861.61 | 1.00x | 1082.71 |
| --weight-loader-disable-mmap (num_threads=2) | 263.69 | 3.27x | 503.2 |
| --load-format fastsafetensors | 113.81 | 7.57x | 210.84 |
| servekit (shm, no overlap) | 40.84 (stage) + 6.51 = 47.35 | 18.20x | 205.02 |
| servekit (shm, overlap) | 6.7 | 128.60x | 105.25 |

| config | run | cuda graph capture (s) | tok/s | completed | errors |
|---|---|---|---|---|---|
| default loader (mmap) | glm4.7-mmap-3185339 | 34.83 | 360.6 | 64 | 0 |
| --weight-loader-disable-mmap (num_threads=2) | glm4.7-nommap-3185659 | 35.33 | 269.2 | 64 | 0 |
| --load-format fastsafetensors | glm4.7-fst-3185566 | 32.43 | 357.6 | 64 | 0 |
| servekit (shm, no overlap) | glm4.7-servekit-3185495 | 34.21 | 389.0 | 64 | 0 |
| servekit (shm, overlap) | glm4.7-servekit-overlap-3185466 | 30.75 | 365.9 | 64 | 0 |

| config | verify vs gold | worst token delta |
|---|---|---|
| default loader (mmap) | recorded the gold | - |
| --weight-loader-disable-mmap (num_threads=2) | PASS | 0.0 |
| --load-format fastsafetensors | PASS | 0.0 |
| servekit (shm, no overlap) | PASS | 0.0 |
| servekit (shm, overlap) | PASS | 0.0 |
