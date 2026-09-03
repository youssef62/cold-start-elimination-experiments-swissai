| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader (mmap) | 737.02 | 1.00x | 885.37 |
| --weight-loader-disable-mmap | 46.03 | 16.01x | 196.38 |
| --load-format fastsafetensors | 61.33 | 12.02x | 210.53 |
| servekit (shm, no overlap) | 9.29 (stage) + 14.22 = 23.51 | 31.35x | 173.79 |
| servekit (shm, overlap) | 14.3 | 51.54x | 164.98 |

| config | run | cuda graph capture (s) | tok/s | completed | errors |
|---|---|---|---|---|---|
| default loader (mmap) | llama70b-mmap-80989 | 8.4 | 402.8 | 64 | 0 |
| --weight-loader-disable-mmap | llama70b-nommap-80993 | 8.89 | 402.1 | 64 | 0 |
| --load-format fastsafetensors | llama70b-fst-80992 | 8.56 | 402.8 | 64 | 0 |
| servekit (shm, no overlap) | llama70b-servekit-80991 | 8.43 | 403.1 | 64 | 0 |
| servekit (shm, overlap) | llama70b-servekit-overlap-80990 | 8.45 | 403.2 | 64 | 0 |

| config | verify vs gold | worst token delta |
|---|---|---|
| default loader (mmap) | recorded the gold | - |
| --weight-loader-disable-mmap | PASS | 0.0 |
| --load-format fastsafetensors | PASS | 0.0 |
| servekit (shm, no overlap) | PASS | 0.0 |
| servekit (shm, overlap) | PASS | 0.0 |
