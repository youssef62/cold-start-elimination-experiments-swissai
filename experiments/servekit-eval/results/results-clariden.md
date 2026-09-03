### Weight loading (s), stage + load

| loader | Apertus-8B-Instruct-2509 | Llama-3.1-70B-Instruct | GLM-4.7 |
|---|---|---|---|
| default loader (mmap) | 92.8 | 794.2 | 861.6 |
| --weight-loader-disable-mmap | 4.5 | 27.8 (num_threads=4) | 263.7 (num_threads=2) |
| --load-format fastsafetensors | 11.9 | 48.6 | 113.8 |
| servekit (shm, no overlap) | 1.1 + 0.9 | 5.1 + 6.0 | 40.8 + 6.5 |
| servekit (shm, overlap) | 0.9 | 6.0 | 6.7 |

### Total cold start (s)

| loader | Apertus-8B-Instruct-2509 | Llama-3.1-70B-Instruct | GLM-4.7 |
|---|---|---|---|
| default loader (mmap) | 186.1 | 909.6 | 1082.7 |
| --weight-loader-disable-mmap | 94.0 | 146.8 (num_threads=4) | 503.2 (num_threads=2) |
| --load-format fastsafetensors | 105.7 | 161.9 | 210.8 |
| servekit (shm, no overlap) | 89.9 | 138.7 | 205.0 |
| servekit (shm, overlap) | 95.0 | 116.8 | 105.2 |

### Correctness and throughput

| loader | Apertus-8B-Instruct-2509 | Llama-3.1-70B-Instruct | GLM-4.7 |
|---|---|---|---|
| default loader (mmap) | PASS, 3146.1 tok/s | recorded the gold, 822.6 tok/s | recorded the gold, 360.6 tok/s |
| --weight-loader-disable-mmap | PASS, 3153.3 tok/s | PASS, 827.9 tok/s | PASS, 269.2 tok/s |
| --load-format fastsafetensors | PASS, 3128.9 tok/s | PASS, 823.8 tok/s | PASS, 357.6 tok/s |
| servekit (shm, no overlap) | PASS, 2931.6 tok/s | PASS, 828.7 tok/s | PASS, 389.0 tok/s |
| servekit (shm, overlap) | PASS, 3147.7 tok/s | PASS, 831.6 tok/s | PASS, 365.9 tok/s |
