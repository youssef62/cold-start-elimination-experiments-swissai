### Weight loading (s), stage + load

| loader | Apertus-8B-Instruct-2509 | GLM-4.7 | Llama-3.1-70B-Instruct |
|---|---|---|---|
| default loader (mmap) | 78.0 | 827.9 | 737.0 |
| --weight-loader-disable-mmap | 9.5 | 294.9 (num_threads=4) | 46.0 |
| --load-format fastsafetensors | 17.9 | 143.8 | 61.3 |
| servekit (shm, no overlap) | 3.3 + 2.0 | 10.6 + 16.4 | 9.3 + 14.2 |
| servekit (shm, overlap) | 2.1 | 16.2 | 14.3 |

### Total cold start (s)

| loader | Apertus-8B-Instruct-2509 | GLM-4.7 | Llama-3.1-70B-Instruct |
|---|---|---|---|
| default loader (mmap) | 187.3 | 1024.7 | 885.4 |
| --weight-loader-disable-mmap | 120.2 | 433.9 (num_threads=4) | 196.4 |
| --load-format fastsafetensors | 130.3 | 265.2 | 210.5 |
| servekit (shm, no overlap) | 115.4 | 98.0 | 173.8 |
| servekit (shm, overlap) | 114.4 | 89.5 | 165.0 |

### Correctness and throughput

| loader | Apertus-8B-Instruct-2509 | GLM-4.7 | Llama-3.1-70B-Instruct |
|---|---|---|---|
| default loader (mmap) | recorded the gold, 1991.0 tok/s | recorded the gold, 215.8 tok/s | recorded the gold, 402.8 tok/s |
| --weight-loader-disable-mmap | PASS, 1975.0 tok/s | PASS, 223.4 tok/s | PASS, 402.1 tok/s |
| --load-format fastsafetensors | PASS, 1986.7 tok/s | PASS, 228.2 tok/s | PASS, 402.8 tok/s |
| servekit (shm, no overlap) | PASS, 1988.1 tok/s | PASS, 213.8 tok/s | PASS, 403.1 tok/s |
| servekit (shm, overlap) | PASS, 1984.4 tok/s | PASS, 223.7 tok/s | PASS, 403.2 tok/s |
