clustering on date: 2026-08-24
| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| mmap — SGLang's default on Lustre | 453.7 | 1.0× | 629.8 |
| nommap | 45.7 | 9.9× | 214.7 |
| fastsafetensors (upstream) | 59.1 | 7.7× | 230.0 |
| /dev/shm staging + mmap | 20.1 + 7.7 (stage) | 16.3× | 194.7 |
| /dev/shm + TP-presharded | 8.8 + 7.4 (stage) | 28.0× | 170.8 |
| /dev/shm + presharded + overlap | 9.7 | 47.0× | 179.0 |

| config | node | job | graph capture (s) | tok/s | completed | errors |
|---|---|---|---|---|---|---|
| mmap — SGLang's default on Lustre | nid002805 | 80538 | 108.3 | 402.3 | 64/64 | 0 |
| nommap | nid002324 | 80511 | 106.3 | 403.2 | 64/64 | 0 |
| fastsafetensors (upstream) | nid002328 | 80483 | 108.5 | 401.9 | 64/64 | 0 |
| /dev/shm staging + mmap | nid002332 | 80450 | 106.1 | 402.4 | 64/64 | 0 |
| /dev/shm + TP-presharded | nid002325 | 80420 | 101.6 | 402.4 | 64/64 | 0 |
| /dev/shm + presharded + overlap | nid002321 | 80392 | 106.0 | 402.1 | 64/64 | 0 |

| phase | time(s) | % |
|---|---|---|
| process_startup | 28.4 | 5% |
| tp_worker_spawn | 16.6 | 3% |
| torch_distributed_init | 3.4 | 1% |
| unknown | 2.0 | 0% |
| weight_loading | 453.7 | 72% |
| cuda_graph_capture | 29.3 | 5% |
| piecewise_cuda_graph_capture | 79.1 | 13% |
| http_bind | 1.7 | 0% |
| warmup_request(JIT) | 15.2 | 2% |
