| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| mmap — SGLang's default on Lustre | 857.1 | 1.0× | 1025.1 |
| nommap | 46.3 | 18.5× | 217.1 |
| fastsafetensors (upstream) | 85.5 | 10.0× | 256.3 |
| /dev/shm staging + mmap | 20.1 + 7.7 (stage) | 30.9× | 197.9 |
| /dev/shm + TP-presharded | 9.4 + 7.7 (stage) | 50.1× | 187.4 |
| /dev/shm + presharded + overlap | 9.4 | 91.1× | 181.9 |

| config | node | job | graph capture (s) | tok/s | completed | errors |
|---|---|---|---|---|---|---|
| mmap — SGLang's default on Lustre | nid002328 | 81060 | 107.2 | 403.1 | 64/64 | 0 |
| nommap | nid002320 | 81059 | 107.3 | 402.8 | 64/64 | 0 |
| fastsafetensors (upstream) | nid002317 | 81058 | 107.2 | 402.6 | 64/64 | 0 |
| /dev/shm staging + mmap | nid002293 | 81057 | 105.7 | 402.5 | 64/64 | 0 |
| /dev/shm + TP-presharded | nid002292 | 81056 | 105.8 | 402.7 | 64/64 | 0 |
| /dev/shm + presharded + overlap | nid002324 | 81055 | 106.4 | 402.6 | 64/64 | 0 |

| phase | time(s) | % |
|---|---|---|
| process_startup | 23.2 | 2% |
| tp_worker_spawn | 15.7 | 2% |
| torch_distributed_init | 3.0 | 0% |
| unknown | 1.8 | 0% |
| weight_loading | 857.1 | 84% |
| cuda_graph_capture | 28.6 | 3% |
| piecewise_cuda_graph_capture | 78.7 | 8% |
| http_bind | 1.6 | 0% |
| warmup_request(JIT) | 14.9 | 1% |
