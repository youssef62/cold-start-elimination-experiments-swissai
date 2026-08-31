# Run-to-run statistics across 3 dates (2026-08-24, 2026-08-25, 2026-08-31)

Bristen, `--cpus-per-task=128`, Llama-3.1-70B-Instruct TP4. One run per method per date.

## Baseline (default loader) phase breakdown

| phase | mean_s | stddev_s | min_s | max_s | explanation |
|---|---|---|---|---|---|
| process_startup | 25.50 | 2.67 | 23.21 | 28.44 | Process launch, mostly Python `import`s |
| tp_worker_spawn | 16.16 | 0.46 | 15.68 | 16.59 | Spawning the tensor-parallel worker processes |
| torch_distributed_init | 3.05 | 0.35 | 2.70 | 3.39 | Initializing the NCCL / torch distributed process group |
| unknown | 1.95 | 0.11 | 1.83 | 2.03 |  |
| weight_loading | **571.13** | 249.01 | 402.51 | 857.13 | Reading the model weights from storage and copying them to GPU memory |
| cuda_graph_capture | 28.63 | 0.61 | 28.05 | 29.26 | Capturing Decode CUDA graphs. In practice, this is mostly JIT compilation happening during the graph capture's forward passes. |
| piecewise_cuda_graph_capture | 78.92 | 0.23 | 78.66 | 79.08 | Capturing piecewise CUDA graphs (cuda graphs for prefill) |
| http_bind | 2.01 | 0.64 | 1.63 | 2.75 | Binding the HTTP server socket |
| warmup_request(JIT) | 15.05 | 0.13 | 14.94 | 15.20 | Warmup request that triggers remaining JIT kernel compilation |
| kv_cache_alloc | 1.58 | 0.00 | 1.58 | 1.58 | Allocating the KV cache |
| **total** | **743.18** | 245.68 | 574.67 | 1025.08 | |

## Loader comparison

| config | weight_loading mean_s (stddev) | min-max | speedup | total mean_s (stddev) | min-max |
|---|---|---|---|---|---|
| default loader | 571.1 (249.0) | 402.5-857.1 | 1.0x | 743.2 (245.7) | 574.7-1025.1 |
| nommap | 47.2 (2.0) | 45.7-49.5 | 12.1x | 215.9 (1.2) | 214.7-217.1 |
| fastsafetensors | 67.0 (16.1) | 56.3-85.5 | 8.5x | 238.1 (15.8) | 227.9-256.3 |
| /dev/shm staging + mmap | 20.2 (0.3) | 20.1-20.6 | 28.2x | 189.6 (2.3) | 187.0-191.5 |
| /dev/shm + TP-presharded | 9.2 (0.3) | 8.8-9.4 | 62.2x | 173.9 (9.2) | 163.3-179.7 |
| /dev/shm staging + presharded + overlap | 9.5 (0.2) | 9.3-9.7 | 60.4x | 182.7 (4.4) | 178.9-187.4 |
