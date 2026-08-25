# Cold Start Elimination for SwissAI Model Launch

Access to LLMs is crucial for academic research, be it for AI research, model testing or data annotation. For this reason, the SwissAI initiative operates an LLM serving platform on top of CSCS's Alps Clusters using [OpenTela](https://about.yao.sh/posts/opentela-swissai/) to pool together serving instances of multiple users in a decentralized manner and [SML](https://github.com/swiss-ai/model-launch) to seamlessly spin up nodes on top of `slurm` or CSCS's FireCrest. However, the current model launch suffers from large cold start times, which can be a bottleneck for research and development. In fact, an SGlang or VLLM server must first go through many costly steps before it can serve requests, including loading the model weights to GPU memory, computing CUDA Graphs, compiling JIT kernels and initializing NCCL communication. This can take tens of minutes for large models. 

In this blog post, we will discuss the cold start elimination experiments conducted for the SwissAI model launch and the results obtained. Our findings culminate in a package for fast cold starts on HPC clusters: [servekit](https://github.com/eth-easl/servekit). 


## I. Time Breakdown

We first map the cold start steps to their wall-clock time to identify the bottlenecks. We do this by parsing the logs printed by the SGLang server during the cold start phase. We added the log parser to `servekit` as `servekit profile`. 

For this experiment, we use `Llama-3.1-70B-Instruct` served with SGLang v0.5.10 (image `lmsysorg/sglang:v0.5.10`) with tensor-parallel size 4 on a single Bristen cluster node, with weights loaded with the default sglang model loader. SML keeps models in `capstor/store`, which is a [Lustre](https://www.lustre.org/) file system.  

/users/yboughizane/scratch/simple-serving-stack/experiments/lustre-loading-exp/results/meeting-sweep/2026-08-21/phase1_3_e2e-mmap-80022-nid002293-profile.json

| phase | duration_s | explanation |
|---|---|---|
| process_startup | 24.59 | Process launch, mostly Python `import`s |
| tp_worker_spawn | 16.11 | Spawning the tensor-parallel worker processes |
| torch_distributed_init | 3.13 | Initializing the NCCL / torch distributed process group |
| unknown | 1.85 |  |
| weight_loading | **652.19** | Reading the model weights from storage and copying them to GPU memory |
| cuda_graph_capture | 31.96 | Capturing Decode CUDA graphs. In practice, this is mostly JIT compilation happening during the graph capture's forward passes.  |
| piecewise_cuda_graph_capture | 82.79 | Capturing piecewise CUDA graphs (cuda graphs for prefill) |
| http_bind | 1.89 | Binding the HTTP server socket |
| warmup_request(JIT) | 15.00 | Warmup request that triggers remaining JIT kernel compilation |
| **total** | **826.05** | |

## II. Weight Loading

As we can see, loading weights from persistent storage (`capstor/store`) is by far the most time-consuming step, with **78%** of the total cold start time. **652.19** seconds for a 70B (130GB) model is a lot, that is **0.2GiB/s**. Capstor's aggregate theoretical bandwidth (across all users and jobs) is a whopping **1.19 TB/s**, and we are connected to it with **4 HPE Cray Slingshot-11 NICs** with a combined bandwidth of **4x23.28 GiB/s**. So we should definitely do better than **0.2GiB/s**. Let's understand why this happens.

https://docs.cscs.ch/alps/storage/
https://docs.cscs.ch/alps/hardware/#alps-high-speed-network


### 1. The default weight loader and mmap

`mmap` is a system call that maps a virtual memory region to a file. That memory region will not be mapped to a physical memory region until it is accessed. When a `mmap`ed page is accessed for the first time, the kernel will realize that the virtual page does not have a corresponding physical page but is `mmap`ed to a file. So it will load the corresponding file page from disk to the page cache and then associate the virtual page with the page cache page. This is called a **major page fault**. On subsequent access, the virtual page is already mapped to a physical page in the page cache and no disk access is needed. This is called a **minor page fault**. [^1]


The default SGLang loader uses `mmap`! The `DefaultModelLoader` calls methods like `multi_thread_safetensors_weights_iterator`, which return an iterator over pairs (`tensor_name`, `tensor_weights`) where `tensor_weights` is a `mmap`ed tensor. This iterator is passed to `LlamaForCausalLM`, which passes each parameter (like `ColumnParallelLinear`) its tensor weights. The parameter will then get a view of its needed weights according to its rank (`tp_rank` in the case of `ColumnParallelLinear`) and will then initiate a host (CPU) to device (GPU) copy of the weights.

<p align="center">
  <img src="assets/weight-loading.png" alt="Weight loading: mmap to shard to GPU" width="50%">
</p>

This will trigger a **major page fault** for each tensor, which will be loaded from Lustre going through the network to the page cache and then copied to GPU. This is a very slow process, especially for large models with many tensors. We validate this by running the exact same experiment with sglang's `--weight-loader-disable-mmap`. We get **45.7s** for weight loading, which is **14x faster** than the default loader and corresponds to **2.8GiB/s**. 

> **Insight.** For weight loading using an HDD-backed Lustre file system, using `mmap` is a bad idea. The simple `--weight-loader-disable-mmap` flag is a huge improvement.


Let's also try another one-flag method that does not use `mmap`: `fastsafetensors` (`--load-format fastsafetensors`)
partitions files across TP ranks; each TP process reads a file with `pread` and then exchanges the weights with other TP ranks using NCCL communication. Once all weights are on each GPU, tensors are parsed one by one directly in GPU memory. This eliminates the need for small tensor copies from host to device. We get **59.1s** for weight loading, which is **11x faster** than the default loader but worse than `--weight-loader-disable-mmap`. This confirms again that the bottleneck was `mmap` and not the small tensor copies from host to device.


| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader | 453.7 | 1.0× | 629.8 |
| nommap | 45.7 | 9.9× | 214.7 |
| fastsafetensors  | 59.1 | 7.7× | 230.0 |


lustre-loading-exp/results/meeting-sweep/bristen-2026-08-24-cpu128

This is a huge improvement and shows that `mmap` is not suitable for weight loading on Lustre file systems. However, **2.8GiB/s** is still far from the theoretical maximum of **4x23.28 GiB/s**. Let's see if we can do better.

- The no-mmap techniques are very sensitive to the number of CPUs. 



### 2. Understanding the lustre data storage

We will now try to see how fast we can load files with Lustre. Lustre is a distributed file system that saves files across different *Object Storage Targets (OSTs)*. Each OST is a storage volume that can be accessed independently. To increase the read bandwidth, we need to distribute the model weights across multiple OSTs so we can benefit from parallelism across OSTs. In our case, we will have each `.safetensors` file in a different OST. For models like `LLama-3.1-70B-Instruct`, there are 30 `.safetensors` files. 


<p align="center">
  <img src="assets/lustre.png" alt="Lustre data storage" width="50%">
</p>

However, single OSTs also benefit from having many requests in flight. For that, we will experiment with `dd iflag=direct`. This command lets us read files from disk without going through the page cache. We can use it as follows 
`dd iflag=direct if=input.bin of=output.bin bs=16M`, here `bs=16M` is the block size, which is the amount of data read from disk in one request. In our experiment, we will use `bs=16M` and read to `/dev/null` to measure the read speed. We study the effect of the number of parallel `dd` processes on the read speed. 

```bash
per=$(( 64 / nprocess ))            
t0=$(date +%s.%N)
for ((i=0;i<nprocess;i++)); do
dd if="${SICK}" of=/dev/null bs=16M skip=$((i*per)) count=$per iflag=direct status=none &
done
```

We measure this on a single 5GB shard (bs=16M, O_DIRECT, `dev/null`), sweeping the number of parallel `dd` processes reading disjoint, contiguous byte ranges of the same file:

<p align="center">
  <img src="assets/ost_queue_depth.png" alt="OST read throughput scales with the number of parallel readers, then flattens near the NIC line rate" width="60%">
</p>

Throughput scales close to linearly with reader count up to 16, then flattens. With many processes, we are able to keep many RPCs in flight, improving the bandwidth. 

Equipped with this knowledge, we try the following:
* Load in parallel (60 processes per file, this is maybe too much) from Lustre to `/dev/shm` (RAM), and then use a normal `mmap`-based default loader from `/dev/shm` to GPU. The staging takes `7s`, which is `**> 18GiB/s**`, already much better than everything we have seen before. The weight loading takes `20s`, which is `> 6GiB/s`. This is a **16x speedup** over the default loader.

<p align="center">
  <img src="assets/parallel-reads-lustre.png" alt="Parallel processes read file chunks from different OSTs on Lustre into /dev/shm, which SGLang then reads from" width="70%">
</p>

* This idea is possible because each node in both our clusters (Bristen and Clariden) has more RAM than GPU RAM. This means a node's specific shard of weights can always be stored in RAM if we preshard the weights across nodes. This is what we do next. We use `--load-format sharded_state`, which lets us save our weights by their TP rank. One added benefit is that our weights are now contiguous for each rank, which speeds up our H2D reads (see below). 

* Staging to `/dev/shm` is better than warming up the page cache for models that don't fit in a single node. For these, to warm all the weights a rank needs, we would need to fill the page cache with weights that don't fit. 

* Staging to `/dev/shm` is different from using `--weight-loader-disable-mmap` in two ways. 
    1. With `--weight-loader-disable-mmap`, each rank still reads the complete model weights: although it discards most of it and keeps only its shard, it still reads all of it. 
    2. Implementation: the standard `--weight-loader-disable-mmap` has 8 threads by default, each running a `pread` on a file. If our files are big, this will overflow RAM. 


| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader | 453.7 | 1.0× | 629.8 |
| nommap | 45.7 | 9.9× | 214.7 |
| fastsafetensors  | 59.1 | 7.7× | 230.0 |
| /dev/shm staging + mmap | 20.1 + 7.7 (stage) | 16.3× | 194.7 |
| /dev/shm + TP-presharded | 8.8 + 7.4 (stage) | 28.0× | 170.8 |
| /dev/shm staging+ presharded + overlap | 9.7 | 47.0× | 179.0 |

* To avoid corrupting our results with any kind of caching, we run the methods in **reverse order** of expected speed and on different nodes, meaning `dev/shm + presharded + overlap` ran before the default loader experiment. 

* 


lustre-loading-exp/results/meeting-sweep/bristen-2026-08-24-cpu128

### 3. Fast weight loading with servekit

/users/yboughizane/scratch/simple-serving-stack/experiments/lustre-loading-exp/results/meeting-sweep/bristen-2026-08-24-cpu128/results.md

I packaged the above ideas into a package called `servekit` that can be used to launch SGLang servers with fast cold starts. The goal is for `servekit` to be a wrapper around SGLang that implements cold start elimination optimizations. Currently, `servekit` implements fast weight loading and JIT kernels caching (more on this later). 

[`servekit`](https://github.com/eth-easl/servekit) has a main command, `servekit launch`, which takes a normal sglang command as an argument and launches it with optimizations. 

```bash
servekit launch --servekit-artifact-path <dir> \
  -- python -m sglang.launch_server --model-path <model> --tensor-parallel-size 4 ...
```

`servekit` also offers several utilities, such as `servekit profile`, `servekit bench` and `servekit verify`, to profile a cold start, benchmark a running server (useful to check that throughput is not affected by the optimizations), and verify that the server produces the same numbers as a trusted reference (useful for correctness checks: correctness is not affected by the optimizations). You can find documentation for these commands in the [servekit README](https://github.com/eth-easl/servekit/blob/main/README.md).


experiments/servekit-eval/results/results.md

**Comprehensive results**



**Config**

| | Apertus-8B-Instruct-2509 | Llama-3.1-70B-Instruct | GLM-4.7 |
|---|---|---|---|
| size | 16 GB | 141 GB | 717 GB |
| parallelism | TP4 | TP4 | TP4-PP4 |

This sweep uses SGLang v0.5.16 (image `lmsysorg/sglang:v0.5.16`).

**Weight loading time (s) on Bristen**

| Loader | Apertus-8B-Instruct-2509 | Llama-3.1-70B-Instruct | GLM-4.7 |
|---|---|---|---|
| default loader (mmap) | 78.0 (1.0x) | 737.0 (1.0x) | 827.9 (1.0x) |
| -`-weight-loader-disable-mmap` | 9.5 (8.2x) | 46.0 (16.0x) | 294.9 (2.8x, num_threads=4) |
| `--load-format fastsafetensors` | 17.9 (4.4x) | 61.3 (12.0x) | 143.8 (5.8x) |
| servekit (shm, no overlap) | 3.3 + 2.0 (14.6x) | 9.3 + 14.2 (31.3x) | 10.6 + 16.4 (30.7x) |
| **servekit (shm, overlap)** | **2.1 (36.5x)** | **14.3 (51.5x)** | **16.2 (51.1x)** |

 
* `-weight-loader-disable-mmap` ooms on GLM-4.7, so we had to reduce the number of threads to 4.
* `fastsafetensors` does not work for multi-node currently, the reported result for GLM4.7 is a patched version. 


**Limitations** 

- Presharding the models implies a separate prepare step; `servekit` tries to simplify this by doing it automatically on the first run, so users don't need to worry about it. In fact, when running `servekit launch --servekit-artefact-path <path> python -m sglang.launch_server ...`, a presharded copy of the model is created in `<path>`. This causes a first run to be slower than the default loader. 
- `ShardedStateLoader`, the loader behind `--load-format sharded_state`, is not a completely mature path. I discovered multiple bugs in it that I raised to the SGLang team: 
    - Issue mxfp4 (for e.g gptoss-20b) : https://github.com/sgl-project/sglang/issues/34448
    - Issue with MLA : https://github.com/sgl-project/sglang/issues/35702

  We are also involved in fixes for these: 
    - https://github.com/sgl-project/sglang/pull/35715
- I made `--overlap` as an opt-in flag because for now, it is unsafe. We did not yet implement a barrier mechanism to ensure that the engine does not start before the staging is complete. This is a known issue and we are working on it.

**Tradeoffs**

| Method | Pros ✅| Cons ❌|
|---|---|---|
| default loader (mmap) |  | Really slow on Lustre |
| `--weight-loader-disable-mmap` | One flag, 2.8x to 16x faster than default | - OOMs on large models (GLM-4.7), needed `num_threads=4` to fit<br>- Loads all weights per rank so scales badly with model size |
| `--load-format fastsafetensors` | - One flag, significant speedups | Doesn't work for multi-node yet; GLM-4.7 result needed a patched version<br> - Scales badly with node count due to costly NCCL though Slingshot |
| servekit | - Fastest across all models <br>- If model size scales linearly with node count, weight size loaded by node is constant and so is time (see LLama vs GLM4.7, 14s vs 16s)  | - Requires a costly prepare step that happens on first engine start<br>- Relies on `ShardedStateLoader`, which is not a mature path yet; a check with `servekit verify` is recommended to ensure correctness|
- mention the difference i am seing between clariden and bristen. 
- mention ShardedStateLoader limitations. 


[^1]: A threadpool of size 8 is used to do mmap in parallel. 

