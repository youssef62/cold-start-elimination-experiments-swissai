# Cold Start Elimination for SwissAI Model Launch

Access to LLMs is crucial for academic research, be it for AI research, model testing or data annotation. For this reason, the SwissAI initiative operates an LLM serving platform on top of CSCS's Alps Clusters using [OpenTela](https://about.yao.sh/posts/opentela-swissai/) to pool together serving instances of multiple users in a decentralized manner and [SML](https://github.com/swiss-ai/model-launch) to seemlessly spin up nodes on top of `slurm` or CSCS's FireCrest. However, the current model launch suffers from large cold start times, which can be a bottleneck for research and development. In fact, an SGlang or VLLM server must first go through many costly steps before it can serve requests, including loading the model weights to GPU memory, computing CUDA Graphs, compiling JIT kernels and initializing NCCL communication. This can takes tens of minutes for large models. 

In this blog post, we will discuss the cold start elimination experiments conducted for the SwissAI model launch and the results obtained. Our findings culminate in a package for fast cold starts on HPC clusters : [servekit](https://github.com/eth-easl/servekit). 


## I. Time Breakdown

We first map the cold start steps to their wall-clock time to identify the bottlenecks. We do this by parsing the logs printed by the SGLang server during the cold start phase. We added the log parser to `servekit` as `servekit profile`. 

For this experiment, we use `Llama-3.1-70B-Instruct` served with SGLang v0.5.10 (image `lmsysorg/sglang:v0.5.10`) with tensor-parallel size 4 on a single Bristen cluster node, with weights loaded with the default sglang model loader. SML keeps models by in `capstor/store` which is a [Lustre](https://www.lustre.org/) file system.  

/users/yboughizane/scratch/simple-serving-stack/experiments/lustre-loading-exp/results/meeting-sweep/2026-08-21/phase1_3_e2e-mmap-80022-nid002293-profile.json

| phase | duration_s |
|---|---|
| process_startup | 24.59 |
| tp_worker_spawn | 16.11 |
| torch_distributed_init | 3.13 |
| unknown | 1.85 |
| weight_loading | **652.19** |
| cuda_graph_capture | 31.96 |
| piecewise_cuda_graph_capture | 82.79 |
| http_bind | 1.89 |
| warmup_request(JIT) | 15.00 |
| **total** | **826.05** |

- TODO I need to add a column to explain what each phase is doing.
- `cuda_graph_capture` is mostly JIT actually. 
- `process_startup` is mostly `import`. 

## II. Weight Loading

As we can see, loading weights from persistent storage (`capsto/store`) is by far the most time consuming step with **78%** of the total cold start time. **652.19** seconds for a 70B (130GB) model is a lot, that is **0.2GiB/s**. Capstor's aggregate theoretical bandwidth (accross all users and jobs) is a whopping **1.19 TB/s** and we are connected to it with **4 HPE Cray Slingshot-11 NICs** with a combined bandwith of **4x23.28 GiB/s**. So we should defintely do better than **0.2GiB/s**. Let's understand, why this happens.

https://docs.cscs.ch/alps/storage/
https://docs.cscs.ch/alps/hardware/#alps-high-speed-network


### 1. The default weight loader and mmap

`mmap` is a system call that maps a virtual memory region to a file. That memory region will not be mapped to a physical memory region until it is accessed. When a `mmap`ed page is accessed first, the kernel will realize that the virtual page does not have a corresponding physical page but it `mmaped` to a file. So it will load the corresponding file page from disk to page cache and then associate the virtual page region to the page cache page. This is called a **major page fault**. On subsequent access, the virtual page is already mapped to a physical page in the page cache and no disk access is needed. This is called a **minor page fault**. [^1]


The default SGLang loader uses `mmap`! The `DefaultModelLoader` will call methods like `multi_thread_safetensors_weights_iterator` which return an iterator over pairs (`tensor_name`, `tensor_weights`) where `tensor_weights` is a `mmap`ed tensor. This iterator is passed to `LlamaForCausalLM` which pass to each parameter (like `ColumnParallelLinear`) its tensor weights. The parameter will then get a view of its needed weights according to its rank (`tp_rank` in case of `ColumnParallelLinear`) and will then initiate a host (cpu) to device (gpu) copy of the weights.

<p align="center">
  <img src="assets/weight-loading.png" alt="Weight loading: mmap to shard to GPU" width="50%">
</p>

This will trigger a **major page fault** for each tensor, which will be loaded from Lustre going through the network to the page cache and then copied to GPU. This is a very slow process, especially for large models with many tensors. We validate this by running the exact same experiment with sglang's `--weight-loader-disable-mmap`. We get **45.7s** for weight loading, which is **14x faster** than the default loader and corresponds to **2.8GiB/s**. 

> **Insight.** For weight loading using a hdd backed Lustre file system, using `mmap` is a bad idea. The simple `--weight-loader-disable-mmap` flag is a huge improvement.


Let's also try another one flag method that does not use `mmap`: `fastsafetensors` (` --load-format fastsafetensors`)
partitions files accross TP ranks, each TP process reads a file with `pread` and then exchanges the weights with other TP ranks using NCCL communication. Once all weights are on each GPU, tensors are parsed one by one directly on GPU memory. This eliminates the need for small tensor copies from host to device. We get **59.1s** for weight loading, which is **11x faster** than the default loader but worse than `--weight-loader-disable-mmap`. This confirms again that bottleneck was `mmap` and not the small tensor copies from host to device.


| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader | 453.7 | 1.0× | 629.8 |
| nommap | 45.7 | 9.9× | 214.7 |
| fastsafetensors  | 59.1 | 7.7× | 230.0 |




This is a huge improvement and shows that `mmap` is not suitable for weight loading on Lustre file systems. However, **2.8GiB/s** is still far from the theoretical maximum of **4x23.28 GiB/s**. Let's see if we can do better.

- no mmap techniques are super sensitive to the number of CPUs. 



### 2. Understanding the lustre data storage

We will now try to see what's the fastest we can load files with Lustre. Lustre is a distributed file system that saves files accross different *Object Storage Targets (OSTs)*. Each OST is a storage volume that can be accessed independently. To increase the read bandwith, we need to distribute the model weights across multiple OSTs so we can benefit from accross OSTs parallelism. In our case, we will that have each `.safetensors` file in a different OST. For models like `LLama-3.1-70B-Instruct`, there's 30 `.safetensors` files. 


<p align="center">
  <img src="assets/lustre.png" alt="Lustre data storage" width="50%">
</p>

However, single OSTs also benefit from having many requests in flight. For that, we will experiment with `dd iflag=direct`. This command lets us read files files from disk without going through the page cache. We can use it as follows 
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

Throughput scales close to linearly with reader count up to 16, then flattens. With many process, we are able to keep many RPCs in flight, improving the bandwidth. 

Equipped with this knowledge we try the following:
* Load in parallel (60 process per file, this is maybe too much) from Lustre to `/dev/shm` (ram), and then a normal `mmap` based defautl loader from `/dev/shm` to GPU. The staging takes `7s`, this is `**> 18GiB/s**` which is already much better that all what we have seen before. The weight loading takes `20s` which is `> 6GiB/s`. This is a **16x speedup** over the default loader.

* This idea is possible because each node in both our clusters (bristen and clariden) have more RAM and GPU RAM. This means that a node's specific weight can always be storage to RAM if we preshard the weights accross nodes. This is what we do next. We use `--load-format sharded_state`, it enables use to save our weights by their TP rank. One added benefit is that now, our weights will be contigous for each rank, which speeds up our H2D reads (see below). 

* Staging to `/dev/shm` is better than warming up the page cache for models that don't fit in a single node. For these, to warm all the weights a rank needs, we need to fill the page cache with all the weights, which do not fit. 

* Staging to `/dev/shm` is different than using `--weight-loader-disable-mmap` in two ways. 
    1. With `--weight-loader-disable-mmap`, each rank still reads the complete model weights, although it discards most of it and keeps its shards, it still reads it. 
    2. Implementation: the standard `--weight-loader-disable-mmap` has by default 8 threads each running a `pread` on a file. If our files are big, this will overflow RAM. 


| config | weight_loading (s) | speedup | total cold start (s) |
|---|---|---|---|
| default loader | 453.7 | 1.0× | 629.8 |
| nommap | 45.7 | 9.9× | 214.7 |
| fastsafetensors  | 59.1 | 7.7× | 230.0 |
| /dev/shm staging + mmap | 20.1 + 7.7 (stage) | 16.3× | 194.7 |
| /dev/shm + TP-presharded | 8.8 + 7.4 (stage) | 28.0× | 170.8 |
| /dev/shm + presharded + overlap | 9.7 | 47.0× | 179.0 |


### 3. Fast weight loading with servekit

/users/yboughizane/scratch/simple-serving-stack/experiments/lustre-loading-exp/results/meeting-sweep/bristen-2026-08-24-cpu128/results.md

experiments/servekit-eval/results/results.md


| Loader | Apertus-8B-Instruct-2509 | Llama-3.1-70B-Instruct | GLM-4.7 |
|---|---|---|---|
| size | 16 GB | 141 GB | 717 GB |
| default loader (mmap) | 78.0 (1.0x) | 737.0 (1.0x) | 827.9 (1.0x) |
| --weight-loader-disable-mmap | 9.5 (8.2x) | 46.0 (16.0x) | 294.9 (2.8x, num_threads=4) |
| --load-format fastsafetensors | 17.9 (4.4x) | 61.3 (12.0x) | 143.8 (5.8x) |
| servekit (shm, no overlap) | 3.3 + 2.0 (14.6x) | 9.3 + 14.2 (31.3x) | 10.6 + 16.4 (30.7x) |
| **servekit (shm, overlap)** | **2.1 (36.5x)** | **14.3 (51.5x)** | **16.2 (51.1x)** |


**Limitations** 
- Presharding the models imply a seperate prepare step, `servekit` tries to simplify this by doing it automatically on the first run, so users don't need to worry about it. In fact, when running `servekit launch --servekit-artefact-path <path> python -m sglang.launch_server ...`, a presharded copy of the model is created in `<path>`. This causes a first run to be slower than the default loader. 
- We discovered `ShardedStateLoader`, the loader beind `--load-format sharded_state`, is not a completely mature path. We discovered multiple bugs with it that we raises to the SGLang team: 
    - Issue mxfp4 (for e.g gptoss-20b) : https://github.com/sgl-project/sglang/issues/34448
    - Issue with MLA : https://github.com/sgl-project/sglang/issues/35702

  We are involved in fixes for these: 
    - https://github.com/sgl-project/sglang/pull/35715
    - 

- mention the difference i am seing between clariden and bristen. 
- mention ShardedStateLoader limitations. 


[^1]: A threadpool of size 8 is used to do mmap in parallel. 

