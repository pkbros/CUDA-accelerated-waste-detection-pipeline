# Kernel notes — why each version is faster or slower

Written for the person who has to defend this repo in an interview. Each section states the
mechanism *before* the numbers, so the explanation survives a different GPU.

The measured values live in [`../results/summary.md`](../results/summary.md), generated from
`results/*.csv`. This file never quotes them from memory.

---

## 0. How every number in this repo was produced

Identical for all benchmarks, which is the point — otherwise the rows are not comparable.

1. **Warm up.** 5–20 untimed iterations. The first launches of a process pay for context
   setup, module loading and clock ramp; timing them measures the driver, not the kernel.
2. **One CUDA-event pair per iteration.** `cudaEventRecord` timestamps are written by the GPU,
   so they measure device time and ignore host scheduling jitter. A wall clock around an async
   launch measures only the launch.
3. **Synchronise at the end of each iteration**, so iterations do not pipeline into each other.
   This makes each sample an isolated measurement, which is what the mean and std are supposed
   to describe.
4. **Report mean ± sample stddev over n iterations**, never a single run and never a best-of.
5. **Verify before timing.** Every GPU result is compared element-by-element against an
   independent CPU reference (or cuBLAS, where the CPU is too slow), with a stated
   `rtol`/`atol`. A kernel that fails the check aborts the run instead of writing a fast wrong
   row.

**Ways this could still fool you, stated up front:** Colab's vCPUs are shared and virtualised,
so the CPU baselines are pessimistic; the GPU may down-clock when the card is shared; the first
iteration after a cold cache is slower than the rest, which is why there is a warmup; and
PCIe bandwidth on a VM is below a physical x16 link, so absolute GB/s numbers are lower than
bare metal even when the *ratios* are right.

---

## 1. Vector add — `phase_a/vector_add.cu`

`C[i] = A[i] + B[i]`.

**It is memory-bound, and arithmetic is irrelevant.** One add per element, but three 4-byte
accesses: read `A`, read `B`, write `C` = 12 bytes per float of output. The operation is
1 add / 12 bytes = 0.083 flops/byte. A T4 balances roughly 8.1 TFLOP/s FP32 against 320 GB/s,
i.e. ~25 flops/byte before it becomes compute-bound. This kernel is ~300× on the wrong side of
that line, so `occupancy`, `ILP` and shared memory are all noise; the only thing that matters is
how efficiently bytes are requested.

| Version | Mechanism | Why it helps |
|---|---|---|
| `cpu_reference` | one core, scalar loop | Not a fair fight, and not meant to be: it is the correctness oracle. Single-core DRAM bandwidth is a fraction of the GPU's, and the loop cannot hide latency. |
| `gpu_scalar` | 1 thread per element, 32-bit accesses | Full parallelism, but each thread issues three 32-bit transactions. A warp requests 32 × 4 B = 128 B per array, which is exactly one 128 B cache line — efficient, but it takes three instructions per element to do it. |
| `gpu_float4` | 1 thread per 4 elements, `float4` (128-bit) accesses | A warp now requests 4 × 128 B per array in one instruction per array. Fewer instructions per byte, and more bytes in flight per thread (memory-level parallelism), which is what actually saturates HBM. Expect it to approach the bandwidth ceiling; it cannot exceed it. |

**The honest ceiling.** If the measured GB/s lands near the card's spec bandwidth, the kernel is
done — there is nothing left to optimise, and any further "speedup" would have to come from
moving fewer bytes. That is why this is the sanity check for the whole project.

---

## 2. Parallel reduction — `phase_a/reduction.cu`

Sum 2²⁴ floats. The data is 64 MB, so the *floor* is one full read of DRAM. Everything the
three versions do differently is about what they do *besides* that one read.

### 2.1 Why the naive tree is the wrong shape

The textbook tree halves the active threads each step:

```
for (s = blockDim.x / 2; s > 0; s >>= 1)
    if (tid < s) data[tid] += data[tid + s];
```

Two problems, both structural:

- **Only one block.** A single block occupies one SM. The other 39 SMs on a T4 idle, so the
  64 MB read is issued by at most a thousand threads instead of tens of thousands. There is no
  way to saturate memory bandwidth that way.
- **Divergence and barriers.** Half the threads are idle after the first step, three quarters
  after the second, and every step needs a `__syncthreads()`.

### 2.2 `naive_global` — the tree in DRAM

Same tree, but the array lives in global memory. Every level is a round trip to DRAM plus a
barrier: read, add, write, sync, repeat, ~24 times for 2²⁴ elements. Global latency is hundreds
of cycles, and the barrier means nothing overlaps. This variant exists to be the bad baseline,
and its number is the cost of *not* using shared memory.

### 2.3 `shared_mem` — the tree on chip

Two stages:

1. Each of G blocks folds its strided slice into a shared-memory array, reduces it there, and
   writes **one** partial sum to global memory.
2. Block 0 reduces those G partials.

What changes: the ~24 global round trips become ~10 shared-memory ones (shared latency is
tens of cycles, not hundreds), the barriers are block-local instead of device-wide, and the
grid now spans many SMs, so the 64 MB read is issued in parallel.

**The bank-conflict detail that trips people up.** Shared memory is divided into 32 banks of
4 bytes. If thread `t` accesses `sdata[t]` sequentially, threads 0–31 hit 32 different banks —
fine. The *interleaved* pattern `sdata[tid] += sdata[tid + s]` also stays conflict-free because
a warp accesses 32 consecutive addresses. What causes a 32-way conflict is the *transposed*
pattern `sdata[tid * 32]`, where every thread lands in the same bank. This kernel uses
consecutive indexing on purpose.

**Why the last steps are wasted work.** Below 32 active threads, the whole reduction is inside
one warp, and a warp executes in lockstep — so `__syncthreads()` and shared-memory round trips
are pure overhead there. That is the hole the next version fills.

### 2.4 `warp_shuffle` — the tree in registers

`__shfl_down_sync(0xffffffff, v, offset)` moves a value from lane `t+offset` to lane `t`
*through registers*, with no memory and no barrier:

```
v += __shfl_down_sync(mask, v, 16);   // lanes 0..15 get lane+16
v += __shfl_down_sync(mask, v, 8);
v += __shfl_down_sync(mask, v, 4);
v += __shfl_down_sync(mask, v, 2);
v += __shfl_down_sync(mask, v, 1);    // lane 0 now holds the warp total
```

Five shuffle instructions replace five shared-memory accesses *and* five `__syncthreads()`
calls per warp. Each warp's lane 0 writes one partial to shared memory, then warp 0 reduces
those partials (32 of them at most) with the same shuffles.

**Why it is not dramatically faster.** Once you are reading 64 MB exactly once with enough parallelism,
you are at the memory-bandwidth floor. The reduction improvement is a *latency and instruction
count* improvement on top of an already bandwidth-bound kernel, so expect a modest gain over
`shared_mem` and a large gain over `naive_global`. If `warp_shuffle` and `shared_mem` come out
close, that is the correct and expected result, not a bug.

### 2.5 `cub_reference`

CUB's `DeviceReduce::Sum` is what a professional would ship: it does the same two-stage
structure with tuned block sizes, vectorised loads, and architecture-specific tuning. If this
row beats every hand-written version, the write-up says so — the point of writing the three
versions by hand is to understand what CUB is doing, not to beat it.

---

## 3. Matrix multiply — `phase_a/matmul.cu`

`C = A·B`, FP32, square.

**Arithmetic intensity is the whole story.** For one output element, the naive kernel does `N`
multiply-adds (`2N` flops) while reading two length-`N` vectors (`8N` bytes). That is
`0.25` flops/byte, versus the ~25 flops/byte a T4 needs before compute matters — roughly 100×
short. So the naive kernel is a memory-bandwidth demonstration wearing a matmul costume.

| Version | What it does differently | Why it matters |
|---|---|---|
| `cpu_naive` | triple `i,j,k` loop, one core | The oracle. Also shows the scale of the problem: one core cannot feed a GEMM at any size that matters. |
| `naive` | one thread per `C[row,col]` | Two separate inefficiencies: each thread re-reads `A[row,:]` and `B[:,col]` from DRAM, and `B` is read **column-wise**, so consecutive threads in a warp touch addresses `N` floats apart — one 32-byte sector per 4-byte value, i.e. ~8× the bytes actually needed. |
| `tiled` | 32×32 tile of `C` per block, `A` and `B` tiles staged in shared memory | Each element loaded into shared memory is reused 32 times by the inner loop, so global traffic per output element drops by ~32×. The kernel moves from memory-bound toward compute-bound, which is where the big jump comes from. |
| `cublas` | vendor library | The honest ceiling. It adds what this kernel deliberately omits (see below). |

**Why the tiled kernel is conflict-free.** A warp is 32 threads along `x` with `y` fixed. In
the inner loop it reads `As[ty][k]`: all 32 threads want the *same* address, which the hardware
broadcasts — no conflict. Then `Bs[k][tx]`: 32 consecutive `tx` values, so 32 consecutive banks
— no conflict. The conflict-free property comes from storing `Bs` transposed (indexed
`[k][tx]` rather than `[tx][k]`), and that is also why the load of `Bs` is `B[t*32+ty][blockX*32+tx]`:
coalesced on the way in, conflict-free on the way out.

**The gap to cuBLAS is a finding, not a failure.** A production GEMM adds:

- **register blocking** — each thread computes an 8×8 micro-tile, so every value fetched from
  shared memory is used 8 times from registers, cutting shared-memory traffic by ~8×;
- **double buffering** — the next global→shared load overlaps the current phase's FMA work,
  hiding DRAM latency instead of stalling on it (and on Ampere+ this becomes `cp.async`);
- **swizzled shared-memory layouts and larger tiles**, plus per-architecture tuning.

This repo stops at the textbook tiled version because the remaining steps are a different
project. The write-up reports the gap as the cost of those missing techniques.

**Edge handling.** `N` is not always a multiple of 32. Out-of-range tile entries are
**zero-filled** rather than branched around, so the inner loop stays unconditional and the extra
terms contribute nothing. Branching in the inner loop would reintroduce warp divergence into the
hottest code in the repo.

---

## 4. Streams and pinned memory — `phase_a/streams.cu`

The pipeline per chunk is `H2D copy → kernel → D2H copy`, four chunks, 64 MB total.

**Two separate optimisations, measured separately.** It is easy to claim "streams made it
faster" when the real win was pinned memory, so this file isolates them:

| Version | What changed | Expected effect |
|---|---|---|
| `sync_pageable` | pageable `malloc` buffers, default stream, `cudaMemcpy`, device sync per chunk | Baseline. Pageable memory cannot be DMA'd directly: the driver stages through a pinned bounce buffer and the copy call blocks. Nothing overlaps. |
| `sync_pinned` | same structure, `cudaMallocHost` buffers | Removes the bounce buffer and lets the copy be a real DMA. Same amount of serialisation, less per-copy overhead. |
| `streams2_pinned` | two non-blocking streams, `cudaMemcpyAsync` + kernel on the same stream | The actual overlap: chunk 1's kernel runs while chunk 0's copy-back or chunk 2's copy-in is in flight. |
| `streams3_pinned` | three streams, four chunks | Same mechanism with a different division. The 4th chunk lands on a stream whose earlier work has drained, so it runs alone at the tail. |

**Why overlap needs all three ingredients.** A non-blocking stream, an *async* copy, and
pinned source memory. Async copy from pageable memory is silently synchronous; and the legacy
default stream implicitly synchronises with other streams, which is why the streams are created
with `cudaStreamNonBlocking`.

**Why the speedup is not linear.** Total wall time is the *maximum* over streams, not the sum.
With four chunks over two streams the tail is one chunk long; over three streams it is also one
chunk long but lands unevenly. The more chunks you split into, the smaller the tail relative to
the total — which is exactly the trade-off a real pipeline makes with more frames in flight.
Note also that a copy and a kernel only overlap if the copy engine and the SMs are both busy;
a kernel that already saturates DRAM will slow the copy down instead of hiding it.

---

## 5. Fused preprocess — `phase_b/preprocess.cu`

The model wants a `3×640×640` FP16 RGB tensor, normalised, CHW. The camera hands over a
`1080×1920×3` uint8 BGR image. The conventional CPU path is five operations:

```
resize (bilinear)  -> write uint8 640x640x3
cvtColor BGR->RGB  -> read + write the same 1.2 MB
to float           -> write 4.9 MB fp32
transpose HWC->CHW -> read 4.9 MB + write 4.9 MB
normalise          -> read 4.9 MB + write 4.9 MB
```

Call it ~30 MB of traffic and five launches to produce a 2.4 MB tensor. The fused kernel does:

> one read of the source pixels it actually samples (~6 MB, upper bound), one write of 2.4 MB.

**What the kernel does per output pixel**, in one pass over *destination* pixels:

1. Map the destination pixel back through the letterbox transform to a source coordinate.
   Slightly outside the source image = letterbox padding, so write zeros — that is what makes
   the transform consistent with what the model was trained on.
2. Bilinear sample: four taps per channel, with edge clamping.
3. Reorder BGR → RGB at the same time (it is only a change of which of the three loads is
   which channel).
4. Normalise in FP32: `(v/255 - mean) / std`.
5. Convert once, at the store, to FP16 or FP32, into CHW layout at `c*H*W + y*W + x`.

**Why it is memory-bound, not compute-bound.** Per output pixel: 12 loads, ~20 flops, 3 stores.
The flops are trivially hidden. The cost is the 12 loads and their cache locality, which is why
the block is 2-D (32×8) over the destination: neighbouring threads sample neighbouring source
pixels, so the four taps of adjacent threads overlap in L1/L2 and the loads coalesce.

**Why the maths is done in FP32 even for an FP16 output.** FFMA in FP32 is not the bottleneck
here, and doing the normalisation in FP16 would add rounding to a step that is compared against
a CPU reference. Convert once, at the end.

**Precision and the stated tolerance.** The GPU samples bilinearly in FP32; the CPU reference
does the same arithmetic in NumPy. For FP32 output the check is `atol=rtol=1e-4`; for FP16 it is
`atol=2e-3`, because FP16 carries ~3 significant decimal digits and a stricter tolerance would
fail for reasons that have nothing to do with the kernel being wrong. Both numbers are printed
and written into the CSV `extra` column.

**The letterbox geometry is computed on the host**, not in the kernel, precisely so the CPU
reference and the GPU use bit-identical `scale`/`pad_x`/`pad_y`. Recomputing it on the device
would be an unnecessary second source of disagreement.

---

## 6. Inference ladder — `phase_b/infer_bench.py`

Same model, same input, four backends. Two things determine the ranking.

**Tensor cores.** A T4 does ~8.1 TFLOP/s in FP32 but ~65 TFLOP/s in FP16 — an 8× peak
difference. At batch 1 the kernel is heavily latency-bound, so the realised gain is much
smaller than 8×; the measured FP32→FP16 step is the honest version of that claim. Note that FP16
also halves the bytes each layer reads, which helps the memory-bound layers independently of the
tensor cores.

**Graph-level work.** ONNX Runtime applies graph optimisations and picks kernels per operator;
TensorRT goes further with layer fusion, precision calibration and kernel autotuning. Both can
beat eager PyTorch at batch 1, and both can lose once the batch gets big enough that PyTorch's
kernels are already saturating the SMs. That is why the ladder is measured rather than assumed.

**What is inside the timed region.** Preprocessing is *not* (it is measured separately); decode
+ NMS and the device→host transfer of the detections *are*, because a camera loop has to pay for
them too. The CSV `extra` column records whether decode was included, so the rows cannot be
silently compared on unequal terms.

**Failures are first-class.** If the CUDA execution provider is missing, ONNX Runtime falls back
to CPU *silently* by default — which would produce a plausible-looking row that means nothing.
The code checks `get_available_providers()` and refuses to run instead. TensorRT's install can
drag in CUDA packages that break the preinstalled PyTorch, so it is attempted last, and a
failure is written to `results/failures.md` with the traceback rather than skipped.

---

## 7. Pipelined loop and batching — `phase_b/pipeline.py`

**The timeline the pipeline is trying to achieve:**

```
stream A (copy):    H2D k   H2D k+1   H2D k+2
stream B (compute):         infer k    infer k+1
```

Frame *k+1* is being copied in while frame *k* is being inferred. Without the overlap, the
inference engine idles for the duration of every copy, and at 1080p that stall is a meaningful
fraction of the frame budget.

**What is genuinely overlapped:** the host→device copy and the Python-side preparation of the
next frame, against the GPU inference of the current one.

**What is not, and is not claimed:** two inferences running concurrently on one backend session.
PyTorch and ONNX Runtime own their internal streams and will serialise work submitted to the
same session; pretending otherwise would produce a fake speedup. Postprocessing and Python
overhead also stay on the critical path.

**Amdahl's law is the point of the serial-vs-pipelined comparison.** If inference is 60% of the
serial frame time, removing the copy stall entirely can only save the other 40%. The measured
speedup should be read as "what eliminating the copy stall was worth on this machine", and the
write-up says exactly that instead of implying a 2× pipeline.

**Batching trades latency for throughput.** Feeding 4 frames as one batch of 4 gives the GPU
more parallel work per launch, so total throughput rises — but the latency of any single frame
rises too, and FPS *per feed* is what a multi-camera deployment cares about. That is why the
batching table reports both total FPS and FPS per feed: the second one is the number that decides
how many cameras one GPU can serve.

---

## 8. Profiling, and what it could not do

- **Nsight Systems (`nsys`)** works on Colab. The timeline shows copy and kernel overlap
  directly (which is how the streams claim is checked, rather than trusted), and the
  `cuda_gpu_kern_sum` table gives per-kernel duration. Stats text is saved under `results/`;
  the `.nsys-rep` files are downloaded and opened in the GUI locally.
- **Nsight Compute (`ncu`)** needs permission to read GPU performance counters, which requires
  privileges a Colab runtime does not grant; it fails with `ERR_NVGPUCTRPERM`. So achieved
  occupancy, memory-throughput percentages and stall reasons are **not** available here. The
  substitute is CUDA event timings plus the nsys kernel table, and the limitation is recorded in
  `results/failures.md` with the actual error rather than quietly omitted.
