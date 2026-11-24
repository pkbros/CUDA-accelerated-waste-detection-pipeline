# CUDA-Accelerated Conveyor Waste Detection Pipeline

A CUDA performance study of a conveyor-belt waste detection pipeline: hand-written GPU kernels,
a pretrained YOLOv8n on real conveyor frames, and profiling that shows what got faster and why.

![Where the time goes](docs/ppt_pack/03_latency_breakdown.png)

**For the interview:** [slide-image pack](docs/ppt_pack/index.md) (14 PNGs, every number from a
CSV) and [plain-English project explainer](docs/PROJECT_EXPLAINED.md). Slide 12 compares the two
preserved runs side by side. The older demo video is left on disk but is no longer the deliverable.

![Pipeline overview](docs/overview.png)

> **Status: prototype.** Every number comes from a CSV written by a run on a Colab
> **Tesla T4 (sm_75, CUDA 13.0)**. Two complete real-data runs are preserved under
> `results/runs/`: `pretrained_coco/` (COCO YOLOv8n) and `finetuned/` (30-epoch ZeroWaste
> fine-tune, measured mAP50 0.4173). The current `results/*.csv` are the fine-tuned run;
> the table below is the preserved pretrained-COCO run, kept for comparison.

The rule this repo is built around: *if a benchmark did not run, it does not appear.*
`results/summary.md` is generated from `results/*.csv`, and anything missing shows up as
**NOT RUN** rather than being estimated.

---

## Results at a glance

Measured on a **Tesla T4 (sm_75, 14.56 GiB, torch 2.14.1+cu130, CUDA 13.0)**.
Full tables: [`results/summary.md`](results/summary.md). The numbers below are the
**preserved pretrained-COCO run**; see
[`12_pretrained_vs_finetuned.png`](docs/ppt_pack/12_pretrained_vs_finetuned.png) for the
side-by-side comparison with the fine-tuned run.

| What | Baseline | Ours | Speedup |
|---|---|---|---|
| **End-to-end, 120 real conveyor frames** (demo timings) | 63.9 ± 10.9 FPS | 93.0 ± 14.9 FPS | **1.46x** |
| Preprocess stage in that loop | OpenCV CPU 4.280 ± 2.066 ms | fused CUDA 0.441 ± 0.230 ms | **9.7x** |
| Fused preprocess, host frame to ready GPU tensor (1920x1080) | OpenCV+upload 6.283 ms | 1.378 ms | **4.56x** |
| Kernel only, input already resident | 0.049 ms | - | - (bandwidth-limited) |
| Fused kernel vs plain PyTorch GPU ops, pinned H2D + kernel | PyTorch 0.673 ms | 0.605 ms | **1.11x** |
| Reduction, N=2^24 | CUB 0.261 ms | ours 0.290 ms | 0.90x of CUB |
| Tiled matmul, N=1024 | cuBLAS 6130 GFLOP/s | ours 732 GFLOP/s | 12% of cuBLAS |
| Copy+kernel, 64 MiB, 4 chunks | pageable 31.16 ms | 3 streams 7.25 ms | **4.30x** |
| Vector add, N=5e7 | scalar 262.3 GB/s | float4 248.8 GB/s | no win (at HBM ceiling) |

Both demo pipelines used the **same pretrained YOLOv8n and the same real frames**; only the
preprocess, backend and transport changed. Mean detections were **1.125 per frame in both**
(76 of 120 frames had boxes), which is the sanity check that the speedup is not coming from
skipping detections. ncu reported ~95% DRAM throughput and ~6.5% SM throughput for
`vector_add_float4`, confirming it is memory-bound.

### How to read these numbers (reviewer corrections)

The active `results/*.csv` are the fine-tuned run; a few headline figures need qualification:

- **FP16 is batch-dependent, not a blanket win.** In `infer.csv` FP16 is *slower* at batch 1
  (13.711 ms vs FP32 10.932 ms) but ~2× faster at batch 4 (10.285 vs 19.504 ms) and batch 8
  (16.658 vs 42.406 ms). Quote the batch.
- **Postprocess = 0 is by construction.** Box decode + NMS and the device-to-host readback are
  folded into the backend's inference timing (`phase_b/demo_timings.py`, `phase_b/backends.py`),
  so the postprocess column is literally 0.000 ms — it is not free work.
- **Backends and an unrecorded choice.** The baseline is PyTorch FP32; the optimized loop tries
  ONNX Runtime CUDA FP32 first and falls back to PyTorch FP32. The CSV does **not** record which
  backend was actually chosen for the demo loop (ORT CUDA FP32 measured 5.259 ms at batch 1 in
  `infer.csv`).
- **Agreement score excludes empty frames.** 60 frames with boxes agree at IoU ≈ 1.0; the 60
  empty frames were scored IoU 0 in an earlier "50% / 0.500" figure. The corrected agreement is
  over frames *with boxes* only: 100%, IoU 1.0 (`pipeline_agreement.csv`).
- **Pipelining regressed here.** In `batching.csv` the pipelined loop is slower than serial at
  every feed count — a measured regression in this prototype, not a win.
- **Preprocess is ~5×, not 84×.** At 1920×1080 end-to-end is ~1.43 ms vs OpenCV CPU 7.20 ms
  (~5×); kernel-only is ~0.049 ms, but plain PyTorch GPU ops (~0.684 ms) vs fused H2D + kernel
  (~0.57 ms) is only ~1.2×. Most of the win is moving work onto the GPU, not the fusion itself.

### What is *not* in these results

- TensorRT is absent (`ModuleNotFoundError: No module named 'tensorrt'`); ONNX Runtime CUDA
  is the top rung that actually ran, and its batch 4/8 attempts produced no rows.
- The first synthetic run is archived in `results/synthetic_run1/` and is never mixed with
  real-data numbers.
- The fine-tuned run's mAP is measured on this 300-image val split only — it is a prototype
  number, not a claim about a deployed detector.

The later **fine-tuned** run adds measured mAP50 0.4173 / mAP50-95 0.3247, batch 1/4/8 rows
in `infer.csv`, and `source=real_video` in `pipeline.csv` / `batching.csv`. Both runs are
preserved verbatim under `results/runs/`.

---

## 1. Problem

Recycling sorting lines need detections at camera frame rate. The interesting engineering is
not the model — a pretrained YOLO is good enough — it is the part around it: getting pixels
from a camera into a tensor fast enough that the GPU is not waiting on the CPU, and knowing
*which* of the many possible optimisations actually pays on the hardware you have.

So the pipeline is decomposed into stages that can each be measured, and each stage has a
baseline it must beat:

| Stage | Baseline | Optimised version |
|---|---|---|
| Preprocess | OpenCV + NumPy on CPU, 5 separate passes | one fused CUDA kernel, one pass |
| Inference | PyTorch FP32 | FP16, ONNX Runtime CUDA, TensorRT (stretch) |
| Transport | pageable memory, default stream | pinned buffers, 2–3 streams |
| Scaling | 1 feed, batch 1 | 1/2/4 feeds, batched |

---

## 2. Architecture

```mermaid
flowchart LR
    CAM[Camera frame<br/>HWC uint8 BGR] --> PIN[Pinned host buffer<br/>cudaMallocHost]
    PIN -->|cudaMemcpyAsync on stream s| PRE[Fused preprocess kernel<br/>letterbox bilinear resize<br/>BGR to RGB, normalize<br/>HWC to CHW, uint8 to FP16]
    PRE -->|event recorded on s| INF[Inference<br/>YOLOv8n FP16 on T4]
    INF --> POST[Decode + NMS]
    POST --> OUT[Detections<br/>boxes, scores, classes]
    PRE -.-> NEXT[Frame k+1 H2D<br/>overlaps infer of frame k]
```

Two ideas carry the design:

1. **Fuse the preprocess.** Resize, colour conversion, normalisation, layout change and dtype
   conversion all read and write the same 640×640×3 tensor. Doing them in one pass over
   destination pixels removes the intermediate tensors and four kernel launches, which is why
   the win is mostly avoided memory traffic rather than arithmetic.
2. **Overlap copy with compute.** The host-to-device copy of frame *k+1* runs on a copy stream
   while inference of frame *k* runs on the GPU. This is a real overlap; running two
   inferences concurrently on one backend session is *not* claimed, because most backends
   serialise internally.

---

## 3. Results

Every table lives in [`results/summary.md`](results/summary.md), generated by:

```bash
python tools/summary.py
```

which reads the CSVs written by the benchmarks and emits the tables, including the GPU the
numbers came from. Run it after the notebook; it is the only source of numbers in this repo.

The current CSVs are from the first **synthetic** run, archived under
`results/synthetic_run1/`. Its preprocess speedup claim has been withdrawn: the GPU time
excluded the host→device upload while the OpenCV baseline excluded it too, and that
baseline had `max_abs_err=2.1` against the reference. The benchmark has been rewritten to
report five honestly-scoped timings (Task 1), but the new numbers do not exist yet — they
need a T4 run. Nothing here is estimated in the meantime.

`results/failures.md` records what did not work in the target environment (for example
Nsight Compute on Colab, or a TensorRT install that fights the preinstalled PyTorch), with the
actual error text. Those rows are absent from the tables on purpose.

---

## 4. Layout

```
include/bench_common.h      shared timing (CUDA events), correctness and CSV helpers
phase_a/                    standalone nvcc benchmarks, one binary each
  vector_add.cu             sanity check + bandwidth ceiling
  reduction.cu              global -> shared -> warp shuffle -> CUB
  matmul.cu                 naive -> 32x32 tiled -> cuBLAS
  streams.cu                pageable vs pinned, 1/2/3 streams
phase_b/                    the applied pipeline (Python + one CUDA extension)
  preprocess.cu             the fused kernel
  preprocess_ext.py         torch.utils.cpp_extension build + CPU reference + bench
  backends.py               pt_fp32 / pt_fp16 / ort_cuda / trt_fp16
  infer_bench.py            the inference ladder
  pipeline.py               serial vs pipelined, plus 1/2/4 feeds
  data_prep.py, finetune.py dataset download/convert and the optional light fine-tune
notebooks/run_all.ipynb     the single entry point
results/                    *.csv, summary.md, failures.md, profiling text
docs/kernel_notes.md        why each kernel is faster or slower
tools/summary.py            CSV -> summary.md
```

---

## 5. Reproduce

On Colab (free tier is enough):

1. Upload the repo folder, or set `GIT_URL` in the first cell of the notebook.
2. `Runtime > Change runtime type > Hardware accelerator = T4 GPU`.
3. Open `notebooks/run_all.ipynb` and `Runtime > Run all`.

The notebook installs the matching CUDA toolkit if `nvcc` is absent, builds Phase A with
`make`, builds the preprocess extension, runs the inference ladder, the pipeline and the
batching experiment, then generates and downloads `results/`.

Locally, the same steps by hand:

```bash
make -C phase_a ARCH=sm_75 run        # Phase A -> results/*.csv
python phase_b/preprocess_ext.py      # fused kernel vs CPU
python phase_b/export_model.py
python phase_b/infer_bench.py
python phase_b/pipeline.py
python tools/summary.py
```

`ARCH=sm_75` is a T4 (and a GTX 1650). `ARCH=native` lets `nvcc` detect the local card.

---

## 6. What each optimisation was supposed to do, and what happened

Full reasoning, including the parts that turned out to be wrong, is in
[`docs/kernel_notes.md`](docs/kernel_notes.md). Short version — the "measured" column
comes from [`results/summary.md`](results/summary.md), not from expectation:

| Technique | Mechanism | Measured on this T4 | Where |
|---|---|---|---|
| 128-bit loads (`float4`) | fewer memory transactions per element | **no win**: 2.412 ms vs 2.287 ms scalar — both already at ~250–260 GB/s | `vector_add.cu` |
| Shared-memory reduction | removes a global round trip per tree level | 0.290 ms vs **CUB 0.261 ms** (within ~11%); the "80×" figure is only against a deliberately slow one-block baseline | `reduction.cu` |
| Warp shuffle | removes intra-warp barriers and shared traffic | 0.295 ms — essentially tied with shared memory (bandwidth floor) | `reduction.cu` |
| Shared-memory tiling | reuse a tile 32× so GEMM turns compute-bound | 732 vs 667 GFLOP/s; cuBLAS still 6130 | `matmul.cu` |
| Pinned memory | lets the copy engine DMA without a bounce buffer | 11.10 ms vs 31.16 ms pageable (2.81×) | `streams.cu` |
| Multiple streams | copy of one chunk overlaps compute of another | 7.25 ms vs 11.10 ms pinned (1.53× more) | `streams.cu` |
| Kernel fusion | one pass instead of five over intermediates | fair re-measurement: end-to-end 1.378 ms vs OpenCV+upload 6.283 ms (**4.56×**); kernel-only 0.049 ms; 0.605 ms with pinned H2D. The old 84× is withdrawn — it compared unequal scopes and that OpenCV row had `max_abs_err=2.1`. | `preprocess.cu` |
| FP16 | tensor cores are ~8× FP32 *peak* on paper | **slower**: 10.030 vs 8.079 ms at batch 1 | `infer_bench.py` |
| ONNX Runtime CUDA | graph optimisation + tuned kernels | 7.916 ms, 1.02× over eager FP32 | `infer_bench.py` |
| Batching / pipeline | amortise launches, overlap copy with infer | serial holds ~106 FPS/feed; pipelined **regresses** at 2/4 feeds | `pipeline.py` |

---

## 7. Honest limitations

- **T4 (sm_75).** No `cp.async`, no TMA, no BF16, no distributed shared memory. Anything that
  relies on Ampere-or-later features is out of scope, and that is a hardware limit, not a
  design choice.
- **Colab is not bare metal.** vCPUs are shared and virtualised, so the CPU baselines are
  noisier and slower than a desktop would be, which *flatters* the GPU speedups. Absolute PCIe
  bandwidth is also below a physical x16 link, and the session can be reclaimed mid-run.
- **The old benchmark CSVs are synthetic** (archived in `results/synthetic_run1/`). Both
  current runs use real ZeroWaste conveyor frames and are preserved under `results/runs/`.
- **Two detectors, two runs.** The preserved `pretrained_coco/` run used COCO YOLOv8n as a
  fixed workload. The active `finetuned/` run trained 30 epochs on the real ZeroWaste subset
  (1,415 train / 300 val) and reports measured mAP50 0.4173. The core CUDA result — the
  1.4–1.5x pipeline speedup from preprocess, backend and streams — holds in both runs; see
  slide 12 for the comparison.
- **Licence:** the Roboflow API reports CC BY 4.0 for the hosted ZeroWaste copy, resolving the
  conflicting CC BY-NC prose on the web page. This is prototype interview use, not
  redistribution; see `docs/data_sources.md`.
- **GEMM stops at tiling.** No register blocking or double buffering, so cuBLAS stays well
  ahead. That gap is reported, not hidden.
- **The pipelined loop overlaps copies, not inferences.** See §2. At 2 and 4 feeds the
  measured pipelined batch loop is *slower* than serial; the buffer bookkeeping costs more
  than the copy stall it removes on these small tensors.
- **Nsight Compute did work on this Colab runtime**, contrary to the usual
  `ERR_NVGPUCTRPERM` limitation. It reported ~95–96% DRAM throughput, ~99–101% achieved
  occupancy and ~6.7% SM throughput for `vector_add_float4` — direct confirmation that the
  kernel is memory-bound. Its kernel *durations* are instrumentation-inflated (1674 ms for
  a 1M-element add) and are deliberately excluded from `summary.md`. That exclusion is
  enforced in `tools/summary.py` by skipping `profiling_*.csv`.

---

## 8. Data and licences

Dataset choice, licence status and what could and could not be verified are in
[`docs/data_sources.md`](docs/data_sources.md). Licences are recorded as `verified` or
`unverified` — an unverified licence is never presented as cleared for redistribution.
