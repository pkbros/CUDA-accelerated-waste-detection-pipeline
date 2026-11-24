<!--
TEMPLATE RULES (delete this comment block in the final README)
- Every {{PLACEHOLDER}} is filled by tools/build_readme.py from results/readme_values.json.
- Do not hand-type numbers. If a value does not exist, the build writes "NOT RUN" and logs it.
- Sentences marked [VERDICT] must be generated from the data (see AGENT_TASKS_readme.md). Never keep a sentence that the data contradicts.
- Images live in docs/ppt_pack/. Use relative links.
-->

# CUDA Conveyor Waste Detection Pipeline

A CUDA performance study of a conveyor-belt waste detector on a {{GPU_MODEL}}: hand-written GPU kernels, a fine-tuned YOLOv8n, and profiling that shows what got faster, what did not, and why.

**Status: prototype.** Built to learn and show GPU engineering with real measurements. It is not a deployed system.

![Where the milliseconds go](docs/ppt_pack/03_latency_breakdown.png)

---

## TL;DR

- **What it is:** a four-stage video pipeline (decode, preprocess, inference, postprocess) for waste on a conveyor belt, with each stage timed and optimized on the GPU.
- **Main result:** preprocessing went from {{PRE_BASE_MS}} ms to {{PRE_OPT_MS}} ms per frame, and the whole pipeline went from {{BASE_FPS}} to {{OPT_FPS}} FPS ({{FPS_RATIO}}x), measured over {{N_FRAMES}} real frames.
- **What limits it now:** {{BOTTLENECK_SENTENCE}}
- **Detector:** YOLOv8n fine-tuned on {{N_TRAIN}} ZeroWaste images, mAP50 {{MAP50}} on {{N_VAL}} held-out val images. This is a small subset and a short fine-tune, so treat the detector as a working placeholder, not a finished model.
- **Honest negatives:** {{NEGATIVES_ONE_LINE}}

---

## Why this project exists

Recycling lines need to spot and sort waste fast. Before any model can run, every camera frame has to be decoded, resized, color-converted, normalized and copied to the GPU. At high frame rates, that plumbing can cost as much as the model itself.

This project takes one realistic pipeline and asks a simple question: **where does the time go, and which GPU techniques actually pay off?** Each technique is built, checked for correctness, timed with proper methodology, and reported even when it did not help.

**What this project is not:** a state-of-the-art detector, a production system, or a robotics integration. The footage is {{FOOTAGE_DESCRIPTION}}.

---

## The pipeline

```mermaid
flowchart LR
    A[Video frame<br/>uint8 HWC BGR] --> B[Decode<br/>CPU]
    B --> C[Preprocess<br/>resize, BGR to RGB,<br/>normalize, HWC to CHW]
    C --> D[Inference<br/>YOLOv8n]
    D --> E[Postprocess<br/>NMS, boxes]
    E --> F[Detections<br/>class, score, box]
```

| Stage | Input | Output | Baseline | Optimized | Baseline ms | Optimized ms |
|---|---|---|---|---|---|---|
| Decode | video file | uint8 HWC frame | OpenCV (CPU) | OpenCV (CPU) | {{DECODE_BASE_MS}} | {{DECODE_OPT_MS}} |
| Preprocess | uint8 HWC frame | {{PRE_OUT_DESC}} tensor on GPU | OpenCV on CPU, then upload | one fused CUDA kernel | {{PRE_BASE_MS}} | {{PRE_OPT_MS}} |
| Inference | preprocessed tensor | raw predictions | {{BASE_BACKEND}} | {{OPT_BACKEND}} | {{INFER_BASE_MS}} | {{INFER_OPT_MS}} |
| Postprocess | raw predictions | boxes, classes, scores | {{POST_DESC_BASE}} | {{POST_DESC_OPT}} | {{POST_BASE_MS}} | {{POST_OPT_MS}} |

Mean per frame over {{N_FRAMES}} frames on {{GPU_MODEL}}. Baseline and optimized both run model inference on the GPU. The difference is preprocessing, the inference backend, and pipelining.

![Four stages](docs/ppt_pack/01_pipeline_stages.png)

---

## Results at a glance

| What | Baseline | Optimized | Change | Image |
|---|---|---|---|---|
| End-to-end FPS ({{N_FRAMES}} real frames) | {{BASE_FPS}} | {{OPT_FPS}} | {{FPS_RATIO}}x | [04](docs/ppt_pack/04_end_to_end_fps.png) |
| Time per frame | {{BASE_TOTAL_MS}} ms | {{OPT_TOTAL_MS}} ms | {{TOTAL_RATIO}}x | [03](docs/ppt_pack/03_latency_breakdown.png) |
| Preprocess, full path from host frame to GPU tensor | {{PRE_OPENCV_MS}} ms (OpenCV CPU) | {{PRE_E2E_MS}} ms (fused kernel) | {{PRE_E2E_RATIO}}x | [06](docs/ppt_pack/06_preprocess_honest.png) |
| Preprocess, kernel only | {{PRE_TORCH_MS}} ms (plain PyTorch GPU ops) | {{PRE_KERNEL_MS}} ms | {{PRE_KERNEL_VS_TORCH}}x | [06](docs/ppt_pack/06_preprocess_honest.png) |
| Reduction, 16.7M floats | {{RED_NAIVE_MS}} ms (naive) | {{RED_SHUFFLE_MS}} ms (warp shuffle) | {{RED_RATIO}}x, CUB library: {{RED_CUB_MS}} ms | [07](docs/ppt_pack/07_kernels_reduction_matmul.png) |
| Matmul, N=1024 | {{MM_NAIVE_GFLOPS}} GFLOPS (naive) | {{MM_TILED_GFLOPS}} GFLOPS (tiled) | cuBLAS: {{MM_CUBLAS_GFLOPS}} ({{MM_TILED_PCT}}% reached) | [07](docs/ppt_pack/07_kernels_reduction_matmul.png) |
| Host to GPU copy + kernel, 64 MB | {{STREAM_PAGEABLE_MS}} ms (pageable) | {{STREAM_BEST_MS}} ms ({{STREAM_BEST_NAME}}) | {{STREAM_RATIO}}x | [08](docs/ppt_pack/08_memory_bound_and_streams.png) |
| Detector quality (val) | | mAP50 {{MAP50}}, mAP50-95 {{MAP5095}} | | [11](docs/ppt_pack/11_detection_quality.png) |

**Reading this table honestly:** [VERDICT] {{HEADLINE_VERDICT}}

---

## Same frame, two pipelines

Both panels use the same real frame and the same fine-tuned detector. Left is the baseline, right is the optimized path.

![Baseline vs optimized](docs/ppt_pack/02_side_by_side_A.png)

Boxes agree on {{AGREE_PCT}}% of frames (mean IoU {{MEAN_IOU}} over matched boxes). [VERDICT] {{AGREEMENT_VERDICT}}

---

## What I built

### 1. Fused preprocess kernel

**Problem.** A camera frame arrives as `uint8` in height-width-channel order, BGR. The model wants a `{{PRE_OUT_DESC}}` tensor, RGB, scaled to 0 to 1, in channel-first order, letterboxed to {{IMGSZ}}x{{IMGSZ}}. Done step by step, each step reads and writes the whole image again.

**Approach.** One CUDA kernel does resize (bilinear), letterbox padding, BGR to RGB, normalize, and HWC to CHW in a single pass. Each GPU thread computes one output pixel, so memory is touched once.

**Measured** (source image {{PRE_SRC_RES}}, {{PRE_RUNS}} runs, mean):

| Variant | Scope | ms |
|---|---|---|
| `opencv_cpu` | CPU resize and convert, then pageable upload | {{PRE_OPENCV_MS}} |
| `torch_gpu_ops` | same steps as plain PyTorch GPU ops | {{PRE_TORCH_MS}} |
| `kernel_only` | fused kernel, input already on GPU | {{PRE_KERNEL_MS}} |
| `h2d_plus_kernel` | pinned upload plus fused kernel | {{PRE_H2D_MS}} |
| `end_to_end` | host frame to ready GPU tensor | {{PRE_E2E_MS}} |

**Correctness.** Output matches the CPU reference within {{PRE_TOL}} (max abs error {{PRE_MAXERR}}). Timing was only trusted after this check passed.

[VERDICT] {{PRE_VERDICT}}

![Preprocess scopes](docs/ppt_pack/06_preprocess_honest.png)

Code: `phase_b/preprocess.cu`, `phase_b/preprocess_ext.py`.

### 2. Parallel reduction

**Problem.** Sum 16.7 million floats. Easy to write, easy to write slowly.

**Three versions.**
- **Naive global:** threads add straight in global memory, with many round trips and many synchronizations.
- **Shared memory:** each block reduces in fast on-chip shared memory and writes one partial result.
- **Warp shuffle:** threads in a warp exchange values in registers with no shared memory and no block-wide sync for the last steps.

| Variant | ms | GB/s |
|---|---|---|
| naive global | {{RED_NAIVE_MS}} | {{RED_NAIVE_GBS}} |
| shared memory | {{RED_SHARED_MS}} | {{RED_SHARED_GBS}} |
| warp shuffle | {{RED_SHUFFLE_MS}} | {{RED_SHUFFLE_GBS}} |
| CUB (NVIDIA library, reference) | {{RED_CUB_MS}} | {{RED_CUB_GBS}} |

The naive version is deliberately slow, so the comparison that matters is against CUB: my best kernel is within {{RED_VS_CUB_PCT}}% of it. [VERDICT] {{RED_VERDICT}}

Code: `phase_a/reduction.cu`.

### 3. Matrix multiply

Naive, 32x32 shared-memory tiled, and cuBLAS as the ceiling.

| N | naive GFLOPS | tiled GFLOPS | cuBLAS GFLOPS | tiled vs cuBLAS |
|---|---|---|---|---|
| {{MM_N}} | {{MM_NAIVE_GFLOPS}} | {{MM_TILED_GFLOPS}} | {{MM_CUBLAS_GFLOPS}} | {{MM_TILED_PCT}}% |

Tiling helps less than the textbook suggests here. What is missing compared with cuBLAS: register blocking, vectorized loads, double buffering, and tensor cores. {{MM_EXTRA_SENTENCE}}

Code: `phase_a/matmul.cu`.

### 4. Memory-bound kernels (vector add)

Vector add does one addition for every 12 bytes moved, so it is limited by memory bandwidth, not compute. Nsight Compute confirms it: DRAM throughput {{DRAM_PCT}}% of peak, SM throughput {{SM_PCT}}% of peak. [VERDICT] {{FLOAT4_VERDICT}}

![Memory bound and streams](docs/ppt_pack/08_memory_bound_and_streams.png)

Code: `phase_a/vector_add.cu`.

### 5. Pinned memory and streams

Copying from normal (pageable) host memory is slow because the driver stages it first. Pinned memory skips that, and multiple CUDA streams let copies and kernels overlap.

| Variant | ms |
|---|---|
| sync, pageable | {{STREAM_PAGEABLE_MS}} |
| sync, pinned | {{STREAM_PINNED_MS}} |
| 2 streams, pinned | {{STREAM_S2_MS}} |
| 3 streams, pinned | {{STREAM_S3_MS}} |

[VERDICT] {{STREAM_VERDICT}}

Code: `phase_a/streams.cu`.

### 6. Inference backends

Same fine-tuned YOLOv8n, different runtimes, batch sizes 1, 4, 8. Times are **{{INFER_UNIT_DESC}}**.

| Backend | batch 1 | batch 4 | batch 8 |
|---|---|---|---|
| PyTorch FP32 | {{PT32_B1}} | {{PT32_B4}} | {{PT32_B8}} |
| PyTorch FP16 | {{PT16_B1}} | {{PT16_B4}} | {{PT16_B8}} |
| ONNX Runtime CUDA | {{ORT_B1}} | {{ORT_B4}} | {{ORT_B8}} |
| TensorRT FP16 | {{TRT_B1}} | {{TRT_B4}} | {{TRT_B8}} |

[VERDICT] {{FP16_VERDICT}} {{TRT_NOTE}}

![Inference ladder](docs/ppt_pack/05_inference_ladder.png)

### 7. Pipelined loop and multiple feeds

Overlap decode, upload, inference and postprocess across CUDA streams, and simulate 1, 2 and 4 camera feeds.

| Feeds | serial FPS per feed | pipelined FPS per feed |
|---|---|---|
| 1 | {{FEED_SERIAL_1}} | {{FEED_PIPE_1}} |
| 2 | {{FEED_SERIAL_2}} | {{FEED_PIPE_2}} |
| 4 | {{FEED_SERIAL_4}} | {{FEED_PIPE_4}} |

[VERDICT] {{PIPELINE_VERDICT}} {{PIPELINE_CAUSE_SENTENCE}}

![Multi-feed](docs/ppt_pack/09_multi_feed.png)

---

## What did not work

I report these on purpose. They are the most useful part of the measurements.

![What did not work](docs/ppt_pack/10_what_did_not_work.png)

{{NEGATIVES_LIST}}

Each item above states the measured result and whether the cause was **verified** (with a profiler or experiment) or is a **hypothesis**.

---

## The detector

| | |
|---|---|
| Dataset | {{DATASET_NAME}} ({{DATASET_URL}}) |
| License | {{DATASET_LICENSE}} (checked on {{LICENSE_CHECK_DATE}}) |
| Subset used | {{N_TRAIN}} train / {{N_VAL}} val images, {{N_BOXES}} boxes |
| Classes | {{CLASSES}} |
| Split | {{SPLIT_DESC}} |
| Model | YOLOv8n, fine-tuned {{EPOCHS}} epochs, imgsz {{IMGSZ}}, {{TRAIN_TIME}} on {{GPU_MODEL}} |
| Val metrics | mAP50 {{MAP50}}, mAP50-95 {{MAP5095}}, precision {{PRECISION}}, recall {{RECALL}} |
| Demo footage | {{FOOTAGE_DESCRIPTION}} |

Pretrained COCO vs fine-tuned, same pipeline and same frames: {{DET_COCO}} vs {{DET_FT}} detections per frame. The fine-tuned model finds more objects, so absolute times differ between runs. Compare the speedup within each run.

![Pretrained vs fine-tuned](docs/ppt_pack/12_pretrained_vs_finetuned.png)

**Limits of the detector:** small data subset, short fine-tune, four coarse classes, one dataset. It is good enough to give the pipeline real detections to process. It is not a claim about sorting accuracy on a real line.

---

## Profiling

- **Nsight Compute (ncu):** vector add at {{DRAM_PCT}}% DRAM throughput and {{SM_PCT}}% SM throughput. {{NCU_FUSED_SENTENCE}}
- **Nsight Systems (nsys):** {{NSYS_SENTENCE}}

---

## Limitations

- {{LIMIT_DATA}}
- {{LIMIT_FOOTAGE}}
- {{LIMIT_GPU}} Numbers on other GPUs will differ.
- {{LIMIT_TRT}}
- Postprocess time: {{POST_NOTE}}
- Timings include or exclude the following, stated per table above: host to GPU copies, NMS, decode.
- No end-to-end claim about a live camera: decode was from a file and the loop was run offline.

---

## Reproduce

Runs on a free Colab T4.

```bash
git clone https://github.com/pkbros/CUDA-accelerated-waste-detection-pipeline
cd CUDA-accelerated-waste-detection-pipeline
# dataset download link is read from an environment variable, never hardcoded
export ROBOFLOW_EXPORT_URL="<your export link>"
python demo/run_colab_demo.py
```

Or open `notebooks/run_all.ipynb` and run top to bottom. Individual stages:

```bash
make -C phase_a ARCH=sm_75 run     # vector add, reduction, matmul, streams
python phase_b/preprocess_ext.py   # fused preprocess + benchmarks
python phase_b/infer_bench.py      # inference ladder
python phase_b/pipeline.py         # serial vs pipelined, multi-feed
python tools/make_ppt_pack.py      # regenerate figures from the CSVs
python tools/build_readme.py       # regenerate this README from the CSVs
```

Every number in this README is generated from files in `results/`. Failures are logged in `results/failures.md` and are kept on purpose.

---

## Repo layout

```
phase_a/          vector_add.cu  reduction.cu  matmul.cu  streams.cu  Makefile
phase_b/          preprocess.cu  preprocess_ext.py  infer_bench.py  pipeline.py  data_prep.py  finetune.py
tools/            make_ppt_pack.py  build_readme.py  summary.py
results/          *.csv  summary.md  failures.md  device.json
docs/             PROJECT_EXPLAINED.md  kernel_notes.md  data_sources.md  ppt_pack/
notebooks/        run_all.ipynb
```

Start with `docs/PROJECT_EXPLAINED.md` for a plain-English walkthrough, and `results/summary.md` for every table.

---

## Glossary

| Term | Meaning |
|---|---|
| Kernel | A function that runs on the GPU, on many threads at once. |
| Memory-bound | Speed limited by how fast data moves, not by arithmetic. |
| Coalesced access | Neighboring threads reading neighboring addresses, so the GPU fetches memory in big efficient chunks. |
| Shared memory | Small fast memory shared by the threads of a block. |
| Warp | 32 threads that execute together. |
| Pinned memory | Host memory the GPU can copy from directly, which makes transfers faster. |
| Stream | An ordered queue of GPU work. Different streams can overlap. |
| CUDA events | GPU-side timers, used to time kernels accurately. |
| NMS | Non-maximum suppression: removes duplicate boxes for the same object. |
| mAP50 | Detection accuracy score at an IoU threshold of 0.5. |
| GFLOPS | Billions of floating-point operations per second. |

---

## Credits and licenses

Dataset: {{DATASET_NAME}}, {{DATASET_LICENSE}}. Model: Ultralytics YOLOv8n ({{ULTRALYTICS_LICENSE}}). Code in this repository: {{REPO_LICENSE}}.
