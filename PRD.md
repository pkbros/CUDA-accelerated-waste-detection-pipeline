PRD: GPU-Accelerated Conveyor Waste Detection Pipeline

Owner: Prateek | Target: Colab T4 | Audience: CUDA Engineer (Fresher) application, One Step Greener

1. Goal

Show CUDA engineering skill on a recycling-relevant workload: build a real-time conveyor-belt waste detection pipeline, and measure and explain each GPU optimization with real numbers.

2. Non-goals
Not a state-of-the-art detector. A pretrained YOLO with light fine-tuning is enough.
No robotics, UI, or deployment work.
3. Success criteria
Every claim in the README is backed by a benchmark the agent actually ran. No invented numbers.
The author can explain every kernel and every speedup without help.
Repo runs end-to-end from a single Colab notebook.
4. Scope

Phase A, fundamentals (must finish tonight)

Vector add: sanity check and nvcc setup.
Parallel reduction in 3 versions: naive (global memory), shared memory, warp shuffle. Time with CUDA events.
Matrix multiply in 3 versions: naive, tiled shared memory, and cuBLAS as reference. Report GFLOPS and a CPU baseline.
Streams demo: pinned memory, 2 streams overlapping H2D copy and kernel, compared with synchronous execution.

Phase B, applied pipeline
5. Fused preprocess kernel: letterbox resize (bilinear), BGR to RGB, normalize, HWC to CHW, uint8 to FP16/FP32, in a single kernel. Benchmark against OpenCV and NumPy on CPU.
6. Inference ladder on a YOLO model: PyTorch FP32, then PyTorch FP16, then ONNX Runtime CUDA, then TensorRT FP16 (stretch). Report latency (ms) and FPS for each.
7. Pipelined end-to-end loop: capture, preprocess, infer, and postprocess across 2-3 CUDA streams, with pinned buffers. Compare serial against pipelined throughput.
8. Batching: simulate 1, 2, 4 camera feeds, and report FPS per feed.
9. Profiling: Nsight Systems timeline and Nsight Compute on one kernel if the tools work on Colab. If not, document that and rely on event timings.

5. Data
Primary: a public conveyor-belt waste dataset (for example ZeroWaste, or TACO/TrashNet as fallback). Agent must verify the license and download source.
Demo input: a looping video of belt footage.
Light fine-tune of a small YOLO (nano/small) for a few epochs. If training is too slow, use the pretrained model and say so.
6. Repo structure
cuda-waste-pipeline/
  README.md
  notebooks/run_all.ipynb
  phase_a/        vector_add.cu  reduction.cu  matmul.cu  streams.cu  Makefile
  phase_b/        preprocess.cu  preprocess_ext.py  infer_bench.py  pipeline.py
  results/        *.csv  plots/
  docs/           kernel_notes.md
7. Technical requirements
CUDA C++ built with nvcc; Python with PyTorch, OpenCV, ONNX Runtime.
Preprocess kernel exposed to Python via torch.utils.cpp_extension.load (fallback: CuPy RawKernel).
Timing with cudaEvent for GPU work and wall clock for end-to-end. Warm up and average over N runs. Report mean and std.
Verify correctness: the GPU preprocess output must match the CPU reference within a stated tolerance.
Detection correctness is checked by visual sample outputs plus mAP or counts if fine-tuned.
8. Deliverables
The repo above.
results/summary.md: one table of all benchmarks (version, time, speedup vs baseline, GPU model).
docs/kernel_notes.md: for each kernel, a short explanation of why it is faster or slower (coalescing, bank conflicts, occupancy, divergence, memory-bound vs compute-bound).
README with problem statement, architecture diagram, results table, how to reproduce, and limitations.
9. Agent rules
Comment every kernel line-by-line enough for a learner. Keep kernels readable over clever.
After each milestone, print a "what I just built and why" summary.
Never fabricate or estimate benchmark results. If something fails (TensorRT install, Nsight), record the failure honestly and use the fallback.
No claims of "deployed" or "production" in the README. Use "prototype".
10. Milestones (1 to 3 hours of agent time, plus your review)
#	Milestone	Done when
1	Colab env + vector add	nvcc runs, output verified
2	Reduction + matmul + streams	results CSV with 3 versions each
3	Preprocess kernel	matches CPU reference, speedup measured
4	Inference ladder	table of ms/FPS per backend
5	Pipelined loop + batching	serial vs pipelined FPS
6	Profiling + docs + README	all deliverables present