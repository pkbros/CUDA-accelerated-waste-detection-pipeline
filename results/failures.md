# Recorded failures

Things that did not work in this environment.
Kept deliberately: the PRD asks for honesty over a clean story.

- **fused preprocess kernel**: RuntimeError('Could not build phase_b/preprocess.cu. No fallback was used. The PRD fallback is a CuPy RawKernel; implementing it is a separate, explicitly-reported task, not a silent substitution.')
- **fused preprocess kernel**: RuntimeError('Could not build phase_b/preprocess.cu. No fallback was used. The PRD fallback is a CuPy RawKernel; implementing it is a separate, explicitly-reported task, not a silent substitution.')
- **fused preprocess kernel**: RuntimeError('Could not build phase_b/preprocess.cu. No fallback was used. The PRD fallback is a CuPy RawKernel; implementing it is a separate, explicitly-reported task, not a silent substitution.')
- 2026-11-24 — ort_cuda: Error when binding input: There's no data transfer registered for copying tensors from Device:[DeviceType:1 MemoryType:0 VendorId:4318 DeviceId:0 Alignment:0] to Device:[DeviceType:0 MemoryType:0 VendorId:0 DeviceId:0 Alignment:0]
- 2026-11-24 — trt_fp16: trt_fp16 backend unavailable (ModuleNotFoundError): Traceback (most recent call last): File "/content/cuda-waste-pipeline/phase_b/backends.py", line 521, in get_backend return _TRTBackend(model_path, imgsz, batch) # trt_fp16 File "/content/cuda-waste-pipeline/phase_b/backends.py", line 342, in __init__ import tensorrt as trt ModuleNotFoundError: No module named 'tensorrt'
- 2026-11-24 — trt_fp16: trt_fp16 backend unavailable (ModuleNotFoundError): Traceback (most recent call last): File "/content/cuda-waste-pipeline/phase_b/backends.py", line 519, in get_backend return _TRTBackend(model_path, imgsz, batch) # trt_fp16 File "/content/cuda-waste-pipeline/phase_b/backends.py", line 340, in __init__ import tensorrt as trt ModuleNotFoundError: No module named 'tensorrt'

- 2026-11-24 - Task 1 fair preprocess benchmark: CUDA timing rows (kernel_only, h2d_plus_kernel, end_to_end, torch_gpu_ops) are code-complete but NOT MEASURED. This workspace has no GPU, torch or nvcc. The CPU-side fix was validated locally with OpenCV: the old path reproduced max_abs_err=2.117904e+00 and the fixed path gives 4.768372e-07. Requires a Colab T4 run.

- 2026-11-24 - Task 2 dataset: ZeroWaste branch STOPPED. The Roboflow page now loads (9,563 images, 4 classes) but states two conflicting licences on the same page (header CC BY 4.0 vs About CC BY-NC 4.0), and the export needs ROBOFLOW_API_KEY. Not used. Owner must confirm the licence and supply a key. Fallback TACO (Zenodo 3587843, CC BY 4.0) selected; subset not yet built in this workspace (no network download performed here). See data/README.md for the exact Colab commands.

- 2026-11-24 — ZeroWaste licence conflict RESOLVED for prototype use: the Roboflow API (projects_get and versions_export via MCP) and the exported data.yaml all report license=CC BY 4.0 for the hosted v3 copy; the CC BY-NC string in the page About block refers to the original academic dataset. Owner explicitly authorised proceeding for a non-redistributed interview prototype. Re-check before publishing the images.
- 2026-11-24 — TACO fallback not used: ZeroWaste was obtained successfully through the Roboflow MCP, so the CC BY 4.0 fallback dataset is no longer needed. Its prep code and licence notes remain in the repo.
- 2026-11-24 — Fine-tune, real-frame ladder, per-frame demo timings and demo.mp4 still NOT RUN: this workspace has no GPU/torch. Code and real dataset are ready; Colab T4 needed. No mAP or detection count is claimed.
