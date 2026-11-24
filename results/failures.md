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
