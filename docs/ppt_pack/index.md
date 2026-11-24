# PowerPoint image pack

Every PNG is 1920x1080. Each row links to the exact numbers used, saved as
`data/<image_stem>.csv`. No number is estimated; panels without a measurement
are labelled `NOT RUN`. The GPU in every footer comes from `results/device.json`.

| Image | What it shows | Sources |
|---|---|---|
| [`01_pipeline_stages.png`](01_pipeline_stages.png) | Shows the four pipeline stages with measured baseline and optimized milliseconds under each. | `demo_frame_timings.csv`, `device.json` |
| [`02_side_by_side_A.png`](02_side_by_side_A.png) / 02_side_by_side_B.png / 02_side_by_side_C.png | Shows the same real footage frame with baseline and optimized boxes plus that frame's four stage times. | `demo_frame_timings.csv`, `device.json` |
| [`03_latency_breakdown.png`](03_latency_breakdown.png) | Stacks the four measured stage times into one baseline bar and one optimized bar with totals. | `demo_frame_timings.csv`, `device.json` |
| [`04_end_to_end_fps.png`](04_end_to_end_fps.png) | Compares rolling and run-level end-to-end FPS, the measured speedup ratio, and detection agreement on frames with boxes. | `demo_frame_timings.csv`, `device.json`, `pipeline_agreement.csv` |
| [`05_inference_ladder.png`](05_inference_ladder.png) | Compares measured inference backends at the batch sizes that were actually run; bars are ms per forward of a batch. | `infer.csv`, `device.json` |
| [`06_preprocess_honest.png`](06_preprocess_honest.png) | Compares preprocessing variants while naming the exact timing scope behind every bar. | `preprocess.csv`, `device.json` |
| [`07_kernels_reduction_matmul.png`](07_kernels_reduction_matmul.png) | Compares reduction kernel times and matmul throughput at the largest measured size. | `reduction.csv`, `matmul.csv`, `device.json` |
| [`08_memory_bound_and_streams.png`](08_memory_bound_and_streams.png) | Shows Nsight Compute DRAM versus SM utilisation and measured single-to-multi-stream transfer times. | `profiling_ncu.csv`, `streams.csv`, `device.json` |
| [`09_multi_feed.png`](09_multi_feed.png) | Compares serial and pipelined FPS per feed at one, two and four feeds; pipelining regressed in this prototype. | `batching.csv`, `device.json` |
| [`10_what_did_not_work.png`](10_what_did_not_work.png) | Collects the measured regressions: float4 loads, FP16 (batch-dependent) and multi-feed pipelining. | `vector_add.csv`, `infer.csv`, `batching.csv`, `device.json` |
| [`11_detection_quality.png`](11_detection_quality.png) | Shows six fixed-seed validation frames with thin outlined val ground-truth boxes and the measured mAP banner. | `dataset.csv`, `waste.yaml`, `finetune.csv`, `device.json` |
| [`12_pretrained_vs_finetuned.png`](12_pretrained_vs_finetuned.png) | Compares the preserved pretrained-COCO run against the fine-tuned run on the same hardware and frames. | `runs/pretrained_coco/demo_frame_timings.csv`, `runs/finetuned/demo_frame_timings.csv`, `runs/finetuned/finetune.csv`, `device.json` |

## Generation notes

- Side-by-side frames: CSV row 11 (video frame 11, id 11), CSV row 74 (video frame 74, id 74), CSV row 106 (video frame 106, id 106). CSV rows are mapped to video frames **by row position**, not by the CSV `frame` id.
- Frame rule: '>=3 baseline detections and brightness>40' was met.
- Inference ladder: batch 4 and batch 8 are absent from `results/infer.csv` and are shown as NOT RUN.
- Detection quality: six val images chosen with `random.Random(0).sample(...)` (seed 0) from val frames bright enough to see; boxes are val ground truth from `data/waste/labels/val`.
- Fine-tune mAP: read from finetune.csv (mAP50=0.4173).
- Pipeline agreement: frames with boxes agree 100.0% (60/60 frames), computed from pipeline_agreement.csv.

## Status summary

| Image | Status |
|---|---|
| `01_pipeline_stages.png` | data-backed |
| `02_side_by_side_A.png` | data-backed |
| `02_side_by_side_B.png` | data-backed |
| `02_side_by_side_C.png` | data-backed |
| `03_latency_breakdown.png` | data-backed |
| `04_end_to_end_fps.png` | data-backed |
| `05_inference_ladder.png` | data-backed |
| `06_preprocess_honest.png` | data-backed |
| `07_kernels_reduction_matmul.png` | data-backed |
| `08_memory_bound_and_streams.png` | data-backed |
| `09_multi_feed.png` | data-backed |
| `10_what_did_not_work.png` | data-backed |
| `11_detection_quality.png` | data-backed |
| `12_pretrained_vs_finetuned.png` | data-backed |
