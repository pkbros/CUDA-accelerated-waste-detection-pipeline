# Preserved measurement runs

Two complete Colab T4 runs are kept here verbatim so the pretrained and fine-tuned
results can be compared. Nothing in these folders is an estimate or a re-creation:
each file is a copy of the CSV written by that run and downloaded in its
`results_bundle.zip`.

| Folder | Detector | What it proves |
|---|---|---|
| `pretrained_coco/` | Pretrained COCO YOLOv8n | The original CUDA-only performance study; no mAP measured, COCO class names on waste frames. |
| `finetuned/` | YOLOv8n fine-tuned 30 epochs on ZeroWaste | Same four CUDA stages on the same frames, plus measured mAP50 0.4173 and waste-class boxes. |

The active `results/*.csv` files are the `finetuned/` run. `docs/ppt_pack/` is generated
from those active CSVs; image 12 reads both folders directly via
`tools/make_ppt_pack.py`.

The `pretrained_coco/` run uses COCO class labels and has no measured mAP, while the
`finetuned/` run uses the four waste classes and reports mAP50 0.4173 in `finetune.csv`.
Absolute times are not comparable across the two runs — the fine-tuned model detects about
3× more objects per frame (3.067 vs 1.125 mean detections in the demo timings) — so compare
only the optimized-vs-baseline speedup *within* each run.

Note on metadata: the downloaded fine-tuned bundle still carries `finetuned=false`
in some CSV `extra` fields because the flag was set only in the runner process while
the benchmarks run as separate subprocesses. The code now propagates it through the
`FINETUNED` environment variable. The authoritative evidence for the fine-tuned run
is `finetuned/finetune.csv` (real mAP) and class IDs 0–3 in
`finetuned/demo_frame_timings.csv`.
