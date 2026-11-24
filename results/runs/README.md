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

Note on metadata: the downloaded fine-tuned bundle still carries `finetuned=false`
in some CSV `extra` fields because the flag was set only in the runner process while
the benchmarks run as separate subprocesses. The code now propagates it through the
`FINETUNED` environment variable. The authoritative evidence for the fine-tuned run
is `finetuned/finetune.csv` (real mAP) and class IDs 0–3 in
`finetuned/demo_frame_timings.csv`.
