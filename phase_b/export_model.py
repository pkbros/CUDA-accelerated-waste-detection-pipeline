"""Download the pretrained detector and export it to ONNX for the ladder.

Standalone::

    python export_model.py

Writes ``<repo>/models/yolov8n.pt`` (downloaded by ultralytics if absent) and
``<repo>/models/yolov8n.onnx`` (opset 17, imgsz 640, batch 1, simplify=False),
then prints both file sizes as measured from disk. No numbers are invented: the
sizes come from ``Path.stat()`` and the export result comes from ultralytics.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# models/ lives at the repo root, one level above phase_b/.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS = MODELS_DIR / "yolov8n.pt"
ONNX = MODELS_DIR / "yolov8n.onnx"


def main() -> None:
    """Download (if needed), export, and report the two file sizes."""
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # YOLO() downloads the weights to this exact path when it is missing, so
    # the download is pinned to models/ instead of whatever the CWD happens to
    # be. Keeping it next to the ONNX file makes the ladder's paths predictable.
    model = YOLO(str(WEIGHTS))

    # simplify=False keeps the graph shape identical to what PyTorch produced;
    # an ONNX simplifier can fold nodes in ways that change Op sets and make
    # the ORT / TensorRT rungs harder to compare fairly.
    exported = model.export(
        format="onnx",
        opset=17,
        imgsz=640,
        batch=1,
        simplify=False,
    )

    # Ultralytics normally writes models/yolov8n.onnx next to the weights. If a
    # version writes somewhere else, honour the returned path but still keep a
    # canonical copy at ONNX so backends.py / infer_bench.py can find it.
    if not ONNX.exists() and exported:
        src = Path(str(exported))
        if src.exists() and src.resolve() != ONNX.resolve():
            shutil.copyfile(src, ONNX)
    if not ONNX.exists():
        raise RuntimeError(
            f"ONNX export did not produce {ONNX} (ultralytics returned {exported!r})."
        )

    for path in (WEIGHTS, ONNX):
        size = path.stat().st_size
        print(f"{path}: {size:,} bytes ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
