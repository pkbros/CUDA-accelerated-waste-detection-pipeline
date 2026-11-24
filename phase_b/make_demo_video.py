"""Make demo/belt_input.mp4 from real ZeroWaste val frames (Task 4).

Source order follows the dataset's own frame numbering, so the result is a
real conveyor sequence rather than a random shuffle. If the required
ROBOFLOW_API_KEY is absent, this writes the failure to results/failures.md and
exits 0 -- it never fabricates a video or claims real footage it does not have.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, results_dir


def record_failure(stage: str, detail: str) -> None:
    path = results_dir() / "failures.md"
    new = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("# Recorded failures\n\n")
        fh.write(f"- 2026-11-24 - {stage}: {detail.strip()[:400]}\n")
    print(f"[recorded failure] {stage}: {detail.strip()[:200]}")


def main(argv=None) -> int:
    import cv2
    from roboflow import Roboflow  # optional dependency, imported lazily

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        record_failure(
            "Task 4 real belt video",
            "ROBOFLOW_API_KEY not set. Cannot download ZeroWaste val frames. "
            "Prateek must export a YOLOv8 dataset from "
            "https://universe.roboflow.com/zerowaste-data/zerowaste-contamination "
            "(after confirming the licence conflict) and set the key in Colab.",
        )
        return 0

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("zerowaste-data").project("zerowaste-contamination")
    dataset = project.version(1).download("yolov8", location=str(REPO_ROOT / "data" / "zerowaste"))
    val_dir = Path(dataset.location) / "valid" / "images"
    paths = sorted(
        p for p in val_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        record_failure("Task 4 real belt video", f"no val images in {val_dir}")
        return 0

    demo = REPO_ROOT / "demo"
    demo.mkdir(exist_ok=True)
    out = demo / "belt_input.mp4"
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (1280, 720))
    if not writer.isOpened():
        record_failure("Task 4 real belt video", "cv2.VideoWriter could not open H.264 output")
        return 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        writer.write(cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA))
    writer.release()
    print(f"wrote {out} from {len(paths)} real val frames in original order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
