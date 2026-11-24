"""Build the demo subset and belt video from a ZeroWaste v3 YOLOv8 export zip.

The zip is gitignored. Get its short-lived URL from the Roboflow MCP
(`versions_export`, project zerowaste-data/zerowaste-contamination, version 3,
format yolov8) or download it from Roboflow Universe, then place it at
data/raw/zerowaste_yolov8.zip.

Everything written here comes from the archive: class names, counts, box labels
and frame ordering. Nothing is typed by hand.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from common import REPO_ROOT

ZIP = REPO_ROOT / "data/raw/zerowaste_yolov8.zip"
OUT = REPO_ROOT / "data/waste"


def frame_no(name: str) -> int:
    m = re.search(r"_frame_(\d+)_", Path(name).name)
    return int(m.group(1)) if m else 0


def scene(name: str) -> str:
    m = re.match(r"(\d+)_frame_", Path(name).name)
    return m.group(1) if m else "unknown"


def main() -> int:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    if not ZIP.exists():
        raise SystemExit(f"missing {ZIP}; see docstring for how to obtain it")

    z = zipfile.ZipFile(ZIP)
    meta = z.read("data.yaml").decode("utf-8")
    names_match = re.search(r"^names:\s*(.+)$", meta, re.MULTILINE)
    if not names_match:
        raise SystemExit("export data.yaml has no names list")
    classes = ast.literal_eval(names_match.group(1).strip())

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)

    tr = sorted(n for n in z.namelist()
                if n.startswith("train/images/") and n.lower().endswith((".jpg", ".png")))
    va = sorted(n for n in z.namelist()
                if n.startswith("valid/images/") and n.lower().endswith((".jpg", ".png")))

    # Keep every scene that appears in val out of train entirely.
    va_scenes = {scene(n) for n in va}
    tr_by_scene: dict[str, list[str]] = defaultdict(list)
    for n in tr:
        if scene(n) not in va_scenes:
            tr_by_scene[scene(n)].append(n)

    va_scene_list = sorted(va_scenes)
    val_pick: list[str] = []
    for s in va_scene_list:
        val_pick.extend(sorted((n for n in va if scene(n) == s), key=frame_no)[:75])
    val_pick = sorted(set(val_pick[:300]), key=frame_no)

    tr_keys = sorted(tr_by_scene)
    tr_pick: list[str] = []
    for s in tr_keys:
        tr_pick.extend(sorted(tr_by_scene[s], key=frame_no)[:250])
    tr_pick = sorted(set(tr_pick[:1500]), key=frame_no)

    counts: Counter[int] = Counter()
    for split, picks in (("train", tr_pick), ("val", val_pick)):
        for img in picks:
            stem = Path(img).stem
            raw = z.read(img)
            (OUT / "images" / split / Path(img).name).write_bytes(raw)
            lbl = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if lbl in z.namelist():
                txt = z.read(lbl).decode("utf-8")
                (OUT / "labels" / split / (stem + ".txt")).write_text(txt, encoding="utf-8")
                for line in txt.splitlines():
                    if line.strip():
                        counts[int(line.split()[0])] += 1

    # Absolute paths: newer Ultralytics resolves a relative `path:` against the
    # process CWD, so `waste/images/val` used to become <repo>/waste/... and
    # training failed. OUT is data/waste, so these are the real image folders.
    data_root = OUT.resolve()
    yaml = (
        f"path: {data_root}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(classes)}\nnames:\n"
        + "\n".join(f"  {i}: '{c}'" for i, c in enumerate(classes)) + "\n"
    )
    (REPO_ROOT / "data/waste.yaml").write_text(yaml, encoding="utf-8")
    (OUT / "split.json").write_text(json.dumps({
        "source": "ZeroWaste v3 YOLOv8 export via Roboflow MCP",
        "strategy": "scene-disjoint against val, ordered by frame number",
        "n_train": len(tr_pick), "n_val": len(val_pick),
    }, indent=2), encoding="utf-8")

    per_class = ",".join(f"{classes[i]}={counts[i]}" for i in range(len(classes)))
    csv_path = REPO_ROOT / "results/dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bench", "variant", "size", "size_unit", "time_ms_mean",
                    "time_ms_std", "throughput", "throughput_unit", "extra",
                    "device", "arch"])
        w.writerow(["dataset", "zerowaste_v3_yolov8", len(tr_pick) + len(val_pick),
                    "images", "0", "0", "0", "",
                    f"train={len(tr_pick)},val={len(val_pick)},"
                    f"boxes={sum(counts.values())},{per_class},"
                    "source=roboflow_mcp,license=CC_BY_4.0,prototype_use_only",
                    "ZeroWaste Data", "n/a"])

    # Ten labelled val frames.
    canvas = Image.new("RGB", (1280, 480), "white")
    d = ImageDraw.Draw(canvas)
    for i, img in enumerate(val_pick[:10]):
        im = Image.open(OUT / "images" / "val" / Path(img).name).convert("RGB")
        im.thumbnail((240, 220))
        x, y = (i % 5) * 256, (i // 5) * 240
        canvas.paste(im, (x, y))
        lp = OUT / "labels" / "val" / (Path(img).stem + ".txt")
        nbox = 0
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                q = line.split()
                if len(q) < 5:
                    continue
                c = int(q[0]); cx, cy, bw, bh = map(float, q[1:5])
                w_, h_ = im.size
                x1, y1 = (cx - bw / 2) * w_, (cy - bh / 2) * h_
                x2, y2 = (cx + bw / 2) * w_, (cy + bh / 2) * h_
                d.rectangle([x + x1, y + y1, x + x2, y + y2], outline="red", width=2)
                if c < len(classes):
                    d.text((x + x1, y + y1), classes[c], fill="blue")
                nbox += 1
        d.text((x, y + im.height), f"{Path(img).stem[:14]} boxes={nbox}", fill="black")
    plot_dir = REPO_ROOT / "docs/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(plot_dir / "dataset_samples.png")

    # Real belt clip from val stills in original frame order.
    # About half the source images in this export are almost black (mean < 10);
    # this is a property of the source files, not of cv2.VideoWriter. Drop them
    # so the clip does not look broken, and record the exact count.
    demo_dir = REPO_ROOT / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(demo_dir / "belt_input.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 640))
    made = dropped = 0
    for n in sorted(va, key=frame_no)[:900]:
        f = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        if f is None:
            dropped += 1
            continue
        if float(f.mean()) < 10.0:  # verified against the source images
            dropped += 1
            continue
        writer.write(cv2.resize(f, (640, 640)))
        made += 1
    writer.release()
    (OUT / "video_source.json").write_text(json.dumps({
        "frames": made, "dropped_dark": dropped, "fps": 20,
        "seconds": round(made / 20, 2),
        "order": "original frame number; dark source frames removed",
        "type": "stitched still frames from the val split, not consecutive video",
        "split": "valid (never trained on)", "file": "demo/belt_input.mp4",
    }, indent=2), encoding="utf-8")

    print(f"classes: {classes}")
    print(f"subset: {len(tr_pick)} train / {len(val_pick)} val; boxes: {per_class}")
    print(f"video: {made} frames, {dropped} dark frames dropped, "
          f"{made / 20:.1f}s at 20 fps from the valid split")
    print(f"wrote {REPO_ROOT / 'data/waste.yaml'}, {plot_dir / 'dataset_samples.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
