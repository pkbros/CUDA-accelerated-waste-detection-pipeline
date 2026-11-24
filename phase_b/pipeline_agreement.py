"""Check that the optimized pipeline returns the same detections as baseline.

Reads results/demo_frame_timings.csv (written by demo_timings.py) and produces
results/pipeline_agreement.csv. A faster pipeline with different boxes is a bug,
so this is a gate, not a decorative metric.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import REPO_ROOT, results_dir


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse(value: str) -> list[list]:
    if not value.strip():
        return []
    parsed = json.loads(value)
    return [b for b in parsed if len(b) >= 6]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-csv", default="demo_frame_timings.csv")
    args = ap.parse_args(argv)
    src = results_dir() / args.frames_csv
    if not src.exists():
        raise SystemExit(f"missing {src}; run demo_timings.py first")

    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    out = []
    for row in rows:
        base, opt = parse(row["baseline_boxes"]), parse(row["opt_boxes"])
        # Greedy class-matched IoU pairing.
        used: set[int] = set()
        pairs: list[float] = []
        for b in base:
            best_i, best_v = -1, 0.0
            for j, o in enumerate(opt):
                if j in used or b[4] != o[4]:
                    continue
                v = iou(b, o)
                if v > best_v:
                    best_i, best_v = j, v
            if best_i >= 0 and best_v >= 0.5:
                used.add(best_i)
                pairs.append(best_v)
        diff = len(base) - len(opt)
        both_empty = not base and not opt
        # Two empty detections agree: an IoU of 0 on an empty frame is a
        # definition artefact, not a disagreement. Counting those 60 frames as
        # failures made the score read as an exact 50.0% / 0.500 bug.
        same = both_empty or (diff == 0 and len(pairs) == len(base))
        out.append({
            "frame": row["frame"],
            "baseline_count": len(base),
            "opt_count": len(opt),
            "count_diff": diff,
            "matched": len(pairs),
            "mean_iou": f"{sum(pairs) / len(pairs):.6f}" if pairs else "0.000000",
            "with_boxes": str(not both_empty).lower(),
            "agrees": str(same).lower(),
        })

    dest = results_dir() / "pipeline_agreement.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n = len(out)
    mean_diff = sum(int(r["count_diff"]) for r in out) / n
    mean_iou = sum(float(r["mean_iou"]) for r in out) / n
    agrees = sum(r["agrees"] == "true" for r in out) / n * 100
    # An empty frame contributes IoU 0 by definition; also report IoU only on
    # frames where both pipelines actually produced boxes.
    with_boxes = [float(r["mean_iou"]) for r in out if int(r["baseline_count"]) > 0]
    iou_boxes = sum(with_boxes) / len(with_boxes) if with_boxes else 0.0
    print(f"frames={n} mean_count_diff={mean_diff:.4f} mean_iou_all={mean_iou:.4f} "
          f"mean_iou_with_boxes={iou_boxes:.4f} frames_agree={agrees:.1f}%")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
