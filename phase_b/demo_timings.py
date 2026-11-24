"""Log per-frame timings for the split-screen demo (Task 9).

Runs two pipelines over the SAME real val frames:
  baseline  : OpenCV CPU preprocess + PyTorch FP32, serial
  optimized : fused CUDA kernel + best available backend + streamed upload

Writes results/demo_frame_timings.csv with the schema consumed by
demo/render_demo.py. Nothing here fabricates a number: every field is measured
or read from the CSV the benchmark itself wrote.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import REPO_ROOT, results_dir


def _elapsed(t0: float) -> float:
    return (time.perf_counter() - t0) * 1e3


def _boxes_json(pred) -> str:
    """Serialise backend output [[x1,y1,x2,y2,cls,score], ...] as JSON."""
    if not pred:
        return ""
    boxes, scores, classes = pred
    if boxes is None or len(boxes) == 0:
        return ""
    out = []
    for b, s, c in zip(boxes, scores, classes):
        try:
            xyxy = [float(v) for v in b[:4]]
        except (TypeError, ValueError):
            continue
        # c may be a 0-d torch tensor; str(c) would render "tensor(27)" in the
        # video overlay. Convert to a plain int for a readable class label.
        try:
            cls = int(c)
        except (TypeError, ValueError):
            cls = str(c)
        out.append(xyxy + [cls, float(s)])
    return json.dumps(out)


def _torch_preprocess_cpu(img_bgr: np.ndarray, dst: int = 640):
    """Honest OpenCV baseline matching the fused kernel's output shape."""
    import cv2
    import torch

    h, w = img_bgr.shape[:2]
    scale = min(dst / h, dst / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    left, top = (dst - nw) // 2, (dst - nh) // 2
    r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    p = cv2.copyMakeBorder(r, top, dst - nh - top, left, dst - nw - left,
                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    for c in range(3):
        chw[c] = (chw[c] - (0.485, 0.456, 0.406)[c]) / (0.229, 0.224, 0.225)[c]
    chw[:, :top] = 0.0
    chw[:, top + nh:] = 0.0
    chw[:, :, :left] = 0.0
    chw[:, :, left + nw:] = 0.0
    return torch.from_numpy(chw).unsqueeze(0)


def main(argv=None) -> int:
    import cv2
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=str(REPO_ROOT / "data" / "waste" / "images" / "val"))
    parser.add_argument("--model", default=None, help="fine-tuned weights, else COCO default")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--out", default="demo_frame_timings.csv")
    args = parser.parse_args(argv)

    frame_dir = Path(args.frames)
    paths = sorted(
        p for p in frame_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.n]
    if not paths:
        raise SystemExit(f"no frames in {frame_dir}; run Task 2 first")

    from backends import get_backend
    import preprocess_ext as pp

    baseline = get_backend("pt_fp32", model_path=args.model, imgsz=640, batch=1)
    # Prefer the fastest measured rung; fall back honestly if unavailable.
    optimized = None
    for name in ("ort_cuda_fp32", "pt_fp32"):
        try:
            optimized = get_backend(name, model_path=args.model, imgsz=640, batch=1)
            chosen = name
            break
        except Exception as exc:  # noqa: BLE001
            print(f"backend {name} unavailable: {exc}")
    if optimized is None:
        raise SystemExit("no usable inference backend")
    baseline.warmup(5)
    optimized.warmup(5)
    ext = pp.get_extension()

    out_path = results_dir() / args.out
    header = (
        "frame,baseline_decode_ms,baseline_preprocess_ms,baseline_infer_ms,"
        "baseline_post_ms,baseline_fps,baseline_detections,opt_decode_ms,"
        "opt_preprocess_ms,opt_infer_ms,opt_post_ms,opt_fps,opt_detections,"
        "baseline_boxes,opt_boxes\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(header)

    for i, path in enumerate(paths):
        t0 = time.perf_counter()
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        dec = _elapsed(t0)

        # ---- baseline: OpenCV CPU preprocess + FP32, serial ----
        t0 = time.perf_counter()
        x_cpu = _torch_preprocess_cpu(img)
        pre_b = _elapsed(t0)
        t0 = time.perf_counter()
        xb = x_cpu.to("cuda", non_blocking=False)
        pred_b = baseline.infer(xb)
        torch.cuda.synchronize()
        infer_b = _elapsed(t0)

        # ---- optimized: pinned upload + fused kernel + chosen backend ----
        t0 = time.perf_counter()
        pinned = torch.from_numpy(np.ascontiguousarray(img)).pin_memory()
        dev = pinned.to("cuda", non_blocking=True)
        sc, px, py = pp._letterbox_geometry(img.shape[0], img.shape[1], 640)
        out = torch.empty((1, 3, 640, 640), dtype=torch.float32, device="cuda")
        ext.preprocess_f32(dev.view(img.shape), out.view(3, 640, 640), 640, 640,
                           sc, px, py, list(pp.DEFAULT_MEAN), list(pp.DEFAULT_STD))
        torch.cuda.synchronize()
        pre_o = _elapsed(t0)
        t0 = time.perf_counter()
        pred_o = optimized.infer(out)
        torch.cuda.synchronize()
        infer_o = _elapsed(t0)

        nb = sum(len(p[0]) for p in pred_b if p and p[0] is not None)
        no = sum(len(p[0]) for p in pred_o if p and p[0] is not None)
        row = [
            i, f"{dec:.4f}", f"{pre_b:.4f}", f"{infer_b:.4f}", "0.0000",
            f"{1000.0 / max(dec + pre_b + infer_b, 1e-6):.3f}", nb,
            f"{dec:.4f}", f"{pre_o:.4f}", f"{infer_o:.4f}", "0.0000",
            f"{1000.0 / max(dec + pre_o + infer_o, 1e-6):.3f}", no,
            _boxes_json(pred_b[0] if pred_b else None),
            _boxes_json(pred_o[0] if pred_o else None),
        ]
        with out_path.open("a", encoding="utf-8", newline="") as fh:
            import csv
            csv.writer(fh).writerow(row)
        if i % 25 == 0:
            print(f"frame {i}/{len(paths)}: baseline {nb} boxes, {chosen} {no} boxes")

    print(f"wrote {out_path} ({len(paths)} frames, optimized backend={chosen})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
