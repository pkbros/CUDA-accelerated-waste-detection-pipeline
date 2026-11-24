"""Measure the inference ladder: latency (ms) and FPS for each available rung.

Run from anywhere, e.g.::

    python phase_b/infer_bench.py --iters 100 --warmup 20
    python phase_b/infer_bench.py --only pt_fp32,pt_fp16

Every rung is timed with CUDA events via common.time_call() on ONE forward pass
over the same normalized NCHW input. A rung that cannot be built is recorded in
results/failures.md and printed with a ``SKIPPED`` prefix -- it NEVER produces a
CSV row, because a row with a made-up number would sink the whole repo.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

try:  # imported as a package: python -m phase_b.infer_bench
    from .backends import BACKEND_NAMES, get_backend
    from .common import (
        append_stats,
        fps,
        require_cuda,
        results_dir,
        time_call,
    )
except ImportError:  # imported as a script: python infer_bench.py from phase_b/
    from backends import BACKEND_NAMES, get_backend
    from common import append_stats, fps, require_cuda, results_dir, time_call


# The ONNX export in export_model.py comes from the FP32 PyTorch model, so ORT
# runs FP32 weights. The variant name carries the precision explicitly so the
# summary table cannot be misread.
PRECISION = {
    "pt_fp32": "fp32",
    "pt_fp16": "fp16",
    "ort_cuda": "fp32",
    "trt_fp16": "fp16",
}
VARIANT = {
    "pt_fp32": "pt_fp32",
    "pt_fp16": "pt_fp16",
    "ort_cuda": "ort_cuda_fp32",
    "trt_fp16": "trt_fp16",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for the inference ladder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="models", help="directory with the exported weights / ONNX / engine")
    parser.add_argument("--imgsz", type=int, default=640, help="square input size in pixels")
    parser.add_argument("--batch", type=int, default=1, help="images per forward pass")
    parser.add_argument("--iters", type=int, default=100, help="timed iterations per rung")
    parser.add_argument("--warmup", type=int, default=20, help="untimed warmup iterations per rung")
    parser.add_argument("--csv", default="../results/infer.csv", help="CSV filename passed to common.append_stats")
    parser.add_argument("--only", default="", help="comma-separated subset of backends to run")
    parser.add_argument(
        "--frames", default=None,
        help="directory of real frames to use as input instead of a zero tensor",
    )
    return parser.parse_args(argv)


def resolve_models_dir(value: str) -> Path:
    """Resolve ``--models`` without depending on the caller's CWD.

    An existing relative path is used as-is; otherwise ``models`` is looked up
    at the repo root, so the default works both from phase_b/ and from the
    repo root.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (Path(__file__).resolve().parent.parent / path).resolve()


def model_path_for(name: str, models_dir: Path) -> str:
    """Canonical weight/engine filename for a rung inside --models."""
    filename = {
        "pt_fp32": "yolov8n.pt",
        "pt_fp16": "yolov8n.pt",
        "ort_cuda": "yolov8n.onnx",
        "trt_fp16": "yolov8n.engine",
    }[name]
    # Prefer the fine-tuned model when it exists, so no rung silently times a
    # different network than the one used in the per-frame pipeline.
    tuned = Path(filename).with_name("waste_" + filename)
    chosen = tuned if (models_dir / tuned).exists() else Path(filename)
    return str(models_dir / chosen)


def record_failure(name: str, reason: str) -> Path:
    """Append one dated bullet to results/failures.md (first 400 chars only)."""
    path = results_dir() / "failures.md"
    snippet = " ".join(str(reason).split())[:400]
    date = _dt.date.today().isoformat()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"- {date} — {name}: {snippet}\n")
    return path


def print_summary(results: dict[str, dict[str, float]]) -> None:
    """Compact measured-only table; speedup uses pt_fp32's measured mean."""
    print("\nInference ladder summary (measured means only)")
    if not results:
        print("No backend ran; nothing to summarize.")
        return
    header = (
        f"{'backend':<14} {'mean ms':>10} {'std ms':>10} {'FPS':>10} {'speedup':>10}"
    )
    print(header)
    print("-" * len(header))
    base = results.get("pt_fp32")
    for variant, stats in results.items():
        if base is not None and stats["mean"] > 0:
            speedup = f"{base['mean'] / stats['mean']:.2f}x"
        else:
            speedup = "-"
        print(
            f"{variant:<14} {stats['mean']:>10.3f} {stats['std']:>10.3f} "
            f"{fps(stats['mean']):>10.1f} {speedup:>10}"
        )
    if base is None:
        print("(pt_fp32 did not run; speedup column omitted for all rows)")
    print("speedup = pt_fp32 mean / this mean; FPS = 1000 / mean (per forward).")


def main(argv: list[str] | None = None) -> None:
    """Guard CUDA, then time every requested rung and write CSV rows."""
    args = parse_args(argv)
    # Fail loudly on a CPU-only runtime: timing there would be a different
    # benchmark entirely and silently poisoning the results table.
    require_cuda()

    requested = [s.strip().lower() for s in args.only.split(",") if s.strip()]
    if not requested:
        requested = list(BACKEND_NAMES)
    unknown = [n for n in requested if n not in BACKEND_NAMES]
    if unknown:
        raise SystemExit(f"Unknown backend(s): {unknown}; choose from {BACKEND_NAMES}")

    import torch

    models_dir = resolve_models_dir(args.models)
    # Same input for every rung: float32, NCHW RGB, already normalized to 0..1.
    # Zeros are a valid normalized input; using a fixed tensor keeps the
    # comparison deterministic (the FP16 rung casts it internally).
    if args.frames:
        import cv2
        import numpy as np

        frame_dir = Path(args.frames)
        paths = sorted(
            p for p in frame_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )[: args.batch]
        if not paths:
            raise SystemExit(f"no frames found in {frame_dir}")
        imgs = []
        for p in paths:
            f = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if f is None:
                continue
            f = cv2.resize(f, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            imgs.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        while len(imgs) < args.batch:
            imgs.append(imgs[-1].copy())
        x = torch.from_numpy(np.stack(imgs)).to("cuda")
        input_note = "real_frames"
    else:
        x = torch.zeros(
            args.batch, 3, args.imgsz, args.imgsz, device="cuda", dtype=torch.float32
        )
        input_note = "zero_tensor"

    results: dict[str, dict[str, float]] = {}
    for name in requested:
        try:
            backend = get_backend(
                name,
                model_path=model_path_for(name, models_dir),
                imgsz=args.imgsz,
                batch=args.batch,
            )
            backend.warmup(args.warmup)
            # time_call does its own synchronizes and CUDA-event timing; we
            # already warmed up above, so pass warmup=0 to avoid doing it twice.
            stats = time_call(lambda b=backend: b.infer(x), warmup=0, iters=args.iters)

            variant = VARIANT[name]
            extra = (
                f"precision={PRECISION[name]}; {backend.decode_note}; "
                f"{backend.sync_note}; output={backend.output_shape}; "
                f"input=NCHW float32 {args.imgsz}px batch={args.batch} "
                f"source={input_note}; "
                "fps_basis=per-forward"
            )
            append_stats(
                "infer",
                variant,
                args.batch,
                "images",
                stats,
                fps(stats["mean"]),
                "FPS",
                extra=extra,
                filename=args.csv,
            )
            results[variant] = stats
            print(
                f"OK {variant}: {stats['mean']:.3f} +- {stats['std']:.3f} ms, "
                f"{fps(stats['mean']):.1f} FPS (n={stats['n']})"
            )
        except RuntimeError as exc:
            reason = str(exc)
            record_failure(name, reason)
            print(f"SKIPPED {name}: {' '.join(reason.split())[:200]}")

    print_summary(results)


if __name__ == "__main__":
    main()
