"""Optional light fine-tune of a small YOLO on the converted waste dataset.

Runs in Colab (needs torch + ultralytics). If the dataset has not been
converted yet we do NOT pretend to train: we say so, record the fallback in
results/failures.md, and exit 0 so the notebook can continue with the
pretrained COCO model for the demo.

Honest limitation: a few epochs on a few thousand images will not beat COCO
pretraining for waste classes. This fine-tune exists to show the *pipeline*
(labels -> train -> mAP -> checkpoint), not to claim state-of-the-art waste
detection. The README states this explicitly.

Usage:
    python phase_b/finetune.py --data data/yolo/taco/data.yaml --epochs 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:  # imported as a package: python -m phase_b.finetune
    from .common import REPO_ROOT, append_row, results_dir
except ImportError:  # executed directly: python phase_b/finetune.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REPO_ROOT, append_row, results_dir

FALLBACK_MESSAGE = (
    "dataset not converted yet; run python phase_b/data_prep.py ... ; "
    "falling back to the pretrained COCO model for the demo"
)


def record_fallback(reason: str) -> Path:
    """Append a dated bullet to results/failures.md so the fallback is visible."""
    path = results_dir() / "failures.md"
    header = "# Failures and honest fallbacks\n\n"
    text = path.read_text(encoding="utf-8") if path.exists() else header
    bullet = f"- {dt.date.today().isoformat()}: fine-tune skipped - {reason}\n"
    if bullet not in text:  # re-runs on the same day should not duplicate
        if not text:
            text = header
        if not text.endswith("\n"):
            text += "\n"
        text += bullet
    path.write_text(text, encoding="utf-8")
    return path


def _yaml_scalar(text: str, key: str) -> str | None:
    """Read one top-level `key: value` from a data.yaml without PyYAML."""
    for line in text.splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def _absolute_data_yaml(data_yaml: Path) -> Path:
    """Return a copy of the dataset yaml with absolute image paths.

    Ultralytics resolves a relative `path:` against the process working
    directory in newer releases. The generated data/waste.yaml uses
    `path: .` + `waste/images/...`, which is correct only when the CWD is
    data/. Resolving train/val against the yaml's own directory here makes
    training work from any CWD, including the Colab notebook root.
    """
    base = data_yaml.parent.resolve()
    lines: list[str] = []
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("path:"):
            continue  # absolute train/val make `path` unnecessary
        handled = False
        for key in ("train", "val"):
            if stripped.startswith(key + ":"):
                value = stripped.split(":", 1)[1].strip().strip("'\"")
                path = Path(value)
                if not path.is_absolute():
                    path = base / path
                lines.append(f"{key}: {path.resolve()}")
                handled = True
                break
        if not handled:
            lines.append(line)
    target = data_yaml.with_name("waste_train_absolute.yaml")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _metric(box: Any, *names: str) -> str:
    """Read the first available metric from ultralytics' box results object."""
    if box is None:
        return "unknown"
    for name in names:
        value = getattr(box, name, None)
        if value is not None:
            try:
                return f"{float(value):.6f}"
            except (TypeError, ValueError):
                return str(value)
    metrics = getattr(box, "results_dict", None) or {}
    for name in names:
        for key in (f"metrics/{name}(B)", name):
            if key in metrics:
                return str(metrics[key])
    return "unknown"


def _n_train(results: Any, data_yaml: Path) -> str:
    """Read the training-set size from the trainer; fall back to counting files.

    Returns a string because an unavailable count must never become a made-up 0.
    """
    trainer = getattr(results, "trainer", None)
    ntrain = getattr(trainer, "ntrain", None)
    if ntrain is not None:
        return str(ntrain)

    try:
        text = data_yaml.read_text(encoding="utf-8")
        # ultralytics resolves `path: .` relative to the yaml file, NOT to the
        # current working directory -- do the same or n_train will be wrong.
        base = Path(_yaml_scalar(text, "path") or ".")
        if not base.is_absolute():
            base = data_yaml.parent / base
        base = base.resolve()
        train_rel = _yaml_scalar(text, "train")
        if not train_rel:
            return "unavailable"
        train_dir = (base / train_rel).resolve()
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return str(sum(1 for p in train_dir.rglob("*") if p.suffix.lower() in suffixes))
    except (OSError, ValueError):
        return "unavailable"


def _map50(results: Any) -> str:
    """Read mAP50 from the trainer's metrics object; never invent a value."""
    metrics = getattr(results, "results_dict", None) or {}
    value = None
    for key in ("metrics/mAP50(B)", "mAP50", "metrics/mAP50"):
        if key in metrics:
            value = metrics[key]
            break
    if value is None:
        value = getattr(getattr(results, "box", None), "map50", None)
    if value is None:
        return "unavailable"
    return f"{float(value):.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/yolo/taco/data.yaml", help="path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/finetune")
    parser.add_argument("--name", default="waste")
    args = parser.parse_args(argv)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(FALLBACK_MESSAGE)
        print(f"  (looked for {data_yaml.resolve()})")
        record_fallback(f"data.yaml not found at {data_yaml}")
        return 0

    from ultralytics import YOLO  # imported lazily so --help works without torch

    train_yaml = _absolute_data_yaml(data_yaml)
    print(f"training with absolute dataset paths from {train_yaml}")
    model = YOLO(args.model)
    # Colab's free tier has 2 vCPUs: workers > 2 is a known deadlock.
    # cache=False on the first epoch: do not copy the dataset into RAM/disk on a
    # machine with tight memory; enable caching only on later runs if needed.
    t0 = time.perf_counter()
    results = model.train(
        data=str(train_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=2,
        cache=False,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    trainer_best = getattr(getattr(results, "trainer", None), "best", None)
    if not best_pt.exists() and trainer_best:
        best_pt = Path(trainer_best)
    if not best_pt.exists():
        raise FileNotFoundError(f"training finished but no checkpoint at {best_pt}")
    print(f"Best checkpoint: {best_pt}")

    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    installed = models_dir / "finetuned.pt"
    shutil.copy2(best_pt, installed)
    print(f"Copied to: {installed}")

    train_ms = (time.perf_counter() - t0) * 1e3  # real wall clock, never typed
    # Also publish under the name the demo scripts expect.
    weights_dir = REPO_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, weights_dir / "waste_yolov8n.pt")

    map50 = _map50(results)
    box = getattr(results, "box", None)
    map5095 = _metric(box, "map50_95", "map")
    precision = _metric(box, "mp", "precision")
    recall = _metric(box, "mr", "recall")
    n_train = _n_train(results, data_yaml)
    append_row(
        bench="finetune",
        variant=args.model,
        size=args.epochs,
        size_unit="epochs",
        time_ms_mean=train_ms,
        time_ms_std=0.0,
        throughput=0.0,
        throughput_unit="",
        extra=(
            f"mAP50={map50},mAP50_95={map5095},precision={precision},"
            f"recall={recall},imgsz={args.imgsz},n_train={n_train},"
            f"weights=weights/waste_yolov8n.pt"
        ),
        filename="finetune.csv",
    )
    print(
        f"Recorded results/finetune.csv: train_ms={train_ms:.1f}, mAP50={map50}, "
        f"mAP50_95={map5095}, precision={precision}, recall={recall}, "
        f"imgsz={args.imgsz}, n_train={n_train}"
    )
    print(
        "Reminder: a few epochs on a few thousand images will not beat COCO "
        "pretraining for waste classes; this run demonstrates the pipeline only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
