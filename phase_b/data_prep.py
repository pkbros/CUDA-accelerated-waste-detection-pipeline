"""Dataset registry, honest download helpers and COCO -> YOLO conversion.

This module is CPU-only and runs on the machine that prepares data (in Colab).
Nothing is downloaded at import time. Licences below are recorded with the URL
they were verified from; anything not verified is labelled "unverified" on
purpose -- this repo is going to be published, so a guessed licence is worse
than a missing dataset.

Usage (from the repo root):

    python phase_b/data_prep.py list
    python phase_b/data_prep.py download --name taco --dry-run
    python phase_b/data_prep.py download --name taco
    python phase_b/data_prep.py convert --json data/raw/taco/data/annotations.json \
        --images data/raw/taco/data --out data/yolo/taco \
        --class-map docs/taco_class_map.json
    python phase_b/data_prep.py split --out data/yolo/taco
    python phase_b/data_prep.py stats --out data/yolo/taco

The dashed spellings (``--list``, ``--download``, ``--convert``) are accepted
too, because the task description uses those; argparse's convention is bare
subcommand names and we support both.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # imported as a package: python -m phase_b.data_prep
    from .common import append_row
except ImportError:  # executed directly: python phase_b/data_prep.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import append_row

# ---------------------------------------------------------------------------
# Verified source facts (checked 2026-11-24). These are source-reported values,
# used only for --dry-run and progress output -- never as benchmark results.
# ---------------------------------------------------------------------------

# Zenodo record 10.5281/zenodo.3587843 lists TACO.zip as 2.7 GB. A HEAD request
# returned 2_719_810_615 bytes, which agrees with that.
TACO_ZENODO_ZIP = "https://zenodo.org/records/3587843/files/TACO.zip?download=1"
TACO_ZIP_BYTES = 2_719_810_615

# GitHub API reports data/dataset-resized.zip in garythung/trashnet as
# 42_834_870 bytes; the README reports 2_527 images in six classes.
TRASHNET_ZIP = (
    "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
)
TRASHNET_ZIP_BYTES = 42_834_870

# Roboflow Universe: universe.roboflow.com/zerowaste-data/zerowaste-contamination
ZEROWASTE_WORKSPACE = "zerowaste-data"
ZEROWASTE_PROJECT = "zerowaste-contamination"
# Confirm the dataset version on that page before a real run; this is a
# configuration default, not a claim about the dataset's contents.
ZEROWASTE_VERSION = 1

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    """One candidate dataset and the provenance of its licence claim."""

    name: str
    url: str
    license: str
    license_url: str
    size_hint: str
    label_format: str
    needs: str
    verified: bool
    verified_from: str
    notes: str


REGISTRY: dict[str, DatasetSpec] = {
    "zerowaste": DatasetSpec(
        name="zerowaste",
        url="https://universe.roboflow.com/zerowaste-data/zerowaste-contamination",
        license="unverified - check before publishing",
        license_url="https://universe.roboflow.com/zerowaste-data/zerowaste-contamination",
        size_hint="unknown - the Roboflow page returned HTTP 403 from this environment",
        label_format="Roboflow export (request yolov8: images + one .txt per image)",
        needs="ROBOFLOW_API_KEY (Roboflow pip package). No key here -> skip, never guess.",
        verified=False,
        verified_from="",
        notes=(
            "The ORIGINAL academic ZeroWaste dataset is CC BY-NC 4.0 (NonCommercial), "
            "verified from https://github.com/dbash/zerowaste README. That does NOT "
            "automatically cover the Roboflow-hosted copy we would actually download "
            "(page 403 here), so the downloadable copy stays unverified."
        ),
    ),
    "taco": DatasetSpec(
        name="taco",
        url="https://github.com/pedropro/TACO",
        license=(
            "CC BY 4.0 for the annotations and as the DEFAULT for images with no "
            "per-image licence; individual images may carry other public licences"
        ),
        license_url="http://tacodataset.org/",
        size_hint="2.7 GB (TACO.zip, reported by Zenodo record 3587843)",
        label_format="COCO JSON (polygons + bboxes), one file for the whole dataset",
        needs="Nothing beyond disk space; archive is downloaded over HTTPS.",
        verified=True,
        verified_from="http://tacodataset.org/",
        notes=(
            "Verified string: tacodataset.org 'Terms and Conditions' states the "
            "annotations are licenced under CC BY 4.0 and images may be under different "
            "public licences, defaulting to CC BY 4.0 when the entry is missing. "
            "Zenodo DOI 10.5281/zenodo.3587843 independently lists the licence as "
            "'Creative Commons Attribution 4.0 International'. The GitHub repo's MIT "
            "licence applies to the toolkit CODE, not to the dataset images."
        ),
    ),
    "trashnet": DatasetSpec(
        name="trashnet",
        url="https://github.com/garythung/trashnet",
        license="unverified - check before publishing",
        license_url="https://github.com/garythung/trashnet",
        size_hint=(
            "2,527 images (README); dataset-resized.zip is 42,834,870 bytes "
            "(reported by the GitHub API)"
        ),
        label_format="Class folders only - NO bounding boxes (classification dataset)",
        needs="Nothing beyond disk space; small enough for a quick experiment.",
        verified=False,
        verified_from="",
        notes=(
            "The repo's MIT badge covers the CODE. The README says only 'please give "
            "a citation' for the images and hosts the original archive on Google Drive; "
            "no image licence statement was found. TrashNet cannot fine-tune a detector "
            "because it has no boxes -- it is listed to rule it out explicitly."
        ),
    ),
}


def _human_bytes(n: int) -> str:
    """Only for console output; the exact byte counts live in the constants above."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{n} B"


def list_datasets() -> None:
    """Print every candidate with its licence status -- no network access."""
    for spec in REGISTRY.values():
        flag = "VERIFIED" if spec.verified else "UNVERIFIED - check before publishing"
        print(f"\n[{spec.name}] {flag}")
        print(f"  source:       {spec.url}")
        print(f"  licence:      {spec.license}")
        print(f"  licence url:  {spec.license_url or '(none found)'}")
        print(f"  verified from:{spec.verified_from or ' NOT VERIFIED'}")
        print(f"  size:         {spec.size_hint}")
        print(f"  labels:       {spec.label_format}")
        print(f"  needs:        {spec.needs}")
        print(f"  note:         {spec.notes}")


def _stream_http(url: str, out_path: Path, expected_bytes: int | None) -> Path:
    """Stream one HTTPS file to disk with a progress line.

    We stream rather than read the whole archive into memory: TACO is ~2.7 GB and
    a Colab kernel does not have that kind of RAM to spare. A User-Agent is sent
    because some hosts reject the default urllib agent.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "waste-detection-pipeline/1.0"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  already present, skipping: {out_path} ({_human_bytes(out_path.stat().st_size)})")
        return out_path

    print(f"  GET {url}")
    with urllib.request.urlopen(request, timeout=60) as response, out_path.open("wb") as fh:
        header = response.headers.get("Content-Length")
        total = int(header) if header and header.isdigit() else expected_bytes
        done = 0
        while True:
            chunk = response.read(1024 * 1024)  # 1 MiB chunks keep memory flat
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r  {_human_bytes(done)} / {_human_bytes(total)} "
                    f"({100.0 * done / total:5.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r  {_human_bytes(done)}", end="", flush=True)
    print(f"\n  saved: {out_path}")
    return out_path


def _download_zerowaste(dest_dir: Path, dry_run: bool) -> Path | None:
    """Use the official roboflow client, but only with an explicit API key."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if dry_run:
        print(
            f"  plan: Roboflow workspace/project {ZEROWASTE_WORKSPACE}/{ZEROWASTE_PROJECT} "
            f"version {ZEROWASTE_VERSION} -> yolov8 export into {dest_dir}"
        )
        print("  size: unknown (source page not reachable from this environment)")
        print(f"  needs: ROBOFLOW_API_KEY (currently {'set' if key else 'MISSING'})")
        return None
    if not key:
        # Say exactly which variable is missing -- do not silently fall back to a
        # different dataset or pretend a private dataset downloaded.
        print("zerowaste skipped: ROBOFLOW_API_KEY is not set; export it and re-run.")
        return None

    from roboflow import Roboflow  # imported lazily: no key -> no dependency needed

    rf = Roboflow(api_key=key)
    project = rf.workspace(ZEROWASTE_WORKSPACE).project(ZEROWASTE_PROJECT)
    folder = project.version(ZEROWASTE_VERSION).download("yolov8", location=str(dest_dir))
    print(f"  saved: {folder}")
    return Path(folder)


def download(name: str, dest: str = "data/raw", dry_run: bool = False) -> Path | None:
    """Download one registered dataset (or only print the plan with dry_run=True).

    Returns the downloaded file/folder, or None when skipped / dry-run.
    """
    spec = REGISTRY[name]  # argparse `choices` guards this at the CLI
    dest_dir = Path(dest) / name
    print(f"\n== {name} ==")
    print(f"  licence ({'verified' if spec.verified else 'UNVERIFIED'}): {spec.license}")
    if dry_run:
        print(f"  destination: {dest_dir}")
        print(f"  size: {spec.size_hint}")
        if name == "zerowaste":
            return _download_zerowaste(dest_dir, dry_run=True)
        url = TACO_ZENODO_ZIP if name == "taco" else TRASHNET_ZIP
        expected = TACO_ZIP_BYTES if name == "taco" else TRASHNET_ZIP_BYTES
        print(f"  would GET: {url}")
        print(f"  approximate bytes: {_human_bytes(expected)}")
        print("  (nothing was downloaded: --dry-run)")
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    if name == "zerowaste":
        return _download_zerowaste(dest_dir, dry_run=False)
    if name == "taco":
        return _stream_http(TACO_ZENODO_ZIP, dest_dir / "TACO.zip", TACO_ZIP_BYTES)
    if name == "trashnet":
        return _stream_http(TRASHNET_ZIP, dest_dir / "dataset-resized.zip", TRASHNET_ZIP_BYTES)
    raise KeyError(f"no download implementation for {name!r}")  # pragma: no cover


def _load_class_map(raw: str) -> dict[int | str, int]:
    """Load class_map from an inline JSON object or a JSON file.

    JSON object keys are always strings, so numeric COCO category ids arrive as
    "1", "2", ...; coco_to_yolo() also tries the integer form.
    """
    text = Path(raw).read_text(encoding="utf-8") if not raw.lstrip().startswith("{") else raw
    data = json.loads(text)
    if not isinstance(data, dict) or not data:
        raise ValueError("class_map must be a non-empty JSON object like {'1': 0, '2': 1}")
    out: dict[int | str, int] = {}
    for key, value in data.items():
        try:
            parsed_key: int | str = int(key)
        except (TypeError, ValueError):
            parsed_key = str(key)
        out[parsed_key] = int(value)
    return out


def _write_data_yaml(out_dir: Path, names: dict[int, str]) -> Path:
    """Write the ultralytics data.yaml.

    ``path: .`` is relative to the yaml file itself, which keeps the dataset
    portable between Colab Drive mounts and a local checkout. The train/val
    folders are created by split_dataset(); run it before training.
    """
    lines = [
        "# Generated by phase_b/data_prep.py. Run `split` before training.",
        "path: .",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for idx in sorted(names):
        # json.dumps gives a valid double-quoted YAML scalar.
        lines.append(f"  {idx}: {json.dumps(str(names[idx]))}")
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def coco_to_yolo(
    json_path: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    class_map: Mapping[int | str, int],
) -> Path:
    """Convert a COCO annotation file to an ultralytics YOLO dataset folder.

    Writes out_dir/images/*.jpg, out_dir/labels/*.txt and out_dir/data.yaml.
    Images with no (mapped) annotations are skipped entirely -- no empty label
    file, because ultralytics treats an empty label as a training image with no
    objects and that silently skews the class balance.

    class_map keys may be COCO category ids (int or numeric string) or category
    names; values must be contiguous YOLO class indices starting at 0.
    """
    json_path, images_dir, out_dir = Path(json_path), Path(images_dir), Path(out_dir)
    coco = json.loads(json_path.read_text(encoding="utf-8"))
    categories: Sequence[Mapping[str, Any]] = coco.get("categories", [])
    id_to_name = {int(c["id"]): str(c["name"]) for c in categories}

    # JSON object keys are always strings ("1"), but COCO category ids are ints.
    # Normalise here so direct Python callers behave the same as the CLI loader.
    normalised: dict[int | str, int] = {}
    for key, idx in class_map.items():
        norm_key: int | str = int(key) if isinstance(key, str) and key.isdigit() else key
        normalised[norm_key] = int(idx)
    class_map = normalised

    # Accept both id-based and name-based maps, and present readable names in yaml.
    names: dict[int, str] = {}
    for key, idx in class_map.items():
        cls = int(idx)
        label = id_to_name.get(key, str(key)) if isinstance(key, int) else str(key)
        names[cls] = label

    def yolo_class(cat_id: int) -> int | None:
        cat_id = int(cat_id)
        if cat_id in class_map:
            return int(class_map[cat_id])
        name = id_to_name.get(cat_id)
        if name is not None and name in class_map:
            return int(class_map[name])
        return None  # class deliberately not used by this demo

    by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("bbox") and ann.get("image_id") is not None:
            by_image[int(ann["image_id"])].append(ann)

    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = annotations = 0
    for image in coco.get("images", []):
        anns = by_image.get(int(image["id"]), [])
        mapped = [a for a in anns if yolo_class(a["category_id"]) is not None]
        if not mapped:
            skipped += 1  # no annotations (or only unmapped classes) -> skip
            continue

        file_name = Path(str(image["file_name"])).name  # COCO paths can have subdirs
        src = images_dir / file_name
        if not src.exists():
            print(f"  warning: missing image {src}, skipping")
            skipped += 1
            continue

        iw, ih = int(image.get("width") or 0), int(image.get("height") or 0)
        if not iw or not ih:
            # Most COCO files carry width/height; fall back to decoding the header.
            from PIL import Image  # available via torchvision/ultralytics in Colab

            with Image.open(src) as im:
                iw, ih = im.size

        lines: list[str] = []
        for ann in mapped:
            x, y, w, h = (float(v) for v in ann["bbox"][:4])
            if w <= 0 or h <= 0:
                continue
            cls = yolo_class(ann["category_id"])
            # COCO boxes may stick a pixel or two outside the image (common
            # annotation quirk). ultralytics rejects coordinates outside [0, 1],
            # so clamp before writing instead of letting training crash later.
            cx = min(1.0, max(0.0, (x + w / 2.0) / iw))
            cy = min(1.0, max(0.0, (y + h / 2.0) / ih))
            bw = min(1.0, max(0.0, w / iw))
            bh = min(1.0, max(0.0, h / ih))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            skipped += 1
            continue
        shutil.copy2(src, img_dir / file_name)
        (lbl_dir / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        written += 1
        annotations += len(lines)

    yaml_path = _write_data_yaml(out_dir, names)
    print(
        f"Converted {written} images / {annotations} boxes into {out_dir} "
        f"({skipped} images skipped)."
    )
    print(f"Wrote {yaml_path}; next step: python phase_b/data_prep.py split --out {out_dir}")
    return yaml_path


def split_dataset(out_dir: str | Path, val_fraction: float = 0.2, seed: int = 0) -> Path:
    """Move labelled images deterministically into train/ and val/ subfolders.

    Uses its own Random(seed), not the global random state, so the split is
    reproducible regardless of what else the notebook has called.
    """
    out_dir = Path(out_dir)
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    if (img_dir / "train").exists() or (lbl_dir / "train").exists():
        print("split_dataset: already split; use a fresh --out directory to re-split.")
        return out_dir

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")}
    stems = sorted(images.keys() & labels.keys())
    if len(stems) < 2:
        raise ValueError("need at least two labelled images to make a train/val split")

    # Keep the val set non-empty and leave at least one training image.
    n_val = max(1, min(len(stems) - 1, round(len(stems) * val_fraction)))
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    val_stems = set(shuffled[:n_val])

    for split in ("train", "val"):
        (img_dir / split).mkdir(parents=True, exist_ok=True)
        (lbl_dir / split).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        split = "val" if stem in val_stems else "train"
        shutil.move(str(images[stem]), str(img_dir / split / images[stem].name))
        shutil.move(str(labels[stem]), str(lbl_dir / split / labels[stem].name))

    # Provenance: stats() reads this instead of guessing the split size.
    (out_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "val_fraction": val_fraction,
                "n_train": len(stems) - n_val,
                "n_val": n_val,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Split {len(stems)} images: {len(stems) - n_val} train / {n_val} val (seed={seed}).")
    return out_dir


def _yaml_names(data_yaml: Path) -> dict[int, str]:
    """Read just the `names:` block; avoids a PyYAML dependency for stats()."""
    names: dict[int, str] = {}
    in_names = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip().isdigit():
            value = value.strip()
            names[int(key)] = json.loads(value) if value.startswith('"') else value.strip("'\"")
    return names


def stats(out_dir: str | Path) -> dict[str, Any]:
    """Print dataset counts and append one standard-schema row to results/dataset.csv.

    No timing here: this is inventory, not a benchmark, so time/throughput are
    explicit zeros and the interesting facts go in `extra`.
    """
    out_dir = Path(out_dir)
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    is_split = (img_dir / "train").exists() and (lbl_dir / "train").exists()
    if is_split:
        image_files = [
            p
            for split in ("train", "val")
            for p in (img_dir / split).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
        label_files = [p for split in ("train", "val") for p in (lbl_dir / split).glob("*.txt")]
    else:
        image_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        label_files = list(lbl_dir.glob("*.txt"))

    hist: Counter[int] = Counter()
    for label in label_files:
        for line in label.read_text(encoding="utf-8").splitlines():
            token = line.split(maxsplit=1)[0] if line.strip() else ""
            if token.isdigit():
                hist[int(token)] += 1

    data_yaml = out_dir / "data.yaml"
    names = _yaml_names(data_yaml) if data_yaml.exists() else {}
    split_info = json.loads((out_dir / "split.json").read_text(encoding="utf-8")) if (
        out_dir / "split.json"
    ).exists() else {}
    n_val = int(split_info.get("n_val", 0))

    print(f"\nDataset: {out_dir.name}")
    print(f"  images:      {len(image_files)}")
    print(f"  label files: {len(label_files)}")
    print(f"  boxes:       {sum(hist.values())}")
    print("  per-class histogram:")
    for cls in sorted(hist):
        print(f"    {cls} {names.get(cls, '(unnamed)')}: {hist[cls]}")
    print(f"  val images:  {n_val}")

    append_row(
        bench="dataset",
        variant=out_dir.name or "dataset",
        size=len(image_files),
        size_unit="images",
        time_ms_mean=0.0,
        time_ms_std=0.0,
        throughput=0.0,
        throughput_unit="",
        extra=f"classes={len(hist)},val={n_val}",
        filename="dataset.csv",
    )
    return {
        "images": len(image_files),
        "labels": len(label_files),
        "boxes": sum(hist.values()),
        "histogram": dict(hist),
        "val": n_val,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list candidate datasets and licence status")
    dl = sub.add_parser("download", help="download a registered dataset")
    dl.add_argument("--name", choices=sorted(REGISTRY), required=True)
    dl.add_argument("--dest", default="data/raw")
    dl.add_argument("--dry-run", action="store_true", help="print the plan only")

    conv = sub.add_parser("convert", help="COCO JSON -> YOLO folder")
    conv.add_argument("--json", dest="json_path", required=True)
    conv.add_argument("--images", dest="images_dir", required=True)
    conv.add_argument("--out", dest="out_dir", required=True)
    conv.add_argument(
        "--class-map",
        required=True,
        help="JSON object: COCO category id (or name) -> YOLO class index, e.g. '{\"1\": 0}'",
    )

    sp = sub.add_parser("split", help="make deterministic train/val folders")
    sp.add_argument("--out", dest="out_dir", required=True)
    sp.add_argument("--val-fraction", type=float, default=0.2)
    sp.add_argument("--seed", type=int, default=0)

    st = sub.add_parser("stats", help="print inventory and append results/dataset.csv")
    st.add_argument("--out", dest="out_dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Accept `--list` / `--download` as well as `list` / `download`.
    if args_list and args_list[0].startswith("--"):
        args_list[0] = args_list[0][2:]
    args = _build_parser().parse_args(args_list)

    if args.command == "list":
        list_datasets()
    elif args.command == "download":
        download(args.name, args.dest, dry_run=args.dry_run)
    elif args.command == "convert":
        coco_to_yolo(args.json_path, args.images_dir, args.out_dir, _load_class_map(args.class_map))
    elif args.command == "split":
        split_dataset(args.out_dir, args.val_fraction, args.seed)
    elif args.command == "stats":
        stats(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
