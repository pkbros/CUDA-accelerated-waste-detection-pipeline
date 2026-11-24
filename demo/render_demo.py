#!/usr/bin/env python3
"""
demo/render_demo.py -- render the 60-90 s interview demo for the CUDA waste-detection
PROTOTYPE.  This script does not measure anything: it only turns measured numbers into
pictures.  Rendering is offline; the overlays report per-frame timings captured earlier.

==============================================================================
RULE 1 -- NO INVENTED NUMBERS
==============================================================================
Every per-frame overlay number is read at runtime from --frames-csv.  If that file is
missing the script exits with instructions to run the Colab benchmark first.

REQUIRED CSV SCHEMA (results/demo_frame_timings.csv), header line exactly:

  frame,baseline_decode_ms,baseline_preprocess_ms,baseline_infer_ms,baseline_post_ms,baseline_fps,baseline_detections,opt_decode_ms,opt_preprocess_ms,opt_infer_ms,opt_post_ms,opt_fps,opt_detections,baseline_boxes,opt_boxes

  * *_ms      : per-stage milliseconds, one row per frame (float).
  * *_fps     : measured FPS for that frame (float).
  * *_detections : detection count for that frame (int).
  * baseline_boxes / opt_boxes : JSON strings of the form
        [[x1,y1,x2,y2,cls,score], ...]
    coordinates in the SOURCE frame's pixel space.  An empty string or "[]" means
    no boxes: nothing is drawn and no detection is claimed.

Result cards read vector_add.csv, infer.csv and batching.csv from results/ at runtime.
A missing/absent value produces the words "result pending" plus a warning -- never an
estimate.  results/summary.md is purposefully not parsed: the CSVs are the source of
truth and the summary is generated from them.

==============================================================================
RULE 2 -- HONEST LABELS
==============================================================================
The only status word used on screen is "prototype"; no shipped-system or
live-system claim is made anywhere.  Every frame carries the note
    "overlay shows measured per-frame timings; video rendered offline"
and the GPU model is read from results/device.json for the footer.

==============================================================================
RULE 3 -- DEPENDENCIES
==============================================================================
numpy, Pillow and ffmpeg (via subprocess) only.  No new frameworks.  OpenCV is
imported lazily inside functions so that this module imports (and --help runs) on a
machine without cv2.  ffmpeg is checked at runtime; this script never installs packages.

==============================================================================
TIMELINE (80 s total, rendered at --fps-independent constants below)
==============================================================================
  0-8 s    title + "where does the time go?"
  8-35 s   split-screen: BASELINE vs OPTIMIZED on the SAME frames
  35-50 s  overview panels B / C / D, one at a time, each with a one-line caption
  50-62 s  nsys timeline image + "copies and kernels overlapping across streams"
  62-75 s  "What I got wrong" card with REAL negative results read from the CSVs
  75-80 s  repo URL + one-line summary

Usage:
  python demo/render_demo.py --help
  python demo/render_demo.py            # uses defaults under demo/ and results/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
DEMO = REPO_ROOT / "demo"

# 1280x720 H.264.  FPS is a rendering constant, not a benchmark number.
W, H = 1280, 720
FPS = 24
TOTAL_SECONDS = 80
SHORT_SECONDS = 15
SPLIT_START, SPLIT_END = 8.0, 35.0

PANEL_W, PANEL_H, PANEL_Y = 628, 536, 52
LEFT_X, RIGHT_X = 4, 644
NOTE_Y, NOTE_H = 588, 44
FOOTER_Y = 632

FRAME_COLUMNS = (
    "frame",
    "baseline_decode_ms", "baseline_preprocess_ms", "baseline_infer_ms",
    "baseline_post_ms", "baseline_fps", "baseline_detections",
    "opt_decode_ms", "opt_preprocess_ms", "opt_infer_ms",
    "opt_post_ms", "opt_fps", "opt_detections",
    "baseline_boxes", "opt_boxes",
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def fmt_ms(value: float | None) -> str:
    return "pending" if value is None else f"{value:.2f} ms"


def fmt_num(value: float | None, digits: int = 1) -> str:
    return "pending" if value is None else f"{value:.{digits}f}"


_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a readable font, falling back across platforms. Never fails."""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    if bold:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ]
    for name in candidates:
        try:
            f = ImageFont.truetype(name, size)
            _FONT_CACHE[key] = f
            return f
        except OSError:
            continue
    try:
        f = ImageFont.load_default(size)  # Pillow >= 10
    except TypeError:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def line_height(fnt: ImageFont.FreeTypeFont, leading: int = 6) -> int:
    return int(getattr(fnt, "size", 14)) + leading


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or draw.textlength(trial, font=fnt) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paragraph(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fnt,
              fill, max_width: int, leading: int = 6) -> int:
    lh = line_height(fnt, leading)
    for k, line in enumerate(wrap_lines(draw, text, fnt, max_width)):
        draw.text((x, y + k * lh), line, font=fnt, fill=fill)
    return y + len(wrap_lines(draw, text, fnt, max_width)) * lh


# Class names of the pretrained COCO detector (a label lookup, not a measured value).
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


# ---------------------------------------------------------------------------
# Input 1: the per-frame timing CSV (mandatory)
# ---------------------------------------------------------------------------
def parse_boxes(text: str | None) -> list[tuple[float, float, float, float, int, float]]:
    raw = (text or "").strip()
    if not raw or raw.lower() in {"[]", "none", "null"}:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("boxes field must be a JSON list")
    boxes = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            raise ValueError("each box must be [x1,y1,x2,y2,cls,score]")
        boxes.append((float(item[0]), float(item[1]), float(item[2]), float(item[3]),
                      int(float(item[4])), float(item[5])))
    return boxes


def load_frame_timings(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"ERROR: frame timings CSV not found: {p}\n"
            "Run the Colab benchmark first (notebooks/run_all.ipynb) so it writes\n"
            "results/demo_frame_timings.csv, then re-run this renderer.\n"
            "No numbers are fabricated here; there is nothing to render without it."
        )
    rows: list[dict] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in FRAME_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"ERROR: {p} is missing required column(s): {', '.join(missing)}\n"
                "Expected header:\n  " + ",".join(FRAME_COLUMNS)
            )
        for lineno, raw in enumerate(reader, start=2):
            try:
                row = {"frame": int(raw["frame"])}
                for key in FRAME_COLUMNS:
                    if key in ("frame", "baseline_boxes", "opt_boxes"):
                        continue
                    row[key] = float(raw[key])
                row["baseline_boxes"] = parse_boxes(raw["baseline_boxes"])
                row["opt_boxes"] = parse_boxes(raw["opt_boxes"])
                rows.append(row)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise SystemExit(f"ERROR: malformed row {lineno} in {p}: {exc}")
    if not rows:
        raise SystemExit(
            f"ERROR: {p} has a header but no data rows. Run the Colab benchmark first."
        )
    return rows


# ---------------------------------------------------------------------------
# Input 2: frames (same frame shown in both panels). cv2 is imported lazily.
# ---------------------------------------------------------------------------
class FrameSource:
    """A directory of still images or a video file; cv2 is never imported at module load."""

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        self._cap = None
        self._paths: list[str] = []
        self._cache: dict[int, Image.Image | None] = {}
        self._warned = False
        self.kind = "none"
        if self.path and self.path.is_dir():
            self._paths = sorted(
                str(q) for q in self.path.iterdir() if q.suffix.lower() in IMG_EXTS
            )
            self.kind = "images" if self._paths else "none"
        elif self.path and self.path.is_file():
            self.kind = "video"
        elif self.path:
            warn(f"--video '{path}' does not exist; split-screen will have no frame image")

    def count(self) -> int:
        if self.kind == "images":
            return len(self._paths)
        if self.kind == "video":
            try:
                import cv2  # lazy: Colab has it; local import must not require it

                cap = self._cap or self._open()
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                return n if n > 0 else 0
            except Exception as exc:
                if not self._warned:
                    warn(f"could not count frames in '{self.path}': {exc}")
                    self._warned = True
                return 0
        return 0

    def _open(self):
        import cv2  # lazy

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture could not open '{self.path}'")
        self._cap = cap
        return cap

    def get(self, index: int) -> Image.Image | None:
        if index in self._cache:
            return self._cache[index]
        image: Image.Image | None = None
        try:
            if self.kind == "images":
                image = Image.open(self._paths[index % len(self._paths)]).convert("RGB")
            elif self.kind == "video":
                import cv2  # lazy

                cap = self._cap or self._open()
                n = self.count()
                if n:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index % n)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError("cv2 read failed after seek")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            if not self._warned:
                warn(f"frame {index} could not be decoded: {exc}")
                self._warned = True
        if len(self._cache) > 96:
            self._cache.clear()
        self._cache[index] = image
        return image


# ---------------------------------------------------------------------------
# Input 3: result CSVs, device record, repo URL
# ---------------------------------------------------------------------------
def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        warn(f"{path.name} not found; the affected card will say 'result pending'")
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return [row for row in csv.DictReader(fh)]
    except (OSError, csv.Error) as exc:
        warn(f"could not read {path.name}: {exc}; card will say 'result pending'")
        return []


def pick_row(rows: list[dict[str, str]], variant: str, size: str | None = None):
    for row in rows:
        if row.get("variant") == variant and (size is None or row.get("size") == str(size)):
            return row
    return None


def metric(rows: list[dict[str, str]], variant: str, column: str,
           size: str | None = None) -> float | None:
    row = pick_row(rows, variant, size)
    if row is None:
        return None
    try:
        return float(row[column])
    except (TypeError, ValueError, KeyError):
        return None


def extra_float(rows: list[dict[str, str]], variant: str, key: str,
                size: str | None = None) -> float | None:
    row = pick_row(rows, variant, size)
    if row is None:
        return None
    for part in (row.get("extra") or "").replace(",", ";").split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k.strip() == key:
            try:
                return float(v.strip())
            except ValueError:
                return None
    return None


def read_device() -> str:
    path = RESULTS / "device.json"
    if path.is_file():
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            gpu = info.get("gpu_name", "unknown GPU")
            arch = info.get("arch", "?")
            return f"GPU: {gpu} ({arch})"
        except (json.JSONDecodeError, OSError) as exc:
            warn(f"could not read device.json: {exc}")
    warn("device.json not found; GPU footer will say unknown")
    return "GPU: unknown (no device record found)"


def repo_url_from_git() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return out[:-4] if out.endswith(".git") else out
    except (OSError, subprocess.SubprocessError):
        return "repo URL pending (pass --repo-url)"


def find_overview_image(arg: str) -> Path | None:
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        found = sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)
        return found[0] if found else None
    for pattern in ("*nsys*.png", "*timeline*.png", "*nsys*.jpg", "*timeline*.jpg"):
        found = sorted(RESULTS.glob(pattern)) + sorted(DEMO.glob(pattern))
        if found:
            return found[0]
    return None


class Metrics:
    """All result-card numbers, resolved once at startup from the CSVs."""

    def __init__(self) -> None:
        self.va = read_rows(RESULTS / "vector_add.csv")
        self.inf = read_rows(RESULTS / "infer.csv")
        self.batch = read_rows(RESULTS / "batching.csv")
        self.va_scalar_ms = metric(self.va, "gpu_scalar", "time_ms_mean", 50000000)
        self.va_float4_ms = metric(self.va, "gpu_float4", "time_ms_mean", 50000000)
        self.va_scalar_gbs = metric(self.va, "gpu_scalar", "throughput", 50000000)
        self.va_float4_gbs = metric(self.va, "gpu_float4", "throughput", 50000000)
        self.fp32_ms = metric(self.inf, "pt_fp32", "time_ms_mean", 1)
        self.fp16_ms = metric(self.inf, "pt_fp16", "time_ms_mean", 1)
        self.ort_ms = metric(self.inf, "ort_cuda_fp32", "time_ms_mean", 1)
        self.serial_ff_4 = extra_float(self.batch, "serial_batch", "fps_per_feed", 4)
        self.pipe_ff_4 = extra_float(self.batch, "pipelined_batch", "fps_per_feed", 4)
        self.serial_ff_1 = extra_float(self.batch, "serial_batch", "fps_per_feed", 1)
        self.pipe_ff_1 = extra_float(self.batch, "pipelined_batch", "fps_per_feed", 1)
        missing = [name for name, value in vars(self).items()
                   if name.endswith(("_ms", "_gbs", "_ff_1", "_ff_4")) and value is None]
        if missing:
            warn("result-card values not found in CSVs: " + ", ".join(missing)
                 + " (those cards will say 'result pending')")


# ---------------------------------------------------------------------------
# Card drawing helpers
# ---------------------------------------------------------------------------
def base_card(title: str, kicker: str = "CUDA WASTE DETECTION PROTOTYPE"):
    img = Image.new("RGB", (W, H), (9, 13, 21))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 6), fill=(70, 205, 205))
    d.text((56, 38), kicker, font=font(18, True), fill=(120, 195, 205))
    d.text((56, 72), title, font=font(42, True), fill=(238, 244, 250))
    d.line((56, 138, W - 56, 138), fill=(48, 68, 86), width=2)
    return img, d


def draw_footer(d: ImageDraw.ImageDraw, gpu_line: str, right: str = "") -> None:
    d.rectangle((0, FOOTER_Y, W, H), fill=(13, 18, 27))
    d.line((0, FOOTER_Y, W, FOOTER_Y), fill=(44, 60, 78), width=2)
    d.text((18, FOOTER_Y + 8),
           "PROTOTYPE  |  overlay shows measured per-frame timings; video rendered offline",
           font=font(16), fill=(198, 212, 226))
    d.text((18, FOOTER_Y + 34), gpu_line, font=font(15), fill=(146, 162, 184))
    if right:
        tw = d.textlength(right, font=font(15))
        d.text((W - 18 - tw, FOOTER_Y + 34), right, font=font(15), fill=(146, 162, 184))


def draw_table(d: ImageDraw.ImageDraw, x: int, y: int, headers: list[str],
               rows: list[list[str]], widths: list[int], accent) -> int:
    fnt = font(20)
    col_x = [x]
    for wd in widths[:-1]:
        col_x.append(col_x[-1] + wd)
    for col, head in enumerate(headers):
        d.text((col_x[col], y), head, font=font(19, True), fill=accent)
    y += 34
    for row in rows:
        for col, cell in enumerate(row):
            d.text((col_x[col], y), cell, font=fnt, fill=(224, 232, 240))
        y += 32
    return y


def draw_title_card(gpu_line: str) -> Image.Image:
    img, d = base_card("CUDA-Accelerated Waste Detection",
                       kicker="PROTOTYPE  |  interview walkthrough")
    d.text((56, 178), "where does the time go?", font=font(52, True), fill=(255, 190, 110))
    d.text((58, 260),
           "Same frames. Two pipelines. One question: which optimisation actually paid?",
           font=font(23), fill=(200, 214, 228))
    agenda = [
        "1   title + the question",
        "2   split-screen: BASELINE vs OPTIMIZED on the same frames",
        "3   overview panels B / C / D",
        "4   Nsight Systems timeline",
        "5   what I got wrong (measured regressions)",
        "6   repo + summary",
    ]
    y = 340
    for item in agenda:
        d.text((70, y), item, font=font(21), fill=(170, 190, 208))
        y += 40
    d.text((58, 610), "Measured benchmarks, no estimated numbers.",
           font=font(19), fill=(120, 140, 160))
    draw_footer(d, gpu_line)
    return img


def fit_frame(frame: Image.Image | None, pw: int, ph: int):
    """Contain-fit a source frame into a panel; return image, scale and offsets."""
    canvas = Image.new("RGB", (pw, ph), (14, 17, 24))
    if frame is None:
        cd = ImageDraw.Draw(canvas)
        paragraph(cd, 24, ph // 2 - 20,
                  "no frame source: pass --video (a video file or a directory of stills)",
                  font(20), (150, 165, 185), pw - 48)
        return canvas, 1.0, 0, 0
    fw, fh = frame.size
    scale = min(pw / max(1, fw), ph / max(1, fh))
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = frame.resize((nw, nh), Image.Resampling.BILINEAR)
    ox, oy = (pw - nw) // 2, (ph - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def draw_boxes(panel: Image.Image, boxes, scale: float, ox: int, oy: int, color) -> None:
    if not boxes:
        return
    d = ImageDraw.Draw(panel)
    pw, ph = panel.size
    fnt = font(15, True)
    for x1, y1, x2, y2, cls, score in boxes:
        X1, X2 = ox + x1 * scale, ox + x2 * scale
        Y1, Y2 = oy + y1 * scale, oy + y2 * scale
        X1, X2 = sorted((max(0, min(pw, X1)), max(0, min(pw, X2))))
        Y1, Y2 = sorted((max(0, min(ph, Y1)), max(0, min(ph, Y2))))
        if X2 <= X1 or Y2 <= Y1:
            continue
        d.rectangle((X1, Y1, X2, Y2), outline=color, width=2)
        name = COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else f"cls{cls}"
        label = f"{name} {score:.2f}"
        lw = d.textlength(label, font=fnt)
        ly = max(0, Y1 - 22)
        d.rectangle((X1, ly, X1 + lw + 8, ly + 20), fill=color)
        d.text((X1 + 4, ly + 1), label, font=fnt, fill=(8, 12, 18))


def rolling_fps(rows: list[dict], index: int, key: str, window: int = 8) -> float:
    lo = max(0, index - window + 1)
    values = np.array([rows[j][key] for j in range(lo, index + 1)], dtype=float)
    return float(np.mean(values))


def draw_split(rows: list[dict], index: int, source: FrameSource,
               gpu_line: str, frame_no: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (7, 9, 14))
    d = ImageDraw.Draw(img)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    d.rectangle((0, 0, W // 2 - 6, 48), fill=(58, 37, 18))
    d.rectangle((W // 2 + 6, 0, W, 48), fill=(10, 50, 56))
    d.text((16, 12), "BASELINE  -  OpenCV CPU preprocess + PyTorch FP32",
           font=font(19, True), fill=(255, 196, 120))
    d.text((W // 2 + 22, 12), "OPTIMIZED  -  fused CUDA kernel + best backend + streams",
           font=font(19, True), fill=(140, 232, 242))

    specs = [
        ("baseline", LEFT_X, "decode", "preprocess", "infer", "post", "fps",
         "detections", "boxes", (255, 179, 71)),
        ("opt", RIGHT_X, "decode", "preprocess", "infer", "post", "fps",
         "detections", "boxes", (86, 220, 210)),
    ]
    for prefix, x0, dk, pk, ik, pok, fk, nk, bk, color in specs:
        row = rows[index % len(rows)]
        frame = source.get(frame_no % max(1, source.count())) if source.count() else None
        panel, scale, ox, oy = fit_frame(frame, PANEL_W, PANEL_H)
        boxes = row[f"{prefix}_boxes"]
        draw_boxes(panel, boxes, scale, ox, oy, color)
        img.paste(panel, (x0, PANEL_Y))

        top = PANEL_Y + PANEL_H - 160
        od.rounded_rectangle((x0 + 12, top, x0 + PANEL_W - 12, PANEL_Y + PANEL_H - 12),
                             radius=10, fill=(0, 0, 0, 200))
        lines = [
            f"decode      {fmt_ms(row[f'{prefix}_{dk}_ms'])}",
            f"preprocess  {fmt_ms(row[f'{prefix}_{pk}_ms'])}",
            f"inference   {fmt_ms(row[f'{prefix}_{ik}_ms'])}",
            f"post        {fmt_ms(row[f'{prefix}_{pok}_ms'])}",
            f"FPS (8-row rolling)  {rolling_fps(rows, index % len(rows), f'{prefix}_{fk}'):.1f}",
            f"detections in CSV: {int(row[f'{prefix}_{nk}'])}"
            + ("" if boxes else "   (no boxes in CSV)"),
        ]
        ly = top + 12
        for line in lines:
            od.text((x0 + 26, ly), line, font=font(18, True), fill=(238, 244, 250))
            ly += 23

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle((0, NOTE_Y, W, NOTE_Y + NOTE_H), fill=(16, 22, 32))
    d.text((18, NOTE_Y + 6),
           "overlay shows measured per-frame timings; video rendered offline",
           font=font(18), fill=(215, 228, 240))
    tag = "PROTOTYPE - benchmark only"
    tw = d.textlength(tag, font=font(17))
    d.text((W - 18 - tw, NOTE_Y + 8), tag, font=font(17), fill=(120, 200, 205))
    draw_footer(d, gpu_line, f"frame {frame_no + 1}")
    return img


def draw_overview_card(which: int, m: Metrics, gpu_line: str) -> Image.Image:
    captions = {
        0: "both sit at the memory ceiling; 128-bit loads bought nothing",
        1: "best backend beats eager FP32 only slightly at batch 1",
        2: "at 4 feeds the pipelined loop regresses against serial",
    }
    titles = {0: "Overview B - vector add", 1: "Overview C - inference ladder",
              2: "Overview D - multi-feed batching"}
    img, d = base_card(titles[which], kicker="MEASURED RESULT PANEL  |  PROTOTYPE")
    accent = (120, 220, 215)
    if which == 0:
        draw_table(d, 70, 170, ["variant", "mean ms", "GB/s"], [
            ["gpu_scalar", fmt_ms(m.va_scalar_ms), fmt_num(m.va_scalar_gbs)],
            ["gpu_float4", fmt_ms(m.va_float4_ms), fmt_num(m.va_float4_gbs)],
        ], [250, 220, 200], accent)
        caption = (f"scalar {fmt_ms(m.va_scalar_ms)} / {fmt_num(m.va_scalar_gbs)} GB/s "
                   f"vs float4 {fmt_ms(m.va_float4_ms)} / {fmt_num(m.va_float4_gbs)} GB/s "
                   "- " + captions[0])
    elif which == 1:
        draw_table(d, 70, 170, ["backend", "mean ms"], [
            ["pt_fp32", fmt_ms(m.fp32_ms)],
            ["pt_fp16", fmt_ms(m.fp16_ms)],
            ["ort_cuda_fp32", fmt_ms(m.ort_ms)],
        ], [320, 260], accent)
        caption = (f"pt_fp32 {fmt_ms(m.fp32_ms)}, ort_cuda_fp32 {fmt_ms(m.ort_ms)} "
                   "- " + captions[1])
    else:
        draw_table(d, 70, 170, ["feeds", "serial FPS/feed", "pipelined FPS/feed"], [
            ["1", fmt_num(m.serial_ff_1), fmt_num(m.pipe_ff_1)],
            ["4", fmt_num(m.serial_ff_4), fmt_num(m.pipe_ff_4)],
        ], [180, 320, 360], accent)
        caption = (f"4 feeds: serial {fmt_num(m.serial_ff_4)} vs pipelined "
                   f"{fmt_num(m.pipe_ff_4)} FPS/feed - " + captions[2])
    paragraph(d, 70, 470, caption, font(23, True), (255, 195, 115), W - 140, leading=8)
    d.text((70, 600), "Result cards read results/*.csv at runtime; pending means not measured.",
           font=font(17), fill=(120, 140, 160))
    draw_footer(d, gpu_line)
    return img


def draw_nsys_card(image_path: Path | None, gpu_line: str) -> Image.Image:
    img, d = base_card("Nsight Systems timeline", kicker="PROFILING EVIDENCE  |  PROTOTYPE")
    caption = "copies and kernels overlapping across streams" if image_path is not None \
        else "timeline pending - no overlap claim until the nsys image is exported"
    if image_path is None:
        d.text((70, 210), "nsys timeline pending", font=font(34, True), fill=(255, 195, 115))
        paragraph(d, 70, 270,
                  "Export the nsys screenshot to results/nsys_timeline.png "
                  "(or pass --overview) and re-run. No timeline is drawn until it exists.",
                  font(21), (195, 210, 225), W - 140)
    else:
        try:
            shot = Image.open(image_path).convert("RGB")
            sw, sh = shot.size
            scale = min((W - 112) / max(1, sw), (H - 290) / max(1, sh))
            nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
            shot = shot.resize((nw, nh), Image.Resampling.BILINEAR)
            px, py = 56 + (W - 112 - nw) // 2, 166 + (H - 290 - nh) // 2
            d.rectangle((px - 2, py - 2, px + nw + 2, py + nh + 2),
                        outline=(60, 90, 110), width=2)
            img.paste(shot, (px, py))
            d = ImageDraw.Draw(img)
        except OSError as exc:
            warn(f"could not open nsys image '{image_path}': {exc}")
            d.text((70, 210), "nsys timeline pending (image unreadable)",
                   font=font(30, True), fill=(255, 195, 115))
    d.text((70, 600), caption, font=font(24, True), fill=(140, 232, 242))
    draw_footer(d, gpu_line)
    return img


def draw_wrong_card(m: Metrics, gpu_line: str) -> Image.Image:
    img, d = base_card("What I got wrong", kicker="MEASURED REGRESSIONS  |  PROTOTYPE")
    items = [
        ("float4 vs scalar loads",
         f"float4 {fmt_ms(m.va_float4_ms)} vs scalar {fmt_ms(m.va_scalar_ms)} "
         "- no win at the bandwidth ceiling (vector_add.csv)"),
        ("FP16 vs FP32 inference",
         f"FP16 {fmt_ms(m.fp16_ms)} vs FP32 {fmt_ms(m.fp32_ms)} "
         "- slower at batch 1 (infer.csv)"),
        ("pipelining multiple feeds",
         f"pipelined {fmt_num(m.pipe_ff_4)} vs serial {fmt_num(m.serial_ff_4)} FPS/feed "
         "at 4 feeds - regression (batching.csv)"),
    ]
    y = 174
    for title, detail in items:
        d.text((70, y), title, font=font(25, True), fill=(140, 232, 242))
        y = paragraph(d, 70, y + 34, detail, font(20), (224, 232, 240), W - 140, leading=4) + 26
    d.text((70, 600), "These are results, not failures hidden from the story.",
           font=font(18), fill=(120, 140, 160))
    draw_footer(d, gpu_line)
    return img


def draw_outro_card(repo_url: str, gpu_line: str) -> Image.Image:
    img, d = base_card("Repo + one-line summary", kicker="PROTOTYPE  |  interview close")
    url_font = font(30, True)
    while getattr(url_font, "size", 30) > 16 and d.textlength(repo_url, font=url_font) > W - 112:
        url_font = font(getattr(url_font, "size", 30) - 2, True)
    d.text((56, 200), repo_url, font=url_font, fill=(140, 232, 242))
    paragraph(d, 56, 300,
              "A measured CUDA prototype: fused preprocess, a chosen inference backend "
              "and stream overlap - with the regressions kept visible.",
              font(28), (238, 244, 250), W - 112, leading=10)
    d.text((56, 600), "prototype - every number came from a CSV",
           font=font(18), fill=(120, 140, 160))
    draw_footer(d, gpu_line)
    return img


# ---------------------------------------------------------------------------
# ffmpeg export
# ---------------------------------------------------------------------------
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit(
            "ERROR: ffmpeg is not on PATH. Install ffmpeg in the render environment "
            "(or run this notebook cell in Colab with ffmpeg available) and re-run. "
            "This script does not install system packages."
        )
    return exe


def ffmpeg_cmd(exe: str, out: Path) -> list[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    return [
        exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]


def frame_at(t: float, rows, source, m: Metrics, gpu_line: str,
             nsys_image, repo_url: str) -> Image.Image:
    if t < SPLIT_START:
        return draw_title_card(gpu_line)
    if t < SPLIT_END:
        index = int((t - SPLIT_START) * FPS)
        return draw_split(rows, index, source, gpu_line, index % max(1, source.count()))
    if t < 50:
        return draw_overview_card(min(2, int((t - 35) / 5)), m, gpu_line)
    if t < 62:
        return draw_nsys_card(nsys_image, gpu_line)
    if t < 75:
        return draw_wrong_card(m, gpu_line)
    return draw_outro_card(repo_url, gpu_line)


def render(args, rows, source, m: Metrics, gpu_line, nsys_image, repo_url: str) -> None:
    exe = ffmpeg_exe()
    out, short = Path(args.out), Path(args.short_out)
    full = subprocess.Popen(ffmpeg_cmd(exe, out), stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    clip = subprocess.Popen(ffmpeg_cmd(exe, short), stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    split_lo = int(SPLIT_START * FPS)
    split_hi = split_lo + int(SHORT_SECONDS * FPS)
    try:
        for i in range(TOTAL_SECONDS * FPS):
            raw = frame_at(i / FPS, rows, source, m, gpu_line, nsys_image, repo_url).tobytes()
            full.stdin.write(raw)
            if split_lo <= i < split_hi:
                clip.stdin.write(raw)
        full.stdin.close()
        clip.stdin.close()
    except BrokenPipeError:
        warn("ffmpeg closed its input early; see its error output below")
    for proc, path in ((full, out), (clip, short)):
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"ERROR: ffmpeg failed for {path} (exit {rc}):\n{err}")
    print(f"wrote {out}")
    print(f"wrote {short} ({SHORT_SECONDS}s of the split-screen)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the offline CUDA waste-detection PROTOTYPE demo (no invented numbers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Defaults: frame CSV under results/, footage/video and outputs under demo/.\n"
            "--frames-csv must exist; run the Colab benchmark first. ffmpeg must be on PATH."
        ),
    )
    p.add_argument("--frames-csv", default=str(RESULTS / "demo_frame_timings.csv"),
                   help="mandatory per-frame timing CSV (schema in this file's header)")
    p.add_argument("--video", default=str(DEMO / "demo_input.mp4"),
                   help="video file or directory of still frames (same frames in both panels)")
    p.add_argument("--out", default=str(DEMO / "demo.mp4"), help="full 80s H.264 mp4")
    p.add_argument("--short-out", default=str(DEMO / "demo_short.mp4"),
                   help="15s H.264 mp4 cut from the split-screen")
    p.add_argument("--overview", default=str(RESULTS / "nsys_timeline.png"),
                   help="nsys timeline screenshot for the 50-62s card")
    p.add_argument("--repo-url", default=None,
                   help="repo URL for the outro (default: git remote origin)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)  # --help exits here, before any file/ffmpeg access
    found = shutil.which("ffmpeg")
    print(f"ffmpeg: {found if found else 'NOT FOUND (rendering requires ffmpeg on PATH)'}")
    rows = load_frame_timings(args.frames_csv)
    print(f"loaded {len(rows)} measured frame rows from {args.frames_csv}")
    source = FrameSource(args.video)
    if source.count() == 0:
        warn("no frame source available; split-screen panels will show a placeholder")
    m = Metrics()
    gpu_line = read_device()
    nsys_image = find_overview_image(args.overview)
    if nsys_image is None:
        warn("nsys timeline image not found; its card will say pending")
    repo_url = args.repo_url or repo_url_from_git()
    render(args, rows, source, m, gpu_line, nsys_image, repo_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
