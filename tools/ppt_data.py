"""CSV/JSON loaders for the PowerPoint image pack.

Run from the repo root::

    python tools/make_ppt_pack.py

This module never reads ``results/summary.md`` and never invents a value:
every number it returns comes from a CSV/JSON file under ``results/`` (or from
``data/waste.yaml`` / val labels where explicitly noted).  A required file that
is absent raises :class:`MissingDataError` naming the file.

Optional inputs -- ``results/finetune.csv`` and
``results/pipeline_agreement.csv`` -- return ``None`` when absent or empty.
Callers must render an explicit "NOT RUN: <reason>" placeholder for them.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Every one of these is a source of measured numbers.  finetune.csv and
# pipeline_agreement.csv are deliberately excluded: they are pending.
REQUIRED_CSV = (
    "demo_frame_timings.csv",
    "preprocess.csv",
    "infer.csv",
    "pipeline.csv",
    "batching.csv",
    "vector_add.csv",
    "reduction.csv",
    "matmul.csv",
    "streams.csv",
    "dataset.csv",
    "profiling_ncu.csv",
)
OPTIONAL_CSV = ("finetune.csv", "pipeline_agreement.csv")

STANDARD_COLUMNS = (
    "bench",
    "variant",
    "size",
    "size_unit",
    "time_ms_mean",
    "time_ms_std",
    "throughput",
    "throughput_unit",
    "extra",
    "device",
    "arch",
)

FRAME_COLUMNS = (
    "frame",
    "baseline_decode_ms",
    "baseline_preprocess_ms",
    "baseline_infer_ms",
    "baseline_post_ms",
    "baseline_fps",
    "baseline_detections",
    "opt_decode_ms",
    "opt_preprocess_ms",
    "opt_infer_ms",
    "opt_post_ms",
    "opt_fps",
    "opt_detections",
    "baseline_boxes",
    "opt_boxes",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MissingDataError(FileNotFoundError):
    """Raised when a required results file is absent or malformed."""


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def parse_extra(raw: str | None) -> dict[str, str]:
    """Parse ``key=value`` pairs separated by commas and/or semicolons.

    Results CSVs are inconsistent: preprocess/batching use commas, infer uses
    semicolons.  Both are accepted; the first ``=`` splits each pair so values
    such as ``atol=0.0001/rtol=0.0001`` survive intact.
    """
    out: dict[str, str] = {}
    for part in re.split(r"[,;]", raw or ""):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _num(raw: Any, where: str) -> float:
    text = str(raw if raw is not None else "").strip()
    if not text:
        raise MissingDataError(f"{where}: empty numeric field")
    try:
        return float(text)
    except ValueError as exc:
        raise MissingDataError(f"{where}: invalid number {text!r}") from exc


def _int(raw: Any, where: str) -> int:
    value = _num(raw, where)
    if not float(value).is_integer():
        raise MissingDataError(f"{where}: expected an integer, got {value!r}")
    return int(value)


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Box:
    """One ``[x1, y1, x2, y2, cls, score]`` entry from a per-frame CSV."""

    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    score: float


@dataclass(frozen=True)
class FrameTiming:
    """One measured frame, with the row position kept for video mapping."""

    frame: int
    position: int
    baseline_decode_ms: float
    baseline_preprocess_ms: float
    baseline_infer_ms: float
    baseline_post_ms: float
    baseline_fps: float
    baseline_detections: int
    opt_decode_ms: float
    opt_preprocess_ms: float
    opt_infer_ms: float
    opt_post_ms: float
    opt_fps: float
    opt_detections: int
    baseline_boxes: tuple[Box, ...]
    opt_boxes: tuple[Box, ...]

    @property
    def baseline_stages_ms(self) -> tuple[float, float, float, float]:
        return (
            self.baseline_decode_ms,
            self.baseline_preprocess_ms,
            self.baseline_infer_ms,
            self.baseline_post_ms,
        )

    @property
    def opt_stages_ms(self) -> tuple[float, float, float, float]:
        return (
            self.opt_decode_ms,
            self.opt_preprocess_ms,
            self.opt_infer_ms,
            self.opt_post_ms,
        )

    @property
    def baseline_total_ms(self) -> float:
        return float(sum(self.baseline_stages_ms))

    @property
    def opt_total_ms(self) -> float:
        return float(sum(self.opt_stages_ms))


@dataclass(frozen=True)
class BenchRow:
    """A row from one of the standard benchmark CSVs."""

    bench: str
    variant: str
    size: int | None
    size_raw: str
    size_unit: str
    time_ms_mean: float
    time_ms_std: float
    throughput: float
    throughput_unit: str
    extra: dict[str, str]
    device: str
    arch: str
    source: str

    @property
    def batch(self) -> int | None:
        raw = self.extra.get("batch")
        if raw is None and self.size_unit == "images":
            raw = self.size_raw
        if raw is None:
            return None
        try:
            return int(float(raw))
        except ValueError:
            return None


@dataclass(frozen=True)
class DeviceInfo:
    gpu_name: str
    arch: str
    compute_capability: str
    raw: dict[str, Any]

    @property
    def footer_model(self) -> str:
        return self.gpu_name or "unknown GPU"


@dataclass(frozen=True)
class NcuPass:
    ident: str
    dram_pct: float
    sm_pct: float
    duration_ns: float | None


@dataclass(frozen=True)
class NcuResult:
    kernel: str
    passes: tuple[NcuPass, ...]

    @property
    def dram_mean(self) -> float:
        return float(sum(p.dram_pct for p in self.passes) / len(self.passes))

    @property
    def sm_mean(self) -> float:
        return float(sum(p.sm_pct for p in self.passes) / len(self.passes))


@dataclass(frozen=True)
class FinetuneResult:
    map50: float
    map50_name: str
    map50_95: float | None
    map50_95_name: str | None
    source: str


@dataclass(frozen=True)
class AgreementResult:
    score: float
    name: str
    source: str
    n_frames: int
    n_agree: int
    mean_iou: float | None


@dataclass(frozen=True)
class FrameStats:
    n_frames: int
    baseline_rolling_fps_mean: float
    opt_rolling_fps_mean: float
    baseline_run_fps: float
    opt_run_fps: float
    baseline_total_mean_ms: float
    opt_total_mean_ms: float

    @property
    def run_speedup(self) -> float:
        return self.opt_run_fps / self.baseline_run_fps

    @property
    def rolling_speedup(self) -> float:
        return self.opt_rolling_fps_mean / self.baseline_rolling_fps_mean


@dataclass
class PPTData:
    """All measured inputs, loaded once."""

    frames: list[FrameTiming] = field(default_factory=list)
    standard: dict[str, list[BenchRow]] = field(default_factory=dict)
    device: DeviceInfo | None = None
    ncu_rows: list[dict[str, str]] = field(default_factory=list)
    finetune: FinetuneResult | None = None
    agreement: AgreementResult | None = None
    notes: list[str] = field(default_factory=list)

    def rows(self, filename: str, bench: str | None = None) -> list[BenchRow]:
        rows = self.standard.get(filename, [])
        if bench is None:
            return rows
        return [r for r in rows if r.bench == bench]


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #


def require(path: Path) -> Path:
    if not path.is_file():
        raise MissingDataError(
            f"ERROR: required data file is missing: {path}\n"
            "No value is estimated in its place. Supply the file (it is written by "
            "the Colab benchmark) and re-run tools/make_ppt_pack.py from the repo root."
        )
    return path


def load_bench_csv(filename: str, *, required: bool = True) -> list[BenchRow]:
    path = RESULTS / filename
    if required:
        require(path)
    elif not path.is_file():
        warn(f"{filename}: absent; panels using it will show NOT RUN")
        return []
    rows: list[BenchRow] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in STANDARD_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise MissingDataError(f"{filename}: missing required column(s): {', '.join(missing)}")
        for lineno, raw in enumerate(reader, start=2):
            if not any(str(v or "").strip() for v in raw.values()):
                continue
            where = f"{filename} line {lineno}"
            size_raw = str(raw.get("size") or "").strip()
            try:
                size = int(float(size_raw)) if size_raw else None
            except ValueError as exc:
                raise MissingDataError(f"{where}: invalid size {size_raw!r}") from exc
            rows.append(
                BenchRow(
                    bench=str(raw.get("bench") or "").strip(),
                    variant=str(raw.get("variant") or "").strip(),
                    size=size,
                    size_raw=size_raw,
                    size_unit=str(raw.get("size_unit") or "").strip(),
                    time_ms_mean=_num(raw.get("time_ms_mean"), where),
                    time_ms_std=_num(raw.get("time_ms_std"), where),
                    throughput=_num(raw.get("throughput"), where),
                    throughput_unit=str(raw.get("throughput_unit") or "").strip(),
                    extra=parse_extra(raw.get("extra")),
                    device=str(raw.get("device") or "").strip(),
                    arch=str(raw.get("arch") or "").strip(),
                    source=filename,
                )
            )
    if required and not rows:
        raise MissingDataError(f"{filename}: header found but no data rows")
    return rows


def _parse_boxes(raw: str | None, where: str) -> tuple[Box, ...]:
    text = (raw or "").strip()
    if not text or text.lower() in {"[]", "none", "null"}:
        return ()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MissingDataError(f"{where}: invalid boxes JSON: {exc}") from exc
    if not isinstance(data, list):
        raise MissingDataError(f"{where}: boxes must be a JSON list")
    boxes: list[Box] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            raise MissingDataError(f"{where}: each box must be [x1,y1,x2,y2,cls,score]")
        boxes.append(
            Box(
                float(item[0]), float(item[1]), float(item[2]), float(item[3]),
                int(float(item[4])), float(item[5]),
            )
        )
    return tuple(boxes)


def load_frame_timings() -> list[FrameTiming]:
    path = require(RESULTS / "demo_frame_timings.csv")
    rows: list[FrameTiming] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in FRAME_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise MissingDataError(
                "demo_frame_timings.csv: missing required column(s): " + ", ".join(missing)
            )
        for lineno, raw in enumerate(reader, start=2):
            where = f"demo_frame_timings.csv line {lineno}"
            rows.append(
                FrameTiming(
                    frame=_int(raw.get("frame"), where),
                    position=lineno - 2,  # video mapping is by ROW POSITION
                    baseline_decode_ms=_num(raw.get("baseline_decode_ms"), where),
                    baseline_preprocess_ms=_num(raw.get("baseline_preprocess_ms"), where),
                    baseline_infer_ms=_num(raw.get("baseline_infer_ms"), where),
                    baseline_post_ms=_num(raw.get("baseline_post_ms"), where),
                    baseline_fps=_num(raw.get("baseline_fps"), where),
                    baseline_detections=_int(raw.get("baseline_detections"), where),
                    opt_decode_ms=_num(raw.get("opt_decode_ms"), where),
                    opt_preprocess_ms=_num(raw.get("opt_preprocess_ms"), where),
                    opt_infer_ms=_num(raw.get("opt_infer_ms"), where),
                    opt_post_ms=_num(raw.get("opt_post_ms"), where),
                    opt_fps=_num(raw.get("opt_fps"), where),
                    opt_detections=_int(raw.get("opt_detections"), where),
                    baseline_boxes=_parse_boxes(raw.get("baseline_boxes"), where),
                    opt_boxes=_parse_boxes(raw.get("opt_boxes"), where),
                )
            )
    if not rows:
        raise MissingDataError("demo_frame_timings.csv: header found but no data rows")
    return rows


def load_device() -> DeviceInfo:
    path = require(RESULTS / "device.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MissingDataError(f"device.json: could not parse: {exc}") from exc
    if not isinstance(raw, dict) or not str(raw.get("gpu_name") or "").strip():
        raise MissingDataError("device.json: no gpu_name; the footer cannot name the GPU")
    return DeviceInfo(
        gpu_name=str(raw["gpu_name"]).strip(),
        arch=str(raw.get("arch") or "").strip(),
        compute_capability=str(raw.get("compute_capability") or "").strip(),
        raw=raw,
    )


def read_ncu() -> list[dict[str, str]]:
    """Skip ncu's ``==PROF==`` preamble and parse the long-form metric table."""
    path = require(RESULTS / "profiling_ncu.csv")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = next(
        (
            i
            for i, line in enumerate(lines)
            if "Kernel Name" in line and "Metric Name" in line and "Metric Value" in line
        ),
        None,
    )
    if header is None:
        raise MissingDataError(
            "profiling_ncu.csv: no header containing Kernel Name / Metric Name / Metric Value"
        )
    return [
        row
        for row in csv.DictReader(lines[header:])
        if str(row.get("Metric Name") or "").strip()
    ]


# --------------------------------------------------------------------------- #
# derived helpers -- all values come from the loaded rows
# --------------------------------------------------------------------------- #


def per_frame_stats(frames: Iterable[FrameTiming]) -> FrameStats:
    items = list(frames)
    if not items:
        raise MissingDataError("demo_frame_timings.csv: no frames to summarise")
    base_total = sum(f.baseline_total_ms for f in items)
    opt_total = sum(f.opt_total_ms for f in items)
    n = len(items)
    return FrameStats(
        n_frames=n,
        baseline_rolling_fps_mean=float(sum(f.baseline_fps for f in items) / n),
        opt_rolling_fps_mean=float(sum(f.opt_fps for f in items) / n),
        # stage times are milliseconds, so n / (sum_ms / 1000) is FPS
        baseline_run_fps=float(1000.0 * n / base_total),
        opt_run_fps=float(1000.0 * n / opt_total),
        baseline_total_mean_ms=float(base_total / n),
        opt_total_mean_ms=float(opt_total / n),
    )


def preprocess_scopes(data: PPTData, *, src: str = "1920x1080") -> dict[str, BenchRow]:
    """Preprocess rows for one source resolution, keyed by scope/variant.

    The CSV stores the source resolution inside ``extra`` as ``src=...``.  If a
    requested resolution has no rows, all rows are returned with a warning and
    the caller is expected to say so on the image.
    """
    rows = [r for r in data.rows("preprocess.csv", "preprocess")]
    scoped = [r for r in rows if r.extra.get("src") == src]
    chosen = scoped or rows
    if not scoped:
        warn(f"preprocess.csv: no src={src} rows; using rows from the available source(s)")
    out: dict[str, BenchRow] = {}
    for row in chosen:
        if row.variant in out:
            warn(f"preprocess.csv: duplicate scope {row.variant}; keeping the first row")
            continue
        out[row.variant] = row
    return out


def inference_ladder(data: PPTData) -> tuple[dict[str, dict[int, BenchRow]], list[int]]:
    """Backend x batch -> row.  Batch comes from ``extra`` / ``size``, never guessed."""
    out: dict[str, dict[int, BenchRow]] = {}
    batches: set[int] = set()
    for row in data.rows("infer.csv", "infer"):
        backend = row.variant
        batch = row.batch
        if batch is None:
            warn(f"infer.csv: no batch recorded for {backend}; that row is skipped")
            continue
        out.setdefault(backend, {})[batch] = row
        batches.add(batch)
    return out, sorted(batches)


def kernels(data: PPTData) -> dict[str, Any]:
    """Largest-size reduction times and largest-N matmul throughput.

    Values are copied from the CSV columns; nothing is recomputed from FLOPs.
    """
    reduction_rows = data.rows("reduction.csv", "reduction")
    reduction: dict[str, BenchRow] = {}
    if reduction_rows:
        largest = max(int(r.size or 0) for r in reduction_rows)
        for name in ("naive_global", "shared_mem", "warp_shuffle", "cub_reference"):
            matches = [r for r in reduction_rows if r.variant == name and r.size == largest]
            if matches:
                reduction[name] = matches[0]
            else:
                warn(f"reduction.csv: no {name} row at size {largest}")

    matmul_rows = data.rows("matmul.csv", "matmul")
    matmul: dict[str, BenchRow] = {}
    matmul_n = 0
    if matmul_rows:
        matmul_n = max(int(r.size or 0) for r in matmul_rows)
        for name in ("naive", "tiled", "cublas"):
            matches = [r for r in matmul_rows if r.variant == name and r.size == matmul_n]
            if matches:
                matmul[name] = matches[0]
            else:
                warn(f"matmul.csv: no {name} row at N={matmul_n}")
    tiled_pct: float | None = None
    if "tiled" in matmul and "cublas" in matmul:
        cublas_gflops = matmul["cublas"].throughput
        if cublas_gflops > 0:
            tiled_pct = 100.0 * matmul["tiled"].throughput / cublas_gflops

    return {
        "reduction": reduction,
        "reduction_size": max((int(r.size or 0) for r in reduction_rows), default=0),
        "matmul": matmul,
        "matmul_n": matmul_n,
        "tiled_pct_of_cublas": tiled_pct,
    }


def streams(data: PPTData) -> dict[str, BenchRow]:
    out: dict[str, BenchRow] = {}
    for row in data.rows("streams.csv", "streams"):
        out[row.variant] = row
    return out


def multi_feed(data: PPTData, feeds: Iterable[int] = (1, 2, 4)) -> dict[int, dict[str, BenchRow]]:
    """``fps_per_feed`` rows grouped by feed count, read from ``extra``."""
    out: dict[int, dict[str, BenchRow]] = {}
    wanted = set(feeds)
    for row in data.rows("batching.csv", "batching"):
        feed = row.size
        if feed not in wanted:
            continue
        if row.variant not in {"serial_batch", "pipelined_batch"}:
            continue
        if "fps_per_feed" not in row.extra:
            warn(f"batching.csv: no fps_per_feed for {row.variant} at {feed} feeds")
            continue
        out.setdefault(feed, {})[row.variant] = row
    return out


def vector_add(data: PPTData) -> dict[str, BenchRow]:
    out: dict[str, BenchRow] = {}
    for row in data.rows("vector_add.csv", "vector_add"):
        out[row.variant] = row
    return out


def ncu_float4(data: PPTData) -> NcuResult | None:
    """DRAM / SM throughput of ``vector_add_float4``, one pair per ncu pass."""
    wanted = ("DRAM Throughput", "Compute (SM) Throughput", "Duration")
    per_pass: dict[str, dict[str, float]] = {}
    kernel_name = ""
    for row in data.ncu_rows:
        kernel = str(row.get("Kernel Name") or "").strip()
        if "float4" not in kernel.lower():
            continue
        metric = str(row.get("Metric Name") or "").strip()
        if metric not in wanted:
            continue
        raw = str(row.get("Metric Value") or "").replace(",", "").strip()
        try:
            value = float(raw)
        except ValueError:
            warn(f"profiling_ncu.csv: unparsable {metric} value {raw!r}; row skipped")
            continue
        if not kernel_name:
            kernel_name = kernel.split("(")[0].strip() or kernel
        ident = str(row.get("ID") or "").strip()
        per_pass.setdefault(ident, {})[metric] = value

    passes: list[NcuPass] = []
    for ident, values in sorted(per_pass.items()):
        if not all(m in values for m in wanted[:2]):
            continue
        passes.append(
            NcuPass(
                ident=ident,
                dram_pct=values["DRAM Throughput"],
                sm_pct=values["Compute (SM) Throughput"],
                duration_ns=values.get("Duration"),
            )
        )
    if not passes:
        warn("profiling_ncu.csv: no complete float4 DRAM/SM throughput pair")
        return None
    return NcuResult(kernel=kernel_name, passes=tuple(passes))


def _find_metric(rows: list[dict[str, str]], kind: str) -> tuple[float, str] | None:
    """Find a numeric metric in a possibly long-form CSV.

    ``kind`` is ``'map50'``, ``'map50_95'`` or ``'agreement'``.  Keys are
    normalised to lowercase alphanumeric so YOLO-style names such as
    ``metrics/mAP50-95(B)`` are recognised.
    """
    # finetune.py stores metrics as key=value text inside `extra`, not as
    # dedicated columns. Expand those pairs into virtual one-column rows so
    # both long-form CSVs and append_row()'s schema are understood.
    expanded: list[dict[str, str]] = []
    for row in rows:
        expanded.append(row)
        for raw_value in row.values():
            for key, number in re.findall(
                r"([A-Za-z0-9_/\-]+)=([-+]?\d+(?:\.\d+)?)", str(raw_value)
            ):
                expanded.append({"metric": key, "value": number})

    for row in expanded:
        named = None
        value_keys: Iterable[str] = row.keys()
        name_raw = str(row.get("metric") or row.get("name") or row.get("key") or "").strip()
        if name_raw:
            named = re.sub(r"[^a-z0-9]", "", name_raw.lower())
            value_keys = [k for k in row.keys() if k not in {"metric", "name", "key"}]
        for key in value_keys:
            norm = re.sub(r"[^a-z0-9]", "", str(key).lower())
            label = named or norm
            hit = False
            if kind == "map50":
                hit = "map50" in label and "95" not in label
            elif kind == "map50_95":
                hit = ("map50" in label and "95" in label) or "map5095" in label
            else:
                hit = "agreement" in label
            if not hit:
                continue
            raw = str(row.get(key) or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if kind.startswith("map") and not 0 < value <= 100:
                continue  # a zero / negative is not a measured mAP
            return value, str(key)
    return None


def finetune(data: PPTData) -> FinetuneResult | None:
    """Optional fine-tune metrics; ``None`` unless the file has a real mAP."""
    path = RESULTS / "finetune.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    found50 = _find_metric(rows, "map50")
    if found50 is None:
        warn("finetune.csv: present but no numeric mAP50 found; treating as NOT RUN")
        return None
    found5095 = _find_metric(rows, "map50_95")
    return FinetuneResult(
        map50=found50[0],
        map50_name=found50[1],
        map50_95=found5095[0] if found5095 else None,
        map50_95_name=found5095[1] if found5095 else None,
        source="finetune.csv",
    )


def agreement(data: PPTData) -> AgreementResult | None:
    """Optional baseline-vs-optimized detection agreement score.

    The real file is per-frame (``frame, baseline_count, opt_count,
    count_diff, matched, mean_iou, agrees``).  The score is the share of frames
    where ``agrees`` is true, computed from the rows -- never taken from an
    unrelated numeric column such as ``matched``.
    """
    path = RESULTS / "pipeline_agreement.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        warn("pipeline_agreement.csv: header only; no agreement score available")
        return None

    if any("agrees" in row for row in rows):
        flags = [
            str(row.get("agrees") or "").strip().lower() in {"true", "1", "yes"}
            for row in rows
        ]
        n_agree = int(sum(flags))
        ious: list[float] = []
        for row in rows:
            try:
                ious.append(float(str(row.get("mean_iou") or "").strip()))
            except ValueError:
                continue
        return AgreementResult(
            score=float(n_agree / len(rows)),
            name="frames agree",
            source="pipeline_agreement.csv",
            n_frames=len(rows),
            n_agree=n_agree,
            mean_iou=float(sum(ious) / len(ious)) if ious else None,
        )

    found = _find_metric(rows, "agreement")
    if found is None:
        warn("pipeline_agreement.csv: present but no agreement score found; NOT RUN")
        return None
    return AgreementResult(
        score=found[0], name=found[1], source="pipeline_agreement.csv",
        n_frames=len(rows), n_agree=0, mean_iou=None,
    )


def dataset_info(data: PPTData) -> dict[str, Any]:
    """Subset counts from dataset.csv -- not from prose documentation."""
    rows = data.rows("dataset.csv")
    if not rows:
        warn("dataset.csv: no dataset row")
        return {}
    extra = rows[0].extra
    out: dict[str, Any] = {"size": int(rows[0].size or 0), "extra": extra}
    for key in ("train", "val", "boxes", "cardboard", "metal", "rigid_plastic", "soft_plastic"):
        raw = extra.get(key)
        if raw is not None:
            try:
                out[key] = int(float(raw))
            except ValueError:
                warn(f"dataset.csv: invalid {key}={raw!r}")
    return out


def yolo_classes(path: Path | None = None) -> dict[int, str]:
    """Parse the ``names:`` map from data/waste.yaml without a YAML dependency."""
    path = path or (ROOT / "data" / "waste.yaml")
    if not path.is_file():
        warn(f"{path.name}: absent; class names cannot be shown")
        return {}
    names: dict[int, str] = {}
    inside = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^names\s*:", line.strip()):
            inside = True
            continue
        if inside and line.strip() and not line[0].isspace():
            break
        if not inside:
            continue
        match = re.match(r"\s*(\d+)\s*:\s*['\"]?([^'\"#]+?)['\"]?\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    return names


def val_images() -> list[Path]:
    folder = ROOT / "data" / "waste" / "images" / "val"
    if not folder.is_dir():
        warn("data/waste/images/val: absent; detection-quality sample frames unavailable")
        return []
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def val_boxes(path: Path) -> tuple[tuple[float, float, float, float, int], ...]:
    """Ground-truth boxes from data/waste/labels/val.

    The ZeroWaste export writes YOLO *polygons* (``cls x1 y1 x2 y2 ...``),
    not the 5-value bbox form.  Both forms are accepted: polygons become the
    axis-aligned bounding box of their normalised vertices.
    """
    label = ROOT / "data" / "waste" / "labels" / "val" / f"{path.stem}.txt"
    if not label.is_file():
        return ()
    boxes: list[tuple[float, float, float, float, int]] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            warn(f"{label.name}: malformed label line skipped")
            continue
        if len(values) == 4:  # cx, cy, w, h
            cx, cy, bw, bh = values
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        elif len(values) >= 6 and len(values) % 2 == 0:  # polygon
            xs, ys = values[0::2], values[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        else:
            warn(f"{label.name}: unsupported label format, line skipped")
            continue
        boxes.append((x1, y1, x2, y2, cls))
    return tuple(boxes)


def load_all() -> PPTData:
    """Load every required input, raising a named error on the first missing file."""
    data = PPTData()
    data.frames = load_frame_timings()
    for filename in REQUIRED_CSV:
        if filename == "demo_frame_timings.csv":
            data.standard[filename] = []  # parsed into data.frames above
            continue
        if filename == "profiling_ncu.csv":
            data.ncu_rows = read_ncu()
            continue
        data.standard[filename] = load_bench_csv(filename)
    data.device = load_device()
    data.finetune = finetune(data)
    data.agreement = agreement(data)
    for name in OPTIONAL_CSV:
        if not (RESULTS / name).is_file():
            data.notes.append(f"{name} absent (NOT RUN)")
    return data


__all__ = [
    "PPTData", "FrameTiming", "FrameStats", "BenchRow", "Box", "DeviceInfo",
    "NcuResult", "NcuPass", "FinetuneResult", "AgreementResult",
    "MissingDataError", "load_all", "per_frame_stats", "preprocess_scopes",
    "inference_ladder", "kernels", "streams", "multi_feed", "vector_add",
    "ncu_float4", "finetune", "agreement", "dataset_info", "yolo_classes",
    "val_images", "val_boxes", "parse_extra", "warn", "RESULTS", "ROOT",
]
