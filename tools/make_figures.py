"""Generate the three documentation figures from the files in results/.

Run from the repo root:

    python tools/make_figures.py

No number in any figure is typed in by hand. Every plotted value (and the GPU
name in the footer of every figure) is read at run time from results/*.csv or
results/device.json. If a value cannot be found, a clear warning is printed
and that panel is omitted rather than filled in with a guess.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
PLOTS = DOCS / "plots"

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

# The rewrite of the preprocess bench is complete but has not been re-measured.
# Only a CSV that contains ALL of these scope rows counts as the fair result.
FAIR_SCOPES = ("kernel_only", "h2d_plus_kernel", "end_to_end", "opencv_cpu", "torch_gpu_ops")

OUTPUTS = (
    "docs/overview.png",
    "docs/plots/nsys_timeline.png",
    "docs/plots/negative_results.png",
)

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# --------------------------------------------------------------------------- #
# helpers: loading and validation
# --------------------------------------------------------------------------- #


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def read_standard(filename: str) -> list[dict[str, str]]:
    """Read one of the standard-schema benchmark CSVs."""
    path = RESULTS / filename
    if not path.is_file():
        warn(f"{filename}: file not found; panels using it will be omitted")
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in STANDARD_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                warn(f"{filename}: missing columns {missing}")
            return [row for row in reader if any(str(v or "").strip() for v in row.values())]
    except (OSError, csv.Error) as exc:
        warn(f"{filename}: could not read ({exc})")
        return []


def load_gpu_footer() -> str:
    """The device line that goes on every figure, straight from device.json."""
    path = RESULTS / "device.json"
    if not path.is_file():
        warn("results/device.json not found; footer will say the GPU is unknown")
        return "Data: results/*.csv  |  GPU model unknown (results/device.json missing)"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"results/device.json could not be parsed ({exc}); footer will say GPU unknown")
        return "Data: results/*.csv  |  GPU model unknown (results/device.json unreadable)"
    gpu = (info.get("gpu_name") or "").strip()
    arch = (info.get("arch") or "").strip()
    if not gpu:
        warn("results/device.json has no gpu_name; footer will say the GPU is unknown")
        gpu = "GPU model unknown (not recorded in device.json)"
    suffix = f" ({arch})" if arch else ""
    return f"Data: results/*.csv  |  {gpu}{suffix}"


def num(row: dict[str, str], key: str, label: str) -> float | None:
    """A single required number. None means 'missing' -> caller omits the panel."""
    raw = (row.get(key) or "").strip()
    if not raw:
        warn(f"{label}: missing {key}")
        return None
    try:
        return float(raw)
    except ValueError:
        warn(f"{label}: invalid {key}={raw!r}")
        return None


def extra_fields(row: dict[str, str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in (row.get("extra") or "").split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def ncu_float(raw: object) -> float | None:
    """ncu writes numbers with thousands separators, e.g. '48,032'."""
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def read_ncu() -> list[dict[str, str]]:
    """Read ncu's long-form CSV, skipping its ==PROF== preamble lines."""
    path = RESULTS / "profiling_ncu.csv"
    if not path.is_file():
        warn("profiling_ncu.csv not found; ncu panels will be omitted")
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        warn(f"profiling_ncu.csv could not be read ({exc})")
        return []
    header = next(
        (
            i
            for i, line in enumerate(lines)
            if "Kernel Name" in line and "Metric Name" in line and "Metric Value" in line
        ),
        None,
    )
    if header is None:
        warn("profiling_ncu.csv: no header with Kernel Name / Metric Name / Metric Value")
        return []
    try:
        return [r for r in csv.DictReader(lines[header:]) if (r.get("Metric Name") or "").strip()]
    except csv.Error as exc:
        warn(f"profiling_ncu.csv: parse error ({exc})")
        return []


def ncu_sol(rows: list[dict[str, str]]) -> tuple[np.ndarray, str, int] | None:
    """DRAM and SM throughput for the float4 kernel, one sample per ncu pass."""
    per_pass: dict[str, dict[str, float]] = {}
    short_name = ""
    wanted = ("DRAM Throughput", "Compute (SM) Throughput")
    for row in rows:
        kernel = (row.get("Kernel Name") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        if "float4" not in kernel.lower() or metric not in wanted:
            continue
        value = ncu_float(row.get("Metric Value"))
        if value is None:
            continue
        if not short_name:
            short_name = kernel.split("(")[0].strip() or kernel
        per_pass.setdefault(str(row.get("ID")), {})[metric] = value
    pairs = [
        (d["DRAM Throughput"], d["Compute (SM) Throughput"])
        for d in per_pass.values()
        if all(m in d for m in wanted)
    ]
    if not pairs:
        warn(
            "profiling_ncu.csv: no complete DRAM Throughput / Compute (SM) Throughput "
            "pair for a float4 kernel"
        )
        return None
    return np.asarray(pairs, dtype=float), short_name, len(pairs)


def ncu_durations(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    """Kernel durations in ns, grouped by kernel name."""
    out: dict[str, list[float]] = {}
    for row in rows:
        if (row.get("Metric Name") or "").strip() != "Duration":
            continue
        unit = (row.get("Metric Unit") or "").strip().lower()
        if unit and unit != "ns":
            warn(f"profiling_ncu.csv: unsupported Duration unit {unit!r}; skipping row")
            continue
        value = ncu_float(row.get("Metric Value"))
        if value is None:
            continue
        kernel = (row.get("Kernel Name") or "").strip()
        short = kernel.split("(")[0].strip() or kernel
        out.setdefault(short, []).append(value)
    return out


# --------------------------------------------------------------------------- #
# helpers: drawing
# --------------------------------------------------------------------------- #


def fmt_value(value: float) -> str:
    a = abs(value)
    if a >= 1000:
        return f"{value:,.0f}"
    if a >= 100:
        return f"{value:.0f}"
    if a >= 1:
        return f"{value:.2f}"
    if a >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4g}"


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def draw_bars(
    ax: plt.Axes,
    labels: list[str],
    means: list[float],
    stds: list[float] | None,
    *,
    log: bool = False,
    title: str = "",
    ylabel: str = "",
    title_color: str = "#222222",
) -> None:
    x = np.arange(len(labels))
    yerr = np.asarray(stds) if stds is not None else None
    ax.bar(
        x,
        means,
        width=0.62,
        yerr=yerr,
        capsize=5,
        color=COLORS[: len(labels)],
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12 if "\n" in title else 14, color=title_color)
    style_axes(ax)
    lo, hi = min(means), max(means)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=lo * 0.35, top=hi * 3.5)
        label_y = [v * 1.18 for v in means]
    else:
        pad = max(hi - lo, hi * 0.1, abs(hi) * 0.1)
        ax.set_ylim(bottom=0, top=hi + pad)
        label_y = [v + pad * 0.08 for v in means]
    for xi, value, y in zip(x, means, label_y):
        ax.text(xi, y, fmt_value(value), ha="center", va="bottom", fontsize=10)


def add_footer(fig: plt.Figure, footer: str) -> None:
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10, color="#555555")


def save(fig: plt.Figure, relative: str) -> Path:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {relative} ({path.stat().st_size:,} bytes)")
    return path


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#444444",
            linestyle="--" if dashed else "-",
        )
    )


# --------------------------------------------------------------------------- #
# overview.png: four panels
# --------------------------------------------------------------------------- #


def panel_architecture(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(A) Pipeline architecture - two streams", fontsize=14)

    stages = [
        (0.030, "VIDEO\nFRAMES", "#d9e8f5"),
        (0.270, "FUSED CUDA\nPREPROCESS", "#c9e6d2"),
        (0.510, "YOLO\nINFERENCE", "#fde2c5"),
        (0.750, "POSTPROCESS\n(NMS)", "#e6d9f2"),
    ]
    y, h = 0.60, 0.21
    for x, text, color in stages:
        box(ax, x, y, 0.20, h, text, color)
    for (x1, _, _), (x2, _, _) in zip(stages, stages[1:]):
        arrow(ax, (x1 + 0.205, y + h / 2), (x2 - 0.012, y + h / 2))

    lanes = [
        (0.30, "STREAM A", "#eef4fb", 0.11, (0.30, 0.50, 0.70)),
        (0.10, "STREAM B", "#f3f8f1", 0.11, (0.33, 0.53, 0.73)),
    ]
    for ly, name, bg, bh, xs in lanes:
        ax.add_patch(
            FancyBboxPatch(
                (0.145, ly - 0.012),
                0.815,
                bh + 0.024,
                boxstyle="round,pad=0.004",
                transform=ax.transAxes,
                facecolor=bg,
                edgecolor="#bbbbbb",
                linewidth=0.8,
            )
        )
        ax.text(0.155, ly + bh / 2, name, transform=ax.transAxes, ha="left", va="center", fontsize=9.5, color="#444444")
        labels = ("H2D", "PREPROCESS", "YOLO")
        centers = []
        for x, text in zip(xs, labels):
            box(ax, x, ly, 0.14, bh, text, "#ffffff")
            centers.append(x)
        for x1, x2 in zip(centers, centers[1:]):
            arrow(ax, (x1 + 0.145, ly + bh / 2), (x2 - 0.010, ly + bh / 2))
    arrow(ax, (0.37, 0.59), (0.37, 0.43), dashed=True)
    arrow(ax, (0.61, 0.59), (0.77, 0.43), dashed=True)
    ax.text(
        0.5,
        0.025,
        "pinned buffers + overlapping streams hide host transfers behind GPU work",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color="#555555",
    )


def panel_preprocess(ax: plt.Axes, rows: list[dict[str, str]]) -> list[str] | None:
    rows = [r for r in rows if r.get("bench") == "preprocess"]
    if not rows:
        warn("preprocess.csv: no 'preprocess' rows; overview panel B omitted")
        return None

    found = {r.get("variant") for r in rows}
    missing = [scope for scope in FAIR_SCOPES if scope not in found]
    if missing:
        verdict = "SUPERSEDED - unfair timing scopes, awaiting T4 re-run"
        warn(
            "preprocess.csv lacks fair scope rows "
            f"{missing}; panel B is labeled '{verdict}'"
        )
        chosen = rows
        color = "#b22222"
    else:
        verdict = "current fair result"
        chosen = [r for r in rows if r.get("variant") in FAIR_SCOPES]
        color = "#222222"

    seen: set[str] = set()
    picks: list[tuple[str, float, float]] = []
    used: list[str] = []
    for row in chosen:
        variant = (row.get("variant") or "").strip()
        if not variant or variant in seen:
            continue
        seen.add(variant)
        mean = num(row, "time_ms_mean", f"preprocess/{variant}")
        std = num(row, "time_ms_std", f"preprocess/{variant}")
        if mean is None or std is None:
            warn("preprocess.csv: incomplete row; overview panel B omitted")
            return None
        picks.append((variant, mean, std))
        used.append(variant)
    if not picks:
        warn("preprocess.csv: no usable rows; overview panel B omitted")
        return None

    draw_bars(
        ax,
        [p[0] for p in picks],
        [p[1] for p in picks],
        [p[2] for p in picks],
        log=True,
        title=f"(B) preprocess\n{verdict}",
        ylabel="mean time (ms, log scale)",
        title_color=color,
    )
    return [f"preprocess.csv rows {used} [{verdict}]"]


def panel_reduction(ax: plt.Axes, rows: list[dict[str, str]]) -> list[str] | None:
    rows = [r for r in rows if r.get("bench") == "reduction"]
    wanted = (
        ("naive", ("naive_global", "naive")),
        ("shared_mem", ("shared_mem",)),
        ("warp_shuffle", ("warp_shuffle",)),
        ("cub_reference", ("cub_reference",)),
    )
    picks: list[tuple[str, float, float, str]] = []
    for label, candidates in wanted:
        matches = [r for r in rows if r.get("variant") in candidates]

        def size_of(row: dict[str, str]) -> float:
            try:
                return float(row.get("size") or 0)
            except ValueError:
                return 0.0

        if not matches:
            warn(f"reduction.csv: no row for {label}; that bar is omitted")
            continue
        row = max(matches, key=size_of)
        mean = num(row, "time_ms_mean", f"reduction/{label}")
        std = num(row, "time_ms_std", f"reduction/{label}")
        if mean is None or std is None:
            warn(f"reduction.csv: incomplete row for {label}; overview panel C omitted")
            return None
        picks.append((row.get("variant") or label, mean, std, label))
    if not picks:
        warn("reduction.csv: no reduction rows; overview panel C omitted")
        return None
    draw_bars(
        ax,
        [p[0] for p in picks],
        [p[1] for p in picks],
        [p[2] for p in picks],
        log=True,
        title="(C) reduction variants",
        ylabel="mean time (ms, log scale)",
    )
    return [f"reduction.csv rows {[p[0] for p in picks]}"]


def panel_ncu(ax: plt.Axes, ncu_rows: list[dict[str, str]]) -> list[str] | None:
    stats = ncu_sol(ncu_rows)
    if stats is None:
        return None
    pairs, short_name, count = stats
    means = [float(pairs[:, 0].mean()), float(pairs[:, 1].mean())]
    stds = [float(pairs[:, 0].std(ddof=1)), float(pairs[:, 1].std(ddof=1))] if count > 1 else None
    draw_bars(
        ax,
        ["DRAM throughput", "SM throughput"],
        means,
        stds,
        title=f"(D) ncu throughput - {short_name}",
        ylabel="percent of peak (%)",
    )
    ax.set_ylim(0, max(means) * 1.35)
    return [f"profiling_ncu.csv {short_name} throughput, {count} passes"]


def build_overview(footer: str, standard: dict[str, list[dict[str, str]]], ncu_rows: list[dict[str, str]], report: list[str]) -> None:
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.26, top=0.90, bottom=0.10)

    panel_architecture(fig.add_subplot(grid[0, 0]))
    report.append("overview (A): architecture diagram, no numeric data (footer from device.json)")

    for slot, panel in (
        (grid[0, 1], lambda ax: panel_preprocess(ax, standard.get("preprocess.csv", []))),
        (grid[1, 0], lambda ax: panel_reduction(ax, standard.get("reduction.csv", []))),
        (grid[1, 1], lambda ax: panel_ncu(ax, ncu_rows)),
    ):
        entry = panel(fig.add_subplot(slot))
        if entry:
            for line in entry:
                report.append("overview " + line)

    add_footer(fig, footer)
    save(fig, OUTPUTS[0])


# --------------------------------------------------------------------------- #
# nsys_timeline.png
# --------------------------------------------------------------------------- #


def build_timeline(footer: str, ncu_rows: list[dict[str, str]], report: list[str]) -> None:
    durations = ncu_durations(ncu_rows)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.14)

    if durations:
        kernels = sorted(durations)
        ticks: list[float] = []
        tick_labels: list[str] = []
        for index, name in enumerate(kernels):
            # ncu reports ns; convert to us for a readable axis.
            values = np.asarray(durations[name], dtype=float) / 1000.0
            starts = np.concatenate(([0.0], np.cumsum(values)[:-1]))
            ax.broken_barh(
                list(zip(starts, values)),
                (index - 0.3, 0.6),
                facecolors=COLORS[index % len(COLORS)],
                edgecolor="#333333",
                linewidth=0.5,
            )
            ticks.append(index)
            tick_labels.append(name)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("illustrative time (us) - kernel durations placed back-to-back")
        ax.set_title("Kernel timeline", fontsize=15)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        caption = (
            "Illustrative timeline - kernel durations rendered from nsys/ncu trace data "
            "(ncu Duration metric; sequential layout, not wall-clock start times)"
        )
        report.append(
            "nsys_timeline: profiling_ncu.csv Duration rows for "
            f"{kernels} ({sum(len(v) for v in durations.values())} launches)"
        )
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "AWAITING NSYS TRACE EXPORT",
            ha="center",
            va="center",
            fontsize=20,
            color="#888888",
            transform=ax.transAxes,
        )
        caption = ""
        warn("no ncu kernel durations found; timeline says AWAITING NSYS TRACE EXPORT")
        report.append("nsys_timeline: omitted, no kernel durations in profiling_ncu.csv")

    if caption:
        fig.text(0.5, 0.072, caption, ha="center", fontsize=11, color="#333333")
    add_footer(fig, footer)
    save(fig, OUTPUTS[1])


# --------------------------------------------------------------------------- #
# negative_results.png
# --------------------------------------------------------------------------- #


def negative_vector_add(ax: plt.Axes, rows: list[dict[str, str]]) -> list[str] | None:
    picks: list[tuple[str, float, float]] = []
    for variant in ("gpu_scalar", "gpu_float4"):
        matches = [r for r in rows if r.get("bench") == "vector_add" and r.get("variant") == variant]
        if not matches:
            warn(f"vector_add.csv: no {variant} row; negative-results panel (a) omitted")
            return None
        mean = num(matches[0], "time_ms_mean", f"vector_add/{variant}")
        std = num(matches[0], "time_ms_std", f"vector_add/{variant}")
        if mean is None or std is None:
            warn("vector_add.csv: incomplete row; negative-results panel (a) omitted")
            return None
        picks.append((variant, mean, std))
    scalar, float4 = picks[0][1], picks[1][1]
    verdict = "float4 slower" if float4 > scalar else "float4 faster" if float4 < scalar else "float4 equal"
    draw_bars(
        ax,
        [p[0] for p in picks],
        [p[1] for p in picks],
        [p[2] for p in picks],
        title=f"(a) vector_add - {verdict} (measured)",
        ylabel="mean time (ms)",
    )
    return [f"negative (a): vector_add.csv rows {[p[0] for p in picks]}"]


def negative_infer(ax: plt.Axes, rows: list[dict[str, str]]) -> list[str] | None:
    picks: list[tuple[str, float, float, str | None]] = []
    for variant in ("pt_fp32", "pt_fp16"):
        matches = [r for r in rows if r.get("bench") == "infer" and r.get("variant") == variant]
        if not matches:
            warn(f"infer.csv: no {variant} row; negative-results panel (b) omitted")
            return None
        row = matches[0]
        mean = num(row, "time_ms_mean", f"infer/{variant}")
        std = num(row, "time_ms_std", f"infer/{variant}")
        if mean is None or std is None:
            warn(f"infer.csv: incomplete {variant} row; negative-results panel (b) omitted")
            return None
        found = re.search(r"\bbatch\s*=\s*([0-9]+)", row.get("extra") or "")
        batch = found.group(1) if found else None
        if batch is None:
            warn(f"infer.csv: {variant} row has no batch= in extra")
        picks.append((variant, mean, std, batch))

    batches = {p[3] for p in picks if p[3] is not None}
    if len(batches) > 1:
        warn(f"infer.csv: rows mix batches {sorted(batches)}")
        batch_text = "batch differs between rows"
    elif batches:
        batch_text = f"batch {next(iter(batches))}"
    else:
        batch_text = "batch not recorded"
    draw_bars(
        ax,
        [p[0] for p in picks],
        [p[1] for p in picks],
        [p[2] for p in picks],
        title=f"(b) infer - FP32 vs FP16 ({batch_text})",
        ylabel="mean time (ms)",
    )
    return [f"negative (b): infer.csv rows {[p[0] for p in picks]} ({batch_text}, from extra)"]


def negative_batching(ax: plt.Axes, rows: list[dict[str, str]]) -> list[str] | None:
    variants = ("serial_batch", "pipelined_batch")
    data: dict[float, dict[str, float]] = {}
    unit = ""
    for row in rows:
        if row.get("bench") != "batching" or row.get("variant") not in variants:
            continue
        try:
            size = float(row.get("size") or "")
        except ValueError:
            warn(f"batching.csv: invalid size {row.get('size')!r}")
            return None
        raw_fps = extra_fields(row).get("fps_per_feed")
        if raw_fps is None:
            warn("batching.csv: no fps_per_feed in extra; negative-results panel (c) omitted")
            return None
        try:
            fps = float(raw_fps)
        except ValueError:
            warn(f"batching.csv: invalid fps_per_feed={raw_fps!r}; panel (c) omitted")
            return None
        data.setdefault(size, {})[row["variant"]] = fps
        unit = (row.get("size_unit") or "").strip()
    if not data:
        warn("batching.csv: no usable fps_per_feed rows; negative-results panel (c) omitted")
        return None
    sizes = sorted(data)
    for size in sizes:
        absent = [v for v in variants if v not in data[size]]
        if absent:
            warn(f"batching.csv: no fps_per_feed for {absent} at size {size}; panel (c) omitted")
            return None

    x = np.arange(len(sizes))
    width = 0.36
    for i, variant in enumerate(variants):
        values = [data[s][variant] for s in sizes]
        ax.bar(x + (i - 0.5) * width, values, width, label=variant, color=COLORS[i], edgecolor="#333333", linewidth=0.8)
        for xi, value in zip(x + (i - 0.5) * width, values):
            ax.text(xi, value * 1.02, fmt_value(value), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g} {unit}".strip() for s in sizes])
    ax.set_ylabel("FPS per feed")
    ax.set_title("(c) batching - FPS per feed", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(d.values()) for d in data.values()) * 1.22)
    style_axes(ax)
    return [f"negative (c): batching.csv fps_per_feed for {variants} at sizes {sizes} {unit}"]


def build_negative(footer: str, standard: dict[str, list[dict[str, str]]], report: list[str]) -> None:
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(1, 3, wspace=0.34, top=0.84, bottom=0.10)
    panels = (
        (grid[0, 0], lambda ax: negative_vector_add(ax, standard.get("vector_add.csv", []))),
        (grid[0, 1], lambda ax: negative_infer(ax, standard.get("infer.csv", []))),
        (grid[0, 2], lambda ax: negative_batching(ax, standard.get("batching.csv", []))),
    )
    built = 0
    for slot, panel in panels:
        entry = panel(fig.add_subplot(slot))
        if entry:
            built += 1
            report.extend(entry)
    if not built:
        fig.text(0.5, 0.5, "NO DATA - see warnings", ha="center", va="center", fontsize=18, color="#888888")
    add_footer(fig, footer)
    save(fig, OUTPUTS[2])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def validate() -> None:
    for relative in OUTPUTS:
        path = ROOT / relative
        if not path.is_file():
            warn(f"validation failed: {relative} was not created")
            continue
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        size = path.stat().st_size
        if size < 10_000:
            warn(f"validation: {relative} is suspiciously small ({size:,} bytes)")
        print(f"validated {relative}: {width}x{height}, {size:,} bytes")


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    footer = load_gpu_footer()
    standard = {name: read_standard(name) for name in ("preprocess.csv", "reduction.csv", "vector_add.csv", "infer.csv", "batching.csv")}
    ncu_rows = read_ncu()
    report: list[str] = []

    build_overview(footer, standard, ncu_rows, report)
    build_timeline(footer, ncu_rows, report)
    build_negative(footer, standard, report)
    validate()

    print("\nPanel provenance:")
    for line in report:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
