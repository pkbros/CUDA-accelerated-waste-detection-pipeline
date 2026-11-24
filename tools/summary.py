"""Generate results/summary.md from the benchmark CSVs.

The only rule that matters here: this script never invents a number. It reads
what the benchmarks actually wrote and formats it. If a benchmark was never run,
it is reported as NOT RUN rather than omitted or filled in.

    python tools/summary.py            # write results/summary.md
    python tools/summary.py --check     # exit 1 if any expected benchmark is missing

Speedups are always computed from measured means: baseline_mean / variant_mean.
A speedup of "n/a" means the baseline row for that benchmark is missing, not that
the variant was slow.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

COLUMNS = [
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
]

# Which row in each CSV is the thing we are comparing against.
# Matches the baseline each benchmark was designed around.
BASELINES = {
    "vector_add": "cpu_reference",
    "reduction": "cpu_reference",
    "matmul": "cpu_naive",
    "streams": "sync_pageable",
    "preprocess": "opencv_cpu",
    "infer": "pt_fp32",
    "pipeline": "serial",
    "batching": "serial_batch",
    "finetune": None,
    "dataset": None,
}

# Benchmarks the project promises. Missing ones are reported, never skipped.
EXPECTED = [
    "vector_add",
    "reduction",
    "matmul",
    "streams",
    "preprocess",
    "infer",
    "pipeline",
    "batching",
]

TITLES = {
    "vector_add": "1. Vector add (sanity check + bandwidth)",
    "reduction": "2. Parallel reduction (naive / shared / warp shuffle)",
    "matmul": "3. Matrix multiply (naive / tiled / cuBLAS)",
    "streams": "4. Streams and pinned memory (copy/kernel overlap)",
    "preprocess": "5. Fused preprocess kernel vs CPU",
    "infer": "6. Inference ladder",
    "pipeline": "7. Serial vs pipelined throughput",
    "batching": "8. Batching / multiple camera feeds",
    "finetune": "9. Fine-tune",
    "dataset": "Data",
}

NOTES = {
    "vector_add": (
        "Memory-bound: 1 add per 12 bytes moved, so the ceiling is HBM bandwidth. "
        "Both GPU variants sit at ~250-260 GB/s. Measured here, float4 is slightly "
        "SLOWER than scalar: at the bandwidth ceiling 128-bit loads buy nothing, "
        "which is itself the result."
    ),
    "reduction": (
        "The interesting axis is the number of global-memory round trips and "
        "__syncthreads() calls removed: global -> shared -> warp shuffle."
    ),
    "matmul": (
        "GFLOP/s is the headline. cuBLAS is the honest ceiling; the gap between "
        "'tiled' and cuBLAS is register blocking and double buffering, which this "
        "repo deliberately does not implement."
    ),
    "streams": (
        "Pinning alone helps; overlapping needs both pinned buffers and more than one "
        "stream. Only 4 chunks, so the tail cannot be hidden completely."
    ),
    "preprocess": (
        "Compare against OpenCV on the CPU as the realistic baseline. The GPU column is "
        "one kernel doing five operations, so the win is mostly avoided traffic and "
        "launch overhead."
    ),
    "infer": (
        "Latency per image at the stated batch size. Despite T4 FP16 tensor cores, "
        "the measured FP16 rung is SLOWER than FP32 at batch 1; batch 1 is "
        "launch/latency-bound and cannot feed the tensor cores. Reported as measured."
    ),
    "pipeline": (
        "End-to-end throughput, wall clock, including Python overhead. This is the "
        "number that matters for a real camera loop, and it is lower than the "
        "1/latency of any single stage."
    ),
    "batching": (
        "FPS per feed is the number to read: total FPS divided by feeds. Serial scaling "
        "holds (~106 FPS/feed at 1/2/4 feeds); the pipelined loop REGRESSES at 2 and 4 "
        "feeds, where buffer bookkeeping costs more than the copy stall it removes."
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh)]


def load_all() -> dict[str, list[dict[str, str]]]:
    by_bench: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not RESULTS.exists():
        return by_bench
    for path in sorted(RESULTS.glob("*.csv")):
        # Nsight writes its own CSVs into results/. ncu instruments kernels and
        # inflates their durations by orders of magnitude (a 1M-element vector
        # add measured 1674 ms under ncu vs ~0.04 ms normally), and nsys reuses
        # results/profiling_tmp.csv across runs. Neither is a benchmark result.
        # They stay on disk as evidence but must never enter a summary table.
        if path.name.startswith("profiling_"):
            continue
        try:
            rows = read_rows(path)
        except (OSError, csv.Error) as exc:  # a half-written CSV must not crash the report
            print(f"warning: could not read {path.name}: {exc}", file=sys.stderr)
            continue
        for row in rows:
            if row.get("bench"):
                by_bench[row["bench"]].append(row)
    return by_bench


def device_line() -> str:
    """Whose GPU these numbers came from. Never assume; read what was recorded."""
    json_path = RESULTS / "device.json"
    if json_path.exists():
        try:
            info = json.loads(json_path.read_text(encoding="utf-8"))
            return (
                f"{info.get('gpu_name', 'unknown GPU')} "
                f"({info.get('arch', '?')}, {info.get('global_mem_gib', '?')} GiB, "
                f"torch {info.get('torch', '?')}, CUDA {info.get('cuda_runtime', '?')})"
            )
        except (json.JSONDecodeError, OSError):
            pass
    txt_path = RESULTS / "device.txt"
    if txt_path.exists():
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("gpu_name="):
                return line.split("=", 1)[1]
    return "unknown GPU (no device record found)"


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) < 0.01:
        return f"{value:.4g}"
    return f"{value:.{digits}f}"


def table_for(bench: str, rows: list[dict[str, str]]) -> list[str]:
    baseline_name = BASELINES.get(bench)
    baseline_mean = None
    for row in rows:
        if baseline_name and row["variant"] == baseline_name:
            baseline_mean = as_float(row["time_ms_mean"])
            break

    # Slower rows first so the baseline anchors the table; then by throughput.
    rows = sorted(rows, key=lambda r: as_float(r["throughput"]), reverse=False)

    has_extra = any(r.get("extra") for r in rows)
    lines = [
        "| variant | size | mean ms | std ms | throughput | speedup vs "
        f"{baseline_name or '-'} |" + (" notes |" if has_extra else ""),
        "|---|---|---|---|---|---|" + ("---|" if has_extra else ""),
    ]
    for row in rows:
        mean = as_float(row["time_ms_mean"])
        std = as_float(row["time_ms_std"])
        tput = as_float(row["throughput"])
        unit = row.get("throughput_unit") or ""
        size = f"{row.get('size', '')} {row.get('size_unit', '')}".strip()
        speedup = "n/a"
        if baseline_mean and mean > 0:
            speedup = f"{baseline_mean / mean:.2f}x"
        line = (
            f"| `{row['variant']}` | {size} | {mean:.3f} | {std:.3f} | "
            f"{fmt(tput)} {unit} | {speedup} |"
        )
        if has_extra:
            extra = (row.get("extra") or "").replace("|", "/")
            line += f" {extra} |"
        lines.append(line)
    return lines


def render(by_bench: dict[str, list[dict[str, str]]]) -> str:
    out: list[str] = []
    out.append("# Benchmark summary")
    out.append("")
    out.append("Generated by `tools/summary.py` from the CSV files in `results/`.")
    out.append("Every number below was written by a benchmark that actually ran; rows")
    out.append("that do not exist are reported as **NOT RUN** instead of being filled in.")
    out.append("")
    out.append(f"**Device:** {device_line()}")
    out.append("")

    gpus = {row.get("device") for rows in by_bench.values() for row in rows if row.get("device")}
    if len(gpus) > 1:
        out.append(
            "> Warning: these CSVs mix results from more than one GPU "
            f"({', '.join(sorted(g for g in gpus if g))}). Do not compare rows across them."
        )
        out.append("")

    missing = [b for b in EXPECTED if not by_bench.get(b)]
    if missing:
        out.append("## Not run yet")
        out.append("")
        out.append(
            "The following benchmarks have no rows in `results/`: "
            + ", ".join(f"`{m}`" for m in missing)
            + ". They are listed here so a missing result can never be mistaken for a"
            " passing one."
        )
        out.append("")

    for bench in sorted(by_bench, key=lambda b: (EXPECTED.index(b) if b in EXPECTED else 99, b)):
        rows = by_bench[bench]
        out.append(f"## {TITLES.get(bench, bench)}")
        out.append("")
        if bench in NOTES:
            out.append(NOTES[bench])
            out.append("")
        out.extend(table_for(bench, rows))
        out.append("")

    out.append("## Reproduce")
    out.append("")
    out.append("```")
    out.append("make -C phase_a ARCH=sm_75 run")
    out.append("python phase_b/preprocess_ext.py")
    out.append("python phase_b/infer_bench.py")
    out.append("python phase_b/pipeline.py")
    out.append("python tools/summary.py")
    out.append("```")
    out.append("")
    out.append("Or run `notebooks/run_all.ipynb` top to bottom.")
    out.append("")
    return "\n".join(out)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a benchmark is missing")
    parser.add_argument("--out", default=str(RESULTS / "summary.md"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    by_bench = load_all()
    text = render(by_bench)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")

    missing = [b for b in EXPECTED if not by_bench.get(b)]
    if missing:
        print(f"missing benchmarks: {', '.join(missing)}", file=sys.stderr)
    if args.check and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
