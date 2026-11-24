"""Shared helpers for every Phase B benchmark.

Same job as include/bench_common.h on the C++ side: one timing methodology, one
CSV schema, no hand-typed numbers anywhere.

Schema (must stay identical to csv_header() in include/bench_common.h):

    bench,variant,size,size_unit,time_ms_mean,time_ms_std,
    throughput,throughput_unit,extra,device,arch

Every function that measures appends a row here. results/summary.md is generated
from these rows, so a result that is not in a CSV does not exist.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

CSV_COLUMNS: Sequence[str] = (
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

# Repo root is one level above phase_b/.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def stats_of(samples: Iterable[float]) -> dict[str, float]:
    """mean / sample-stddev / min / max / n of a list of ms timings."""
    data = [float(x) for x in samples]
    if not data:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data),
        "n": len(data),
    }


# ---------------------------------------------------------------------------
# GPU timing with CUDA events (mirrors CudaTimer / time_iters)
# ---------------------------------------------------------------------------
@contextmanager
def gpu_timer():
    """Times whatever GPU work the block enqueues, in milliseconds.

    torch.cuda.Event timestamps are written by the GPU, so this measures device
    time rather than Python/launch overhead.
    """
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    box: dict[str, float] = {}
    yield box
    end.record()
    end.synchronize()
    box["ms"] = start.elapsed_time(end)


def time_call(fn: Callable[[], Any], warmup: int = 10, iters: int = 50) -> dict[str, float]:
    """Warms up, then times `fn` once per iteration with events."""
    import torch

    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        with gpu_timer() as box:
            fn()
        samples.append(box["ms"])
    return stats_of(samples)


def time_wall(fn: Callable[[], Any], warmup: int = 3, iters: int = 20) -> dict[str, float]:
    """Wall-clock timing for CPU (and end-to-end) work, in milliseconds.

    Used for the CPU baselines the GPU is compared against. On Colab the vCPUs
    are shared, so these numbers are noisy -- report them with their stddev and
    say so in the write-up.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return stats_of(samples)


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------
def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible. In Colab: Runtime > Change runtime type > "
            "Hardware accelerator = T4 GPU, then reconnect."
        )


def device_info() -> dict[str, Any]:
    """Machine identity recorded next to every result."""
    import torch

    require_cuda()
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "arch": f"sm_{props.major}{props.minor}",
        "multiprocessors": props.multi_processor_count,
        "global_mem_gib": round(props.total_memory / (1024**3), 2),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def write_device_info() -> dict[str, Any]:
    info = device_info()
    (results_dir() / "device.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


def arch() -> str:
    import torch

    props = torch.cuda.get_device_properties(0)
    return f"sm_{props.major}{props.minor}"


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------
# Set by the runner after the fine-tune stage. Every Python-written benchmark
# row records which weights were used, so a pretrained result can never be
# presented as a fine-tuned one. C++ kernel benchmarks do not use a model and
# therefore do not carry this field.
# Benchmarks run as separate processes, so the runner cannot set this in their
# memory. It exports FINETUNED=true after a successful fine-tune and every
# child process picks it up here, keeping CSV labels honest.
_FINETUNED = os.environ.get("FINETUNED", "false").lower() == "true"


def set_finetuned(value: bool) -> None:
    global _FINETUNED
    _FINETUNED = bool(value)


def append_row(
    bench: str,
    variant: str,
    size: float | int | str,
    size_unit: str,
    time_ms_mean: float,
    time_ms_std: float,
    throughput: float,
    throughput_unit: str,
    extra: str = "",
    filename: str | None = None,
) -> Path:
    """Append one benchmark row. Returns the file it was written to."""
    path = results_dir() / (filename or f"{bench}.csv")
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(
            [
                bench,
                variant,
                f"{size:.0f}" if isinstance(size, (int, float)) else size,
                size_unit,
                f"{time_ms_mean:.6f}",
                f"{time_ms_std:.6f}",
                f"{throughput:.4f}",
                throughput_unit,
                f"{extra},finetuned={str(_FINETUNED).lower()}" if extra else
                f"finetuned={str(_FINETUNED).lower()}",
                device_name(),
                arch(),
            ]
        )
    return path


def append_stats(
    bench: str,
    variant: str,
    size: float | int | str,
    size_unit: str,
    stats: dict[str, float],
    throughput: float,
    throughput_unit: str,
    extra: str = "",
    filename: str | None = None,
) -> Path:
    return append_row(
        bench=bench,
        variant=variant,
        size=size,
        size_unit=size_unit,
        time_ms_mean=stats["mean"],
        time_ms_std=stats["std"],
        throughput=throughput,
        throughput_unit=throughput_unit,
        extra=extra,
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def human(stats: dict[str, float], unit: str = "ms") -> str:
    return f"{stats['mean']:.3f} +- {stats['std']:.3f} {unit} (n={stats['n']})"


def gbs(nbytes: float, ms: float) -> float:
    return (nbytes / 1e9) / (ms / 1e3) if ms > 0 else 0.0


def gflops(flops: float, ms: float) -> float:
    return (flops / 1e9) / (ms / 1e3) if ms > 0 else 0.0


def fps(ms: float) -> float:
    return 1000.0 / ms if ms > 0 else 0.0
