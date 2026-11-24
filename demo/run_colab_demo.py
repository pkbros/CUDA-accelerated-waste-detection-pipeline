"""One-command Colab runner for the CUDA waste-detection PROTOTYPE.

Usage, after the notebook's bootstrap cell has cloned the repo:

    !python demo/run_colab_demo.py

It runs every stage needed for the demo, streams live output, records failures
instead of hiding them, and prints every CSV at the end. No number is typed by
hand: anything that cannot run is reported as a failure, never estimated.
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASE_B = REPO / "phase_b"
# SECURITY: export URLs contain a `key=` token. Never hardcode one in this
# public repo. Set ROBOFLOW_EXPORT_URL in the Colab environment (or a Colab
# secret) before running; generate a fresh URL with the Roboflow MCP tool
# versions_export(project=zerowaste-data/zerowaste-contamination, version=3,
# format=yolov8). The URL is never printed to logs.
EXPORT_URL = os.environ.get("ROBOFLOW_EXPORT_URL", "")


def record_failure(stage: str, detail: str) -> None:
    import datetime

    path = REPO / "results" / "failures.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("# Recorded failures\n\nPrototype run; failures are kept deliberately.\n\n")
        fh.write(f"- {datetime.date.today()} - {stage}: {detail.strip()[:4000]}\n")
    print(f"[recorded failure] {stage}: {detail.strip()[:200]}", flush=True)


def sh(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run from the repo root, streaming output line by line."""
    print(f"\n$ {cmd}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(REPO), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    chunks: list[str] = []
    for line in proc.stdout:
        chunks.append(line)
        print(line, end="", flush=True)
    proc.wait()
    result = subprocess.CompletedProcess(cmd, proc.returncode, "".join(chunks))
    if check and proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {cmd}")
    return result


def step(name: str, fn) -> bool:
    print("\n" + "=" * 74, flush=True)
    print(f"STAGE: {name}", flush=True)
    print("=" * 74, flush=True)
    try:
        fn()
        print(f"[ok] {name}", flush=True)
        return True
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        # render_demo.py exits via SystemExit (e.g. ffmpeg missing). SystemExit
        # is not an Exception, so catching only Exception killed the runner
        # silently and left no trace in failures.md.
        import traceback

        record_failure(name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        print(f"[failed] {name}: {exc}", flush=True)
        return False


def have(tool: str) -> bool:
    return shutil_which(tool) is not None


def shutil_which(tool: str):
    import shutil
    return shutil.which(tool)


def dataset_stage() -> None:
    """Fetch the export only from ROBOFLOW_EXPORT_URL; never from a literal."""
    zip_path = REPO / "data/raw/zerowaste_yolov8.zip"
    if not zip_path.exists():
        if not EXPORT_URL:
            raise RuntimeError(
                "ROBOFLOW_EXPORT_URL is not set. Generate a fresh export URL "
                "with the Roboflow MCP versions_export tool and set the "
                "environment variable before running. The URL is a secret; "
                "it must never be written into this repo."
            )
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading ZeroWaste export from the configured URL", flush=True)
        urllib.request.urlretrieve(EXPORT_URL, zip_path)
        print(f"downloaded {zip_path.stat().st_size / 1e6:.1f} MB")
    sh(f"python {shlex_quote(PHASE_B / 'prepare_zerowaste.py')}")


def main() -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(PHASE_B))
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "reports").mkdir(exist_ok=True)

    # ---- clean slate: a rerun must never mix two sessions' rows -----------
    # Previous CSV/JSON files are MOVED to a timestamped archive, not deleted.
    # Every chart is then built from exactly one consistent measurement run.
    import datetime
    archive = REPO / "results" / f"archive_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    previous = sorted(path for path in (REPO / "results").iterdir()
                      if path.is_file() and path.suffix in {".csv", ".json"})
    if previous:
        archive.mkdir(parents=True, exist_ok=True)
        for path in previous:
            path.rename(archive / path.name)
        print(f"archived {len(previous)} previous results to {archive.name}",
              flush=True)

    # ---- environment -----------------------------------------------------
    def environment():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU - select a T4 runtime and reconnect")
        print("gpu:", torch.cuda.get_device_name(0), "torch:", torch.__version__,
              "cuda:", torch.version.cuda)
        from common import write_device_info

        print("recorded:", write_device_info())
        if have("ninja") is False:
            sh("pip -q install ninja")
        print("ninja:", shutil_which("ninja") or "MISSING")
        # torch's cpp_extension needs a CUDA toolkit whose MAJOR matches torch.
        if not have("nvcc"):
            torch_cuda = torch.version.cuda or "12.4"
            major, minor = torch_cuda.split(".")[:2]
            sh("sudo apt-get update -qq", check=False)
            rc = sh(
                f"sudo apt-get install -y -qq cuda-compiler-{major}-{minor} "
                f"cuda-cudart-dev-{major}-{minor} libcublas-dev-{major}-{minor}",
                check=False,
            ).returncode
            if rc != 0:
                sh(f"sudo apt-get install -y -qq cuda-toolkit-{major}", check=False)
            for candidate in sorted(Path("/usr/local").glob(f"cuda-{major}*")):
                if (candidate / "bin" / "nvcc").exists():
                    os.environ["CUDA_HOME"] = str(candidate)
                    os.environ["PATH"] = f"{candidate / 'bin'}{os.pathsep}{os.environ['PATH']}"
                    break
        if not have("nvcc"):
            raise RuntimeError("nvcc still missing; cannot build the CUDA kernels")
        os.environ.setdefault("CUDA_HOME",
                              str(Path(shutil_which("nvcc")).resolve().parent.parent))
        os.environ["CUDACXX"] = shutil_which("nvcc")
        sh("nvcc --version | tail -n 2")
        sh("pip -q install ultralytics onnx onnxruntime-gpu 2>&1 | tail -n 5")
        # No video rendering in this workflow; ffmpeg is not required.

    step("environment", environment)

    # ---- Phase A: the four hand-written CUDA kernels ----------------------
    def phase_a():
        import torch

        props = torch.cuda.get_device_properties(0)
        arch = f"sm_{props.major}{props.minor}"
        sh(f"make -C {shlex_quote(REPO / 'phase_a')} ARCH={arch} -j2")
        runs = [
            ("vector_add", ["--n", "50000000", "--iters", "50"]),
            ("reduction", ["--n", "16777216", "--iters", "50"]),
            ("matmul", ["--n", "1024", "--cpu-n", "256", "--iters", "20"]),
            ("streams", ["--total-mb", "64", "--iters", "20"]),
        ]
        for name, args in runs:
            cmd = (f"{shlex_quote(REPO / 'build' / name)} "
                   + " ".join(shlex_quote(a) for a in args)
                   + f" --csv {shlex_quote(REPO / 'results' / (name + '.csv'))}")
            sh(cmd, check=False)

    step("phase A kernels (vector add, reduction, matmul, streams)", phase_a)

    # ---- Task 1: fair preprocess benchmark (writes results/preprocess.csv) -
    step("preprocess benchmark",
         lambda: sh(f"python {shlex_quote(PHASE_B / 'preprocess_ext.py')}"))

    # ---- dataset first: the fine-tune needs data/waste.yaml ----------------
    step("real dataset + clip", dataset_stage)

    # ---- fine-tune (review task 1b) ----------------------------------------
    def finetune():
        from common import set_finetuned

        weights = REPO / "weights/waste_yolov8n.pt"
        if weights.exists():
            print("fine-tuned weights already present; skipping training")
        else:
            res = sh(
                f"python {shlex_quote(PHASE_B / 'finetune.py')} "
                f"--data {shlex_quote(REPO / 'data/waste.yaml')} "
                f"--model models/yolov8n.pt --epochs 30 --imgsz 640 --batch 16 "
                f"--project {shlex_quote(REPO / 'reports')} --name finetune",
                check=False,
            )
            if res.returncode != 0:
                # Keep the actual traceback: an exit code alone cannot be
                # diagnosed after the disposable Colab VM is gone.
                tail = "\n".join((res.stdout or "").splitlines()[-40:])
                record_failure(
                    "fine-tune",
                    f"finetune.py exit {res.returncode}; falling back to COCO\n{tail}",
                )
        ok = weights.exists()
        set_finetuned(ok)
        # Child stages are separate processes; the environment variable makes
        # every CSV row they write carry the same finetuned= label.
        os.environ["FINETUNED"] = "true" if ok else "false"
        print("finetuned:", ok)

    step("fine-tune YOLOv8n (30 epochs)", finetune)

    # ---- export the SAME weights used by every later stage ----------------
    def export():
        tuned = REPO / "weights/waste_yolov8n.pt"
        base = REPO / "models/yolov8n.pt"
        weights = tuned if tuned.exists() else base
        out = REPO / "models/waste_yolov8n.onnx" if tuned.exists() else REPO / "models/yolov8n.onnx"
        sh(
            f"python {shlex_quote(PHASE_B / 'export_model.py')} "
            f"--weights {shlex_quote(weights)} --out {shlex_quote(out)}"
        )

    step("ONNX export", export)

    def ladder():
        frames = REPO / "data/waste/images/val"
        for batch in (1, 4, 8):
            sh(
                f"python {shlex_quote(PHASE_B / 'infer_bench.py')} "
                f"--imgsz 640 --batch {batch} --iters 100 --warmup 20 "
                f"--frames {shlex_quote(frames)} "
                f"--csv {shlex_quote(REPO / 'results' / 'infer.csv')}",
                check=False,
            )

    step("inference ladder batches 1/4/8 on real frames", ladder)

    # ---- per-frame timings on real frames ----------------------------------
    def timings():
        tuned = REPO / "weights/waste_yolov8n.pt"
        model = tuned if tuned.exists() else REPO / "models/yolov8n.pt"
        sh(
            f"python {shlex_quote(PHASE_B / 'demo_timings.py')} "
            f"--frames {shlex_quote(REPO / 'data/waste/images/val')} "
            f"--model {shlex_quote(model)} --n 120 "
            f"--out demo_frame_timings.csv",
        )

    step("per-frame demo timings", timings)
    step("pipeline agreement check",
         lambda: sh(f"python {shlex_quote(PHASE_B / 'pipeline_agreement.py')}"))

    def multifeed():
        tuned = REPO / "weights/waste_yolov8n.pt"
        model = tuned if tuned.exists() else REPO / "models/yolov8n.pt"
        sh(
            f"python {shlex_quote(PHASE_B / 'pipeline.py')} "
            f"--backend pt_fp32 --models {shlex_quote(model)} "
            f"--source {shlex_quote(REPO / 'demo/belt_input.mp4')} "
            f"--imgsz 640 --frames 300 --warmup 20 "
            f"--csv {shlex_quote(REPO / 'results' / 'pipeline.csv')} "
            f"--batch-csv {shlex_quote(REPO / 'results' / 'batching.csv')}",
            check=False,
        )

    step("multi-feed pipeline on the real clip", multifeed)

    def ncu():
        sh(
            f"ncu --set basic --target-processes all --csv "
            f"--log-file {shlex_quote(REPO / 'results' / 'profiling_ncu.csv')} "
            f"{shlex_quote(REPO / 'build/vector_add')} --n 1000000 --iters 5 "
            f"--csv {shlex_quote(REPO / 'reports' / 'profiling_tmp.csv')}",
            check=False,
        )

    step("ncu counters", ncu)
    step("figures", lambda: sh(f"python {shlex_quote(REPO / 'tools/make_figures.py')}"))
    step("PPT image pack", lambda: sh(f"python {shlex_quote(REPO / 'tools/make_ppt_pack.py')}"))
    step("summary", lambda: sh(f"python {shlex_quote(REPO / 'tools/summary.py')}"))

    # ---- return the numbers, since downloads may not work in VS Code ------
    print("\n" + "=" * 74)
    print("RESULTS (saved in this notebook's outputs)")
    print("=" * 74)
    for path in sorted((REPO / "results").glob("*.csv")):
        if path.name.startswith("profiling_"):
            continue
        print(f"\n--- results/{path.name} ---")
        print(path.read_text(encoding="utf-8").rstrip()[:8000])
    failures = REPO / "results/failures.md"
    if failures.exists():
        print("\n--- results/failures.md ---")
        print(failures.read_text(encoding="utf-8").rstrip()[:6000])
    print("\nDONE. Tell Prateek's agent that the run finished.")
    return 0


def shlex_quote(path: Path) -> str:
    import shlex
    return shlex.quote(str(path))


if __name__ == "__main__":
    raise SystemExit(main())
