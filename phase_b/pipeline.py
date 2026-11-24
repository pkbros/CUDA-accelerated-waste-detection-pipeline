"""
phase_b/pipeline.py -- end-to-end, pipelined conveyor waste-detection loop.

===============================================================================
WHAT THIS MEASURES
===============================================================================
Every number in the CSVs and on stdout comes from a wall-clock measurement of a
whole loop of --frames iterations.  This is an END-TO-END throughput number
(capture + preprocess + infer + postprocess, including Python launch overhead),
NOT a kernel time and NOT a CUDA-event time.  The C++/Phase A benchmarks use
cudaEvent; here the unit of work is a whole frame taken from a camera-like
source, so we time with time.perf_counter() and report FPS = frames / seconds.

===============================================================================
TIMELINE (time flows to the right; k = frame index)
===============================================================================
pipelined_2stream: N=2 pinned buffers + N=2 copy streams; preprocess/infer/
postprocess run on the default stream (backends bind to the *current* stream).

 copy stream 0 :  H2D(k)              H2D(k+2)             H2D(k+4) ...
 copy stream 1 :        H2D(k+1)              H2D(k+3)             ...
 default       :        wait  pre(k) infer(k) post(k)  wait pre(k+1) infer(k+1) ...
                                   ^^^^^^^^^^^^^^^^^ H2D(k+1) overlaps infer(k) ^^^^^

pipelined_3stream is the same picture with N=3 copy streams / pinned buffers,
so the CPU producer can be filling buffer k+2 while frame k is still in infer.

The CUDA events are the dependency EDGES (record on producer stream, wait on
consumer stream):
    copy_done[s]  : H2D(buffer s)  -> preprocess(buffer s)   (default waits)
    infer_done[s] : end of frame k -> reuse of buffer s by a later copy
No cudaDeviceSynchronize() is used between frames.  There is exactly ONE device
synchronize per measured run, at the very end, so all GPU work is included.

===============================================================================
WHAT IS *NOT* OVERLAPPED (and why) -- honesty over a fake 2x
===============================================================================
1. Within a SINGLE frame, preprocess -> infer -> postprocess are strictly serial
   on the default stream.  This is a true data dependency: infer needs the
   preprocessed tensor; postprocess needs the inference output.
2. We NEVER run two inferences concurrently on the same backend session.
   PyTorch / ONNX Runtime enqueue their kernels on the current stream and most
   backends serialise work internally; launching two overlapping infers would
   either not overlap or be incorrect.  One infer stream, one infer at a time.
3. Our postprocess is a lightweight tensor filter and does NOT call
   .cpu()/.item(), so it adds no sync of its own.  It still cannot overlap the
   SAME frame's infer.  NOTE: backends.py already performs its own NMS +
   .detach().cpu() decode INSIDE infer(), which is a real D2H readback and a
   CPU-side wait for that frame's GPU work.  That backend-internal readback
   limits how much of H2D(k+1) can overlap infer(k) -- we report the measured
   speedup honestly instead of assuming a perfect 2x.
4. Python/CPU work (cv2 frame decode, filling pinned memory, result-list
   handling, kernel launches) is single-threaded Python.  It overlaps with GPU
   work only opportunistically; we do not claim Python is parallelised.
5. The producer may stall (a non-blocking CPU event poll) when it has raced a
   full pipeline depth ahead of the GPU -- that is correct backpressure, not
   a device-wide sync.

Synthetic footage is clearly labelled in the CSV extra column ("synthetic=true")
and a warning is printed.  It is never presented as real conveyor footage.
"""

from __future__ import annotations

import argparse
import time

# common.py lives next to this file.  Support both `python phase_b/pipeline.py`
# (script dir on sys.path) and `python -m phase_b.pipeline` (package form).
try:
    from common import (
        append_stats,
        require_cuda,
        results_dir,
        stats_of,
    )
except ModuleNotFoundError:  # pragma: no cover - package import form
    from phase_b.common import (  # type: ignore[no-redef]
        append_stats,
        require_cuda,
        results_dir,
        stats_of,
    )


# Number of independent full-loop timing samples per configuration.  Each sample
# is one loop of --frames iterations; mean/std are computed over these samples.
REPS = 5
CONF_THRES = 0.25  # postprocess score threshold (GPU-side, no host readback)


# ---------------------------------------------------------------------------
# Lazy factories (backends.py / preprocess_ext.py are written in parallel and
# may not be importable on this machine -- never import them at module load).
# ---------------------------------------------------------------------------
def _backend_factory():
    """Return get_backend from backends.py, lazily, without importing torch here."""
    try:
        from backends import get_backend
    except ModuleNotFoundError:  # pragma: no cover - package import form
        from phase_b.backends import get_backend  # type: ignore[no-redef]
    return get_backend


def _resolve_model_path(models: str, backend_name: str) -> str | None:
    """Pick a model file from --models.

    If --models is a file, use it.  If it is a directory, prefer a file whose
    name contains the backend name.  If nothing matches we return None and let
    get_backend fall back to its own default (e.g. a pretrained YOLO); we print
    what happened so the choice is recorded, not hidden.
    """
    from pathlib import Path

    p = Path(models)
    if p.is_file():
        return str(p)
    if p.is_dir():
        exts = {".pt", ".pth", ".onnx", ".engine", ".plan"}
        candidates = sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in exts)
        key = backend_name.lower()
        for q in candidates:
            if key in q.name.lower() or q.stem.lower() in key:
                return str(q)
        print(f"[pipeline] no model in '{models}' matched '{backend_name}'; "
              f"letting get_backend use its default.")
        return None
    if not p.exists():
        print(f"[pipeline] --models '{models}' does not exist; "
              f"letting get_backend use its default.")
        return None
    return str(p)


# ---------------------------------------------------------------------------
# Camera simulator / frame source
# ---------------------------------------------------------------------------
class FrameSource:
    """Yield HWC uint8 BGR numpy frames, looping forever so N-frame runs work.

    Priority: real video (--source)  >  directory of images (--images)  >
    synthetic generator.  The synthetic source labels itself as such.
    """

    def __init__(self, source: str | None = None, images: str | None = None,
                 imgsz: int = 640) -> None:
        from pathlib import Path

        import cv2  # lazy: only needed when a source actually produces frames

        self.h = self.w = int(imgsz)  # all sources are normalised to imgsz x imgsz
        self._i = 0
        self.kind = "synthetic"
        self.is_synthetic = True
        self._paths: list[str] = []
        self._cap = None

        if source:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture could not open --source '{source}'")
            self._cap = cap
            self.kind = "video"
            self.is_synthetic = False
        elif images:
            d = Path(images)
            if not d.is_dir():
                raise RuntimeError(f"--images '{images}' is not a directory")
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self._paths = sorted(
                str(q) for q in d.iterdir()
                if q.is_file() and q.suffix.lower() in exts
            )
            if not self._paths:
                raise RuntimeError(f"no images found in '{images}'")
            self.kind = "images"
            self.is_synthetic = False
        else:
            print("\n" + "!" * 72)
            print("WARNING: no --source/--images given -> using SYNTHETIC footage.")
            print("Frames are moving rectangles drawn with cv2, NOT real belt video.")
            print("The CSV extra column records synthetic=true for these results.")
            print("!" * 72 + "\n")

    # -- individual frame generators -----------------------------------------
    def _synthetic_frame(self):
        import cv2
        import numpy as np

        i = self._i
        img = np.full((self.h, self.w, 3), 36, dtype=np.uint8)  # dark belt
        # A moving "belt" seam.
        cv2.line(img, (0, self.h // 2), (self.w, self.h // 2), (96, 96, 96), 2)
        # Deterministic moving BGR rectangles (fake waste items).
        for j in range(5):
            color = ((40 + 38 * j) % 255, (90 + 27 * j) % 255, (170 - 20 * j) % 255)
            x = int((i * (3 + j) + j * 97) % max(1, self.w - 56))
            y = int((i * (2 + j) + j * 53) % max(1, self.h - 56))
            cv2.rectangle(img, (x, y), (x + 48, y + 48), color, -1)
        return img

    def _read_one(self):
        import cv2

        if self.kind == "synthetic":
            return self._synthetic_frame()
        if self.kind == "images":
            p = self._paths[self._i % len(self._paths)]
            f = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
            if f is None:
                raise RuntimeError(f"cv2.imread failed on '{p}'")
        else:  # video
            ok, f = self._cap.read()
            if not ok:  # loop the video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self._cap.read()
            if not ok:
                raise RuntimeError("--source video could not be read even after rewind")
        # Normalise real frames to a fixed size so the pinned/staging buffers
        # have a constant shape.  (The GPU preprocess still does its own resize.)
        if f.shape[0] != self.h or f.shape[1] != self.w:
            f = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return f

    # -- iterator protocol ----------------------------------------------------
    def __iter__(self) -> "FrameSource":
        return self

    def __next__(self):
        f = self._read_one()
        self._i += 1
        return f  # HWC uint8 BGR numpy array


# ---------------------------------------------------------------------------
# Preprocess adapter: fused CUDA kernel if available, documented torch fallback
# ---------------------------------------------------------------------------
class Preprocessor:
    """Turn GPU uint8 BGR HWC tensors into normalised float32 RGB NCHW tensors.

    Input : (B, H, W, 3) uint8 CUDA, BGR, HWC.
    Output: (B, 3, imgsz, imgsz) float32 CUDA, RGB, values in [0, 1].

    `label` is "cuda_kernel" ONLY if the external fused kernel was actually
    called successfully; otherwise it is "torch_fallback" and the CSV says so.
    We never pretend the CUDA kernel ran.
    """

    def __init__(self, imgsz: int) -> None:
        self.imgsz = int(imgsz)
        self.label = "torch_fallback"
        self._fn = None          # optional batch-level helper
        self._fn_name = None
        self._ext = None         # compiled extension exposing preprocess_f32
        self._geom = None
        self._mean = None
        self._std = None
        self._try_load_external()

    def _try_load_external(self) -> None:
        try:
            import preprocess_ext
        except Exception as exc:  # unavailable -> honest fallback
            print(f"[pipeline] preprocess_ext unavailable ({exc}); "
                  f"using torch fallback (F.interpolate + /255).")
            return

        # (a) Prefer a batch helper if this module ever grows one.
        for name in ("preprocess_batch", "preprocess", "fused_preprocess",
                     "letterbox_batch", "bgr_to_rgb_normalize"):
            fn = getattr(preprocess_ext, name, None)
            if callable(fn):
                self._fn = fn
                self._fn_name = name
                return

        # (b) The current preprocess_ext.py exposes the fused kernel through
        #     get_extension().preprocess_f32(src, out, dst_h, dst_w,
        #                                    scale, pad_x, pad_y, mean, std).
        #     Compiling the extension can fail (nvcc errors); that counts as
        #     "unavailable" and we fall back honestly rather than crashing.
        try:
            ext = preprocess_ext.get_extension()
            if not callable(getattr(ext, "preprocess_f32", None)):
                raise RuntimeError("extension has no preprocess_f32")
            h = w = self.imgsz  # FrameSource already resizes frames to imgsz
            self._geom = preprocess_ext._letterbox_geometry(h, w, self.imgsz)
            self._mean = list(preprocess_ext.DEFAULT_MEAN)
            self._std = list(preprocess_ext.DEFAULT_STD)
            self._ext = ext
        except Exception as exc:
            print(f"[pipeline] fused CUDA preprocess unavailable ({exc}); "
                  f"using torch fallback (F.interpolate + /255).")

    def __call__(self, dev_u8):
        """Run preprocess on the GPU tensor `dev_u8` (already on the current stream)."""
        import torch
        import torch.nn.functional as F

        if self._fn is not None:
            try:
                # Best-effort call with the two obvious signatures.  We accept the
                # result only if it is a CUDA float tensor of the expected shape;
                # anything else is treated as "kernel did not run" -> fallback.
                try:
                    out = self._fn(dev_u8, self.imgsz)
                except TypeError:
                    out = self._fn(dev_u8)
                if (torch.is_tensor(out) and out.is_cuda
                        and out.dtype == torch.float32
                        and tuple(out.shape[1:]) == (3, self.imgsz, self.imgsz)):
                    self.label = "cuda_kernel"
                    return out
                print(f"[pipeline] {self._fn_name} returned an unexpected tensor "
                      f"({getattr(out, 'shape', type(out))}); using torch fallback.")
            except Exception as exc:
                print(f"[pipeline] {self._fn_name} failed ({exc}); using torch fallback.")
            self._fn = None  # do not keep trying a broken helper

        if self._ext is not None:
            try:
                B = dev_u8.shape[0]
                out = torch.empty((B, 3, self.imgsz, self.imgsz),
                                  dtype=torch.float32, device="cuda")
                scale, pad_x, pad_y = self._geom
                # One fused kernel launch per image in the batch: letterbox
                # resize + BGR->RGB + HWC->CHW + uint8->float + normalise, all
                # on the CURRENT stream (so the pipeline's event edges hold).
                for i in range(B):
                    self._ext.preprocess_f32(dev_u8[i], out[i],
                                             self.imgsz, self.imgsz,
                                             scale, pad_x, pad_y,
                                             self._mean, self._std)
                self.label = "cuda_kernel"
                return out
            except Exception as exc:
                print(f"[pipeline] fused preprocess failed at runtime ({exc}); "
                      f"using torch fallback.")
                self._ext = None  # never claim the kernel ran after a failure

        # ---- Documented torch fallback: BGR->RGB, HWC->CHW, bilinear resize, /255.
        # This stays on the GPU (no host sync) and is suitable for throughput
        # measurement.  It is NOT a letterbox and is NOT the fused CUDA kernel.
        x = dev_u8[..., [2, 1, 0]]            # BGR -> RGB (index last dim)
        x = x.permute(0, 3, 1, 2).float()     # HWC uint8 -> NCHW float 0..255
        x = F.interpolate(x, size=(self.imgsz, self.imgsz),
                          mode="bilinear", align_corners=False)
        x = x / 255.0                         # normalise to 0..1
        self.label = "torch_fallback"
        return x


# ---------------------------------------------------------------------------
# Postprocess: GPU-side only, deliberately no host readback inside the loop.
# ---------------------------------------------------------------------------
def postprocess(results) -> list:
    """Threshold scores and count detections per image using tensor ops.

    Returns a list of per-image counts.  For a torch backend on CUDA these are
    GPU scalar tensors: we deliberately do NOT call .item()/.cpu() here because
    that would force a per-frame sync and destroy the pipeline.  The counts are
    intentionally left on the device (this benchmark measures throughput, not
    detection accuracy).  Works with numpy scores too, for injected test backends.
    """
    counts = []
    for r in results or []:
        if r is None:
            counts.append(0)
            continue
        # Normal contract: (boxes, scores, classes).  Some backends fall back to
        # returning a raw feature tensor; do not try to unpack that as a triple.
        if isinstance(r, (tuple, list)) and len(r) == 3:
            _boxes, scores, _classes = r
            if scores is None:
                counts.append(0)
                continue
            counts.append((scores > CONF_THRES).sum())
        else:
            counts.append(0)  # raw/unknown output: nothing we can threshold
    return counts


# ---------------------------------------------------------------------------
# Serial baseline -- variant "serial"
# ---------------------------------------------------------------------------
def run_serial(backend, source: FrameSource, prep: Preprocessor, frames: int,
               imgsz: int, warmup: int = 0, reps: int = REPS) -> dict:
    """One frame at a time on the DEFAULT stream with a synchronize PER STAGE.

    Timed with a wall clock around the whole loop (end-to-end, not a kernel time).
    """
    import torch

    def one_pass(n: int) -> None:
        for _ in range(n):
            frame = next(source)                         # HWC uint8 BGR numpy
            # --- preprocess stage: H2D (default stream) + preprocessing kernel ---
            t = torch.from_numpy(frame).unsqueeze(0).to("cuda", non_blocking=False)
            x = prep(t)                                  # (1,3,imgsz,imgsz) float
            torch.cuda.synchronize()                     # sync after preprocess
            # --- infer stage: backend uses the current (default) stream ---
            res = backend.infer(x)
            torch.cuda.synchronize()                     # sync after infer
            # --- postprocess stage ---
            postprocess(res)
            torch.cuda.synchronize()                     # sync after postprocess

    if warmup:
        one_pass(min(warmup, frames))
    samples: list[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_pass(frames)
        torch.cuda.synchronize()                         # include all GPU work
        samples.append((time.perf_counter() - t0) * 1e3)
    return stats_of(samples)


# ---------------------------------------------------------------------------
# Pipelined loop -- variants "pipelined_2stream" / "pipelined_3stream"
# ---------------------------------------------------------------------------
def run_pipelined(backend, source: FrameSource, prep: Preprocessor, frames: int,
                  imgsz: int, nstreams: int, batch: int = 1, warmup: int = 0,
                  reps: int = REPS) -> dict:
    """Frame-level pipeline: N pinned buffers + N copy streams.

    We overlap (a) the CPU producer filling buffer k+1 and (b) the H2D copy of
    buffer k+1 on its own copy stream, with (c) inference of frame k on the
    default stream.  We do NOT overlap two inferences (see module docstring).
    One device synchronize per measured run, at the very end.
    """
    import torch

    B = int(batch)
    H = W = int(imgsz)
    default = torch.cuda.current_stream()

    # ---- N copy streams, each with its own pinned staging buffer and GPU buffer.
    copy_streams = [torch.cuda.Stream() for _ in range(nstreams)]
    dev_u8 = [torch.empty((B, H, W, 3), dtype=torch.uint8, device="cuda")
              for _ in range(nstreams)]
    pinned = [torch.empty((B, H, W, 3), dtype=torch.uint8, pin_memory=True)
              for _ in range(nstreams)]  # <-- pinned host staging buffers
    copy_done = [torch.cuda.Event() for _ in range(nstreams)]
    infer_done = [torch.cuda.Event() for _ in range(nstreams)]
    # Seed the events so the first poll/wait is well-defined (completed).
    for s in range(nstreams):
        copy_done[s].record(default)
        infer_done[s].record(default)

    def one_pass(n: int) -> None:
        for k in range(n):
            s = k % nstreams               # round-robin over in-flight slots
            cs = copy_streams[s]

            # (BACKPRESSURE) The CPU must not overwrite the pinned buffer while
            # the H2D engine is still reading it from an earlier frame.  The
            # prior frame's infer_done implies its copy_done.  query() is a
            # non-blocking CPU-side poll -- NOT a device-wide synchronize.
            while not infer_done[s].query():
                pass

            # PRODUCER (CPU): fill this slot's pinned buffer with B frames
            # (one timestep across B simulated camera feeds when batching).
            for i in range(B):
                pinned[s][i].copy_(torch.from_numpy(next(source)))

            # COPY STAGE (copy stream s): async H2D from pinned memory into the
            # slot's GPU buffer, then record the copy->preprocess dependency edge.
            with torch.cuda.stream(cs):
                cs.wait_event(infer_done[s])  # GPU edge: do not clobber dev_u8[s]
                                              # before the previous infer read it
                dev_u8[s].copy_(pinned[s], non_blocking=True)  # async H2D
                copy_done[s].record(cs)       # EDGE: H2D(s) -> preprocess(s)

            # PREPROCESS -> INFER -> POSTPROCESS on the default stream.  Backends
            # bind to the *current* stream, so we keep all three here.
            with torch.cuda.stream(default):
                default.wait_event(copy_done[s])  # EDGE: wait for this frame's H2D
                x = prep(dev_u8[s])               # GPU preprocess on default stream
                res = backend.infer(x)            # ONE infer, on default stream
                postprocess(res)                  # GPU-side postprocess
                infer_done[s].record(default)     # EDGE: frame k -> reuse of slot s
        # NOTE: no sync here.  Caller stops the clock then syncs ONCE below.

    if warmup:
        one_pass(min(warmup, frames))
    samples: list[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_pass(frames)
        torch.cuda.synchronize()                         # the single end-of-run sync
        samples.append((time.perf_counter() - t0) * 1e3)
    return stats_of(samples)


# ---------------------------------------------------------------------------
# Reporting helpers -- every value printed is derived from measured stats.
# ---------------------------------------------------------------------------
def _extra_base(source: FrameSource, backend_name: str, prep_label: str) -> str:
    bits = [
        f"source={source.kind}",
        f"backend={backend_name}",
        f"preprocess={prep_label}",
        "end_to_end_wall_clock=true",
    ]
    if source.is_synthetic:
        bits.append("synthetic=true")
    return ",".join(bits)


def _csv_filename(path_arg: str) -> str:
    """Repo convention: all results live in results/.  Use the basename of --csv
    as the filename inside results_dir() (common.append_row writes the header
    and device/arch columns for us)."""
    from pathlib import Path
    return Path(path_arg).name


def _parse_feeds(text: str) -> list[int]:
    feeds = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        v = int(part)
        if v < 1:
            raise ValueError("feeds must be positive integers")
        feeds.append(v)
    return feeds or [1]


def _throughput(stats: dict, frames: int, batch: int = 1) -> float:
    """FPS from the measured mean loop time: total frames / mean seconds."""
    return (frames * batch) / (stats["mean"] / 1e3) if stats["mean"] > 0 else 0.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--backend", default="pt_fp32")
    p.add_argument("--models", default="models")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--source", default=None, help="video file / camera index")
    p.add_argument("--images", default=None, help="directory of images")
    p.add_argument("--feeds", default="1,2,4")
    p.add_argument("--csv", default="../results/pipeline.csv")
    p.add_argument("--batch-csv", default="../results/batching.csv")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    require_cuda()

    factory = _backend_factory()
    model_path = _resolve_model_path(args.models, args.backend)
    print(f"[pipeline] backend={args.backend} model_path={model_path} "
          f"imgsz={args.imgsz} frames={args.frames} warmup={args.warmup} "
          f"reps={REPS}")

    source = FrameSource(source=args.source, images=args.images, imgsz=args.imgsz)
    prep = Preprocessor(args.imgsz)

    # Create the batch=1 backend used by the three pipeline variants and warm it
    # up.  Running the warmup pass also finalises the preprocessor label, so the
    # extra column can report accurately what actually ran.
    backend = factory(args.backend, model_path, imgsz=args.imgsz, batch=1)
    backend.warmup(args.warmup)
    run_pipelined(backend, source, prep, min(args.warmup, args.frames), args.imgsz,
                  nstreams=2, batch=1, warmup=0, reps=1)
    extra = _extra_base(source, args.backend, prep.label)
    print(f"[pipeline] extra='{extra}'")

    # ---- bench="pipeline": serial baseline and two pipeline depths ----------
    pipeline_stats: dict[str, dict] = {}
    pipeline_rows: list[tuple[str, dict]] = []

    serial = run_serial(backend, source, prep, args.frames, args.imgsz,
                        warmup=0, reps=REPS)
    pipeline_stats["serial"] = serial
    pipeline_rows.append(("serial", serial))
    append_stats("pipeline", "serial", args.frames, "frames", serial,
                 _throughput(serial, args.frames), "FPS", extra=extra,
                 filename=_csv_filename(args.csv))

    for variant, nstreams in (("pipelined_2stream", 2), ("pipelined_3stream", 3)):
        st = run_pipelined(backend, source, prep, args.frames, args.imgsz,
                           nstreams=nstreams, batch=1, warmup=0, reps=REPS)
        pipeline_stats[variant] = st
        pipeline_rows.append((variant, st))
        append_stats("pipeline", variant, args.frames, "frames", st,
                     _throughput(st, args.frames), "FPS", extra=extra,
                     filename=_csv_filename(args.csv))

    print("\n=== Pipeline summary (end-to-end wall clock, "
          f"{args.frames} frames/run, n={REPS}) ===")
    print(f"{'variant':<22}{'ms/frame':>12}{'FPS':>12}")
    for variant, st in pipeline_rows:
        ms_per_frame = st["mean"] / args.frames
        print(f"{variant:<22}{ms_per_frame:>12.3f}"
              f"{_throughput(st, args.frames):>12.2f}")
    serial_mean = pipeline_stats["serial"]["mean"]
    for variant in ("pipelined_2stream", "pipelined_3stream"):
        speedup = serial_mean / pipeline_stats[variant]["mean"] \
            if pipeline_stats[variant]["mean"] > 0 else 0.0
        print(f"speedup vs serial = {speedup:.2f}x  [{variant}]")

    # ---- bench="batching": simulate 1, 2, 4 camera feeds --------------------
    feeds = _parse_feeds(args.feeds)
    print("\n=== Batching summary (stack feeds into one inference) ===")
    print(f"{'feeds':>6}{'variant':>17}{'ms/step':>12}"
          f"{'total FPS':>12}{'FPS/feed':>12}")

    for F in feeds:
        fb = factory(args.backend, model_path, imgsz=args.imgsz, batch=F)
        fb.warmup(args.warmup)

        s_batch = run_serial(fb, source, prep, args.frames, args.imgsz,
                             warmup=0, reps=REPS)
        p_batch = run_pipelined(fb, source, prep, args.frames, args.imgsz,
                                nstreams=2, batch=F, warmup=0, reps=REPS)

        total_serial = _throughput(s_batch, args.frames, batch=F)
        total_pipe = _throughput(p_batch, args.frames, batch=F)
        fps_per_feed_serial = total_serial / F
        fps_per_feed_pipe = total_pipe / F

        common_extra = _extra_base(source, args.backend, prep.label)
        serial_extra = f"{common_extra},feeds={F},fps_per_feed={fps_per_feed_serial:.4f}"
        pipe_extra = f"{common_extra},feeds={F},fps_per_feed={fps_per_feed_pipe:.4f}"

        append_stats("batching", "serial_batch", F, "feeds", s_batch,
                     total_serial, "FPS", extra=serial_extra,
                     filename=_csv_filename(args.batch_csv))
        append_stats("batching", "pipelined_batch", F, "feeds", p_batch,
                     total_pipe, "FPS", extra=pipe_extra,
                     filename=_csv_filename(args.batch_csv))

        for variant, st, total, per_feed in (
            ("serial_batch", s_batch, total_serial, fps_per_feed_serial),
            ("pipelined_batch", p_batch, total_pipe, fps_per_feed_pipe),
        ):
            print(f"{F:>6}{variant:>17}{st['mean'] / args.frames:>12.3f}"
                  f"{total:>12.2f}{per_feed:>12.2f}")

    print(f"\n[pipeline] CSVs written under {results_dir()}")


if __name__ == "__main__":
    main()
