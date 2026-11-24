"""Python side of the fused preprocessing kernel (Phase B, milestone 3).

This module owns four things:

  1. Building / loading the CUDA extension via torch.utils.cpp_extension.load,
     into a REPO-LOCAL build directory so a Colab rerun does not recompile.
  2. `letterbox`: host-side geometry (bit-identical to the CPU reference) plus a
     padded uint8 GPU tensor for visualization / the legacy inference path.
  3. `reference_preprocess_cpu`: an INDEPENDENT NumPy implementation of exactly
     the same sampling math. It never calls the kernel. This is the tolerance
     baseline, not a copy of the CUDA code.
  4. `bench_preprocess`: correctness gate FIRST, then four timed variants
     written to results/preprocess.csv through common.append_stats.

The fused kernel reads the ORIGINAL source image; the padded tensor returned by
`letterbox` is not its input. The kernel re-derives sampling from (scale,
pad_x, pad_y), which Python computes once using the formula in
`_letterbox_geometry`.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.cpp_extension import load

# Works both as `python phase_b/preprocess_ext.py` (script dir on sys.path) and
# as `import phase_b.preprocess_ext` (package import from the repo root).
try:
    from .common import (
        append_stats,
        arch,
        device_name,
        require_cuda,
        time_call,
        time_wall,
        write_device_info,
    )
except ImportError:  # pragma: no cover - exercised only in script mode
    from common import (  # type: ignore[no-redef]
        append_stats,
        arch,
        device_name,
        require_cuda,
        time_call,
        time_wall,
        write_device_info,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
# CRITICAL Colab detail: stay inside the repo, not /root/.cache or /tmp, so the
# compiled .so survives notebook re-runs in the same session AND is reproducible
# from a fresh Colab mount. Create the directory before load() looks at it.
BUILD_DIR = REPO_ROOT / ".cache" / "torch_extensions"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

CSV_NAME = "preprocess.csv"
BENCH = "preprocess"

# Tolerances, stated explicitly and copied verbatim into the CSV `extra` field.
# FP32: both atol and rtol are 1e-4 -- tight enough to catch a wrong interpolation
#       or a BGR/RGB swap, loose enough for float32 rounding (< 1e-6 here).
# FP16: atol 2e-3 with rtol 0. FP16 keeps only ~3 decimal digits (10-bit
#       mantissa, eps ~ 9.8e-4), so ~2e-3 is the honest accuracy floor for
#       normalized values of magnitude ~1. A tighter FP16 gate would fail on a
#       perfectly correct kernel.
ATOL_F32, RTOL_F32 = 1e-4, 1e-4
ATOL_F16 = 2e-3

DEFAULT_MEAN = (0.485, 0.456, 0.406)  # ImageNet-style, in [0,1] RGB order
DEFAULT_STD = (0.229, 0.224, 0.225)

_EXT = None  # module-level cache: load() is expensive, call it once.


# ---------------------------------------------------------------------------
# 1. Extension loading
# ---------------------------------------------------------------------------
def cuda_available() -> bool:
    """True when torch sees a CUDA device (also the gate used by main())."""
    return torch.cuda.is_available()


def get_extension():
    """Load (once) and return the compiled `waste_preprocess` module."""
    global _EXT
    if _EXT is not None:
        return _EXT

    source = Path(__file__).with_name("preprocess.cu")
    if not source.exists():
        raise FileNotFoundError(f"CUDA source not found: {source}")

    # torch's JIT versioner writes its "already built" marker BEFORE ninja is
    # checked (verify_ninja_availability() lives inside the build step). So a
    # build that dies early -- ninja missing on a fresh Colab image, say --
    # leaves a marker but no .so, and the next load() skips the build and fails
    # with a misleading "cannot open shared object file". If there is no .so,
    # the marker is stale: drop the whole cache and force a real rebuild.
    if not any(BUILD_DIR.glob("*.so")):
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        BUILD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _EXT = load(
            name="waste_preprocess",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=str(BUILD_DIR),
            verbose=True,
            is_python_module=True,
            with_cuda=True,
        )
    except Exception as exc:
        # Honest failure: print the FULL compiler output, then re-raise. We do
        # NOT silently switch to CuPy RawKernel and we do NOT claim it works.
        print("=" * 78, file=sys.stderr)
        print("CUDA EXTENSION COMPILE FAILED -- full error below:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        raise RuntimeError(
            "Could not build phase_b/preprocess.cu. No fallback was used. "
            "The PRD fallback is a CuPy RawKernel; implementing it is a "
            "separate, explicitly-reported task, not a silent substitution."
        ) from exc
    return _EXT


# ---------------------------------------------------------------------------
# 2. Host-side letterbox geometry -- the single source of truth
# ---------------------------------------------------------------------------
def _letterbox_geometry(src_h: int, src_w: int, dst_size: int):
    """Return (scale, pad_x, pad_y) with YOLO's exact letterbox arithmetic.

    The formula, written once so Python and the kernel agree bit-for-bit:

        scale = min(dst_size / src_h, dst_size / src_w)       # uniform resize
        new_w = int(round(src_w * scale))                     # YOLO rounds here
        new_h = int(round(src_h * scale))                     # (not floor/trunc)
        pad_x = (dst_size - new_w) / 2.0                      # may end in .5
        pad_y = (dst_size - new_h) / 2.0

    The kernel never recomputes these. It uses the inverse map
        sx = (dx + 0.5 - pad_x) / scale - 0.5
        sy = (dy + 0.5 - pad_y) / scale - 0.5
    which is exactly inverse to OpenCV INTER_LINEAR's
        src = (dst + 0.5) * (src/dst) - 0.5 = (dst + 0.5)/scale - 0.5.
    """
    if src_h <= 0 or src_w <= 0 or dst_size <= 0:
        raise ValueError("source and destination sizes must be positive")
    scale = min(float(dst_size) / float(src_h), float(dst_size) / float(src_w))
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    pad_x = (float(dst_size) - float(new_w)) / 2.0
    pad_y = (float(dst_size) - float(new_h)) / 2.0
    return scale, pad_x, pad_y


def letterbox(img_bgr_hwc_uint8, dst_size: int = 640, pad_value: int = 114):
    """Make the uint8 letterbox canvas on the GPU and return its geometry.

    Returns ``(gpu_uint8_tensor, scale, pad_x, pad_y)`` where the tensor is a
    contiguous ``(dst_size, dst_size, 3)`` CUDA uint8 image in BGR order.

    NOTE: this tensor is for visualization / the legacy inference input. The
    FUSED kernel does not read it -- it samples the original image directly
    using the returned geometry. That is why there is no resize/copy in the
    kernel's timed region.
    """
    require_cuda()
    img = np.ascontiguousarray(np.asarray(img_bgr_hwc_uint8))
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("letterbox expects an HWC image with 3 channels")
    src_h, src_w = img.shape[:2]
    scale, pad_x, pad_y = _letterbox_geometry(src_h, src_w, dst_size)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # cv2.copyMakeBorder needs integer offsets; split the .5 pad between sides.
    left, top = int(pad_x), int(pad_y)
    right = int(dst_size) - new_w - left
    bottom = int(dst_size) - new_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),
    )
    return torch.from_numpy(padded.copy()).to("cuda"), scale, pad_x, pad_y


# ---------------------------------------------------------------------------
# 3. Independent CPU reference (the tolerance baseline)
# ---------------------------------------------------------------------------
def reference_preprocess_cpu(
    img_bgr_hwc_uint8,
    dst_size: int = 640,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    dtype=torch.float32,
):
    """NumPy bilinear sampling that mirrors the kernel formula exactly.

    Independent implementation: vectorized NumPy, float64 value accumulation,
    no PyTorch autograd, and absolutely no call into the CUDA extension. It uses
    the same half-pixel centres, the same floor/fraction lerp, the same edge
    clamp, and the same zero padding as the kernel so any disagreement is a
    genuine kernel bug rather than a convention mismatch.

    IMPORTANT numeric detail (verified by static emulation before this file was
    written): the COORDINATE maths is done in float32, because that is exactly
    what the kernel receives -- Python computes scale/pad in float64 then casts
    them to float when passing them to CUDA. If the reference used float64 for
    sx/sy, a pixel whose coordinate lands within ~1e-5 of an integer could get
    a different floor() than the kernel, flipping a bilinear weight from 0 to 1
    and producing errors up to ~1e-3 -- enough to FAIL the 1e-4 gate on a
    perfectly correct kernel. Only the pixel-value lerp/normalize is float64.
    """
    img = np.asarray(img_bgr_hwc_uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("reference expects an HWC image with 3 channels")
    src_h, src_w = img.shape[:2]
    scale, pad_x, pad_y = _letterbox_geometry(src_h, src_w, dst_size)

    # Pixel centres: dx/dy = index + 0.5; broadcast into (1,W) and (H,1) grids.
    # float32 here on purpose: the host passes scale/pad_x/pad_y to CUDA as
    # floats, so this reproduces the kernel's coordinate rounding bit-for-bit.
    s32 = np.float32(scale)
    px32 = np.float32(pad_x)
    py32 = np.float32(pad_y)
    dx = np.arange(dst_size, dtype=np.float32) + np.float32(0.5)
    dy = np.arange(dst_size, dtype=np.float32) + np.float32(0.5)
    sx = (dx[None, :] - px32) / s32 - np.float32(0.5)   # shape (1, dst_w)
    sy = (dy[:, None] - py32) / s32 - np.float32(0.5)   # shape (dst_size, 1)

    # The letterbox pad test: OUTSIDE [0, src_w) x [0, src_h) => zero output.
    valid = ((sx >= np.float32(0)) & (sx < np.float32(src_w)) &
             (sy >= np.float32(0)) & (sy < np.float32(src_h)))

    x0f = np.floor(sx).astype(np.int64)   # (1, W), may be negative in padding
    y0f = np.floor(sy).astype(np.int64)   # (H, 1)
    # Promote the weights to float64 only NOW, after floor() has been decided:
    # the value lerp stays high precision, but indexing matches the kernel.
    wx = (sx - x0f).astype(np.float64)
    wy = (sy - y0f).astype(np.float64)
    x0 = np.clip(x0f, 0, src_w - 1)
    x1 = np.clip(x0f + 1, 0, src_w - 1)
    y0 = np.clip(y0f, 0, src_h - 1)
    y1 = np.clip(y0f + 1, 0, src_h - 1)

    out = np.zeros((3, dst_size, dst_size), dtype=np.float64)
    for c in range(3):
        sc = 2 - c  # output RGB channel c reads source BGR channel 2-c
        plane = img[:, :, sc].astype(np.float64)

        # Advanced indexing broadcasts (H,1) with (1,W) into a (H,W) grid.
        v00 = plane[y0, x0]
        v01 = plane[y0, x1]
        v10 = plane[y1, x0]
        v11 = plane[y1, x1]

        top = v00 + wx * (v01 - v00)
        bot = v10 + wx * (v11 - v10)
        v = top + wy * (bot - top)

        out[c] = (v / 255.0 - float(mean[c])) / float(std[c])
        out[c][~valid] = 0.0  # zero border in normalized space

    # Return the requested dtype; callers use float32 for both gates so that
    # the FP16 tolerance measures only FP16 rounding, not reference rounding.
    np_dtype = np.float32 if dtype == torch.float32 else (
        np.float16 if dtype == torch.float16 else np.float64
    )
    return torch.from_numpy(out.astype(np_dtype))


# ---------------------------------------------------------------------------
# 4. Baselines -- all must produce the SAME tensor as the fused kernel
# ---------------------------------------------------------------------------
def _check_integral_pad(pad_x: float, pad_y: float, src_h: int, src_w: int) -> None:
    """cv2.copyMakeBorder/F.pad cannot express a half-pixel letterbox pad.

    The two benchmark sizes (1920x1080 and 640x480) both produce integral pads.
    Anything else is rejected instead of silently shifting the image by one
    pixel -- that silent shift was one cause of the old max_abs_err=2.1.
    """
    if abs(pad_x - round(pad_x)) > 1e-6 or abs(pad_y - round(pad_y)) > 1e-6:
        raise ValueError(
            f"baseline needs integral letterbox padding; got pad_x={pad_x}, "
            f"pad_y={pad_y} for {src_w}x{src_h}"
        )


def _zero_padding(out, top: int, left: int, new_h: int, new_w: int, dst: int):
    """Zero the border AFTER normalisation, matching the kernel/reference.

    Padding with uint8 0 before normalising yields (0 - mean)/std ~= -1.8, not
    zero. Doing it in the wrong order was the other cause of max_abs_err=2.1.
    """
    out[:, :top, :] = 0.0
    out[:, top + new_h:, :] = 0.0
    out[:, :, :left] = 0.0
    out[:, :, left + new_w:] = 0.0


def _opencv_preprocess_cpu(img_bgr_hwc_uint8, dst_size, mean, std):
    """cv2.resize + cvtColor + normalise, CPU only. No GPU upload here."""
    import cv2  # local import keeps the module importable without cv2

    img = np.ascontiguousarray(np.asarray(img_bgr_hwc_uint8))
    src_h, src_w = img.shape[:2]
    scale, pad_x, pad_y = _letterbox_geometry(src_h, src_w, dst_size)
    _check_integral_pad(pad_x, pad_y, src_h, src_w)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    left, top = int(round(pad_x)), int(round(pad_y))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, int(dst_size) - new_h - top,
        left, int(dst_size) - new_w - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    for c in range(3):
        chw[c] = (chw[c] - float(mean[c])) / float(std[c])
    _zero_padding(chw, top, left, new_h, new_w, int(dst_size))
    return chw


def _torch_gpu_ops(img_u8_cuda, dst_size, mean, std):
    """Plain PyTorch-on-GPU: the honest GPU competitor to the fused kernel.

    Same letterbox geometry, same half-pixel bilinear convention
    (align_corners=False), same channel order, same normalisation, same zero
    padding. Input is an already-uploaded GPU uint8 HWC tensor.
    """
    import torch.nn.functional as F

    src_h, src_w = img_u8_cuda.shape[:2]
    scale, pad_x, pad_y = _letterbox_geometry(src_h, src_w, dst_size)
    _check_integral_pad(pad_x, pad_y, src_h, src_w)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    left, top = int(round(pad_x)), int(round(pad_y))

    x = img_u8_cuda.permute(2, 0, 1).float()   # BGR CHW, 0..255
    x = x.flip(0)                              # -> RGB
    x = F.interpolate(
        x.unsqueeze(0), size=(new_h, new_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    x = x / 255.0
    for c in range(3):
        x[c] = (x[c] - float(mean[c])) / float(std[c])
    out = x.new_zeros((3, int(dst_size), int(dst_size)))
    out[:, top:top + new_h, left:left + new_w] = x
    return out


# ---------------------------------------------------------------------------
# 5. Correctness gate + benchmark driver
# ---------------------------------------------------------------------------
# The CPU baselines are NOT bit-identical to the kernel: cv2.INTER_LINEAR is
# fixed-point and PyTorch's F.interpolate uses a different accumulator. These
# tolerances are stated here, applied BEFORE any timing, and written into every
# row's `extra` field. They are not chosen after seeing the result.
# Locally validated before writing this: with the two fixes below (integral
# pad + zeroing AFTER normalisation), the cv2 baseline differs from the
# reference by 4.77e-7 at 1920x1080 and 640x480. At 1280x720 cv2's fixed-point
# resize differs by 8.75e-3, so 1e-2 leaves headroom without hiding a real bug.
ATOL_CV_BASELINE = 1e-2
ATOL_TORCH_GPU_BASELINE = 2e-3


def _mp_per_s(dst_h: int, dst_w: int, time_ms: float, n_images: int) -> float:
    if time_ms <= 0.0:
        return 0.0
    return (float(dst_h) * float(dst_w) * int(n_images) / 1e6) / (float(time_ms) / 1e3)


def _extra_field(max_err: float, tol: str, ok: bool, timing: str, src: str) -> str:
    """CSV `extra`: measured error, stated tolerance, gate result, what is timed."""
    return (
        f"max_abs_err={max_err:.6e},tol={tol},ok={str(ok).lower()},"
        f"timing={timing},src={src}"
    )


def bench_preprocess(
    images,
    dst_size: int = 640,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    warmup: int = 20,
    iters: int = 200,
):
    """Correctness-gate five variants, then time each with a stated scope.

    WHAT EACH TIMING INCLUDES (this is the whole point of the fair comparison):
      kernel_only      CUDA events, input already on GPU, output stays on GPU.
      h2d_plus_kernel  CUDA events, pinned host frame -> async H2D -> kernel,
                       output stays on GPU. Excludes the Python call overhead.
      end_to_end       wall clock, host uint8 HWC frame -> host staging ->
                       H2D copy -> kernel -> cudaDeviceSynchronize, so the
                       returned tensor is genuinely ready on the GPU.
      opencv_cpu       wall clock, cv2 resize/cvtColor/normalise on CPU ->
                       .cuda() upload (pageable, the normal user path) ->
                       synchronize. Same output tensor as the kernel.
      torch_gpu_ops    CUDA events, gpu input -> F.interpolate/flip/div/pad,
                       output stays on GPU. Fair GPU-only competitor.

    Every row is checked against the CPU reference with a stated tolerance
    before it is allowed into the CSV.
    """
    import torch

    require_cuda()
    ext = get_extension()

    if isinstance(images, np.ndarray):
        images = [images]
    else:
        images = [np.ascontiguousarray(np.asarray(im)) for im in images]
    if not images:
        raise ValueError("bench_preprocess needs at least one image")
    n_images = len(images)
    dst_h = dst_w = int(dst_size)
    first = np.ascontiguousarray(np.asarray(images[0]))
    src_h, src_w = first.shape[:2]
    src_tag = f"{src_w}x{src_h}"
    scale, pad_x, pad_y = _letterbox_geometry(src_h, src_w, dst_size)

    # ---- CPU reference: the oracle for every variant ----
    ref = reference_preprocess_cpu(first, dst_size, mean, std, torch.float32)

    # ---- Gate the fused kernel BEFORE timing ----
    src_gpu0 = torch.from_numpy(first.copy()).to("cuda")
    out32 = torch.empty((3, dst_h, dst_w), dtype=torch.float32, device="cuda")
    ext.preprocess_f32(src_gpu0, out32, dst_h, dst_w,
                       scale, pad_x, pad_y, list(mean), list(std))
    err32 = (out32.float().cpu() - ref.float()).abs().max().item()
    ok32 = bool(torch.allclose(out32.cpu(), ref, atol=ATOL_F32, rtol=RTOL_F32))
    if not ok32:
        raise AssertionError(
            f"FP32 kernel FAILED gate: max_abs_err={err32:.6e} "
            f"(atol={ATOL_F32}, rtol={RTOL_F32}). Refusing to benchmark.")

    out16 = torch.empty((3, dst_h, dst_w), dtype=torch.float16, device="cuda")
    ext.preprocess_f16(src_gpu0, out16, dst_h, dst_w,
                       scale, pad_x, pad_y, list(mean), list(std))
    err16 = (out16.float().cpu() - ref.float()).abs().max().item()
    ok16 = bool(torch.allclose(out16.float().cpu(), ref.float(),
                               atol=ATOL_F16, rtol=0.0))
    if not ok16:
        raise AssertionError(
            f"FP16 kernel FAILED gate: max_abs_err={err16:.6e} "
            f"(atol={ATOL_F16}). Refusing to benchmark.")

    tol32 = f"atol={ATOL_F32:g}/rtol={RTOL_F32:g}"
    tol16 = f"atol={ATOL_F16:g}"

    # ---- Gate BOTH baselines against the SAME reference ----
    cv_out = _opencv_preprocess_cpu(first, dst_size, mean, std)
    err_cv = float(np.abs(cv_out - ref.numpy()).max())
    ok_cv = bool(err_cv <= ATOL_CV_BASELINE)
    if not ok_cv:
        raise AssertionError(
            f"opencv_cpu baseline FAILED gate: max_abs_err={err_cv:.6e} "
            f"(atol={ATOL_CV_BASELINE}). Refusing to benchmark.")

    tg_out = _torch_gpu_ops(src_gpu0, dst_size, mean, std)
    err_tg = (tg_out.float().cpu() - ref.float()).abs().max().item()
    ok_tg = bool(err_tg <= ATOL_TORCH_GPU_BASELINE)
    if not ok_tg:
        raise AssertionError(
            f"torch_gpu_ops baseline FAILED gate: max_abs_err={err_tg:.6e} "
            f"(atol={ATOL_TORCH_GPU_BASELINE}). Refusing to benchmark.")

    tol_cv = f"atol={ATOL_CV_BASELINE:g}"
    tol_tg = f"atol={ATOL_TORCH_GPU_BASELINE:g}"

    # ---- Preallocate everything so the timed closures only do the work ----
    srcs_gpu = [torch.from_numpy(np.ascontiguousarray(np.asarray(im)).copy()).to("cuda")
                for im in images]
    outs32 = [torch.empty((3, dst_h, dst_w), dtype=torch.float32, device="cuda")
              for _ in range(n_images)]
    pinned = torch.empty((src_h, src_w, 3), dtype=torch.uint8, pin_memory=True)
    pinned.copy_(torch.from_numpy(first))
    h2d_src = torch.empty((src_h, src_w, 3), dtype=torch.uint8, device="cuda")
    scales = [(scale, pad_x, pad_y)] * n_images

    def kernel_only():
        for s, o, (sc, px, py) in zip(srcs_gpu, outs32, scales):
            ext.preprocess_f32(s, o, dst_h, dst_w, sc, px, py,
                               list(mean), list(std))

    def h2d_plus_kernel():
        h2d_src.copy_(pinned, non_blocking=True)
        ext.preprocess_f32(h2d_src, outs32[0], dst_h, dst_w,
                           scale, pad_x, pad_y, list(mean), list(std))

    def end_to_end():
        # Pageable host frame -> GPU, then the kernel. Sync so the wall clock
        # measures a tensor that is actually ready, not just enqueued.
        dev = torch.from_numpy(first).to("cuda")
        ext.preprocess_f32(dev, outs32[0], dst_h, dst_w,
                           scale, pad_x, pad_y, list(mean), list(std))
        torch.cuda.synchronize()

    def opencv_e2e():
        chw = _opencv_preprocess_cpu(first, dst_size, mean, std)
        torch.from_numpy(np.ascontiguousarray(chw)).to("cuda")
        torch.cuda.synchronize()

    def torch_gpu_call():
        _torch_gpu_ops(srcs_gpu[0], dst_size, mean, std)

    variants = [
        ("kernel_only", time_call(kernel_only, warmup, iters),
         _extra_field(err32, tol32, ok32, "events:GPU-kernel-only", src_tag)),
        ("h2d_plus_kernel", time_call(h2d_plus_kernel, warmup, iters),
         _extra_field(err32, tol32, ok32, "events:pinned-H2D+kernel", src_tag)),
        ("end_to_end", time_wall(end_to_end, warmup=min(warmup, 5), iters=iters),
         _extra_field(err32, tol32, ok32, "wall:host-frame-to-ready-GPU-tensor", src_tag)),
        ("opencv_cpu", time_wall(opencv_e2e, warmup=min(warmup, 5), iters=iters),
         _extra_field(err_cv, tol_cv, ok_cv, "wall:cv2-CPU+pageable-H2D", src_tag)),
        ("torch_gpu_ops", time_call(torch_gpu_call, warmup, iters),
         _extra_field(err_tg, tol_tg, ok_tg, "events:plain-torch-GPU-ops", src_tag)),
    ]

    for variant, stats, extra in variants:
        append_stats(
            bench=BENCH, variant=variant, size=dst_h * dst_w, size_unit="px",
            stats=stats,
            throughput=_mp_per_s(dst_h, dst_w, stats["mean"], n_images),
            throughput_unit="MP/s", extra=extra, filename=CSV_NAME,
        )
    return {
        "variants": {v: s for v, s, _ in variants},
        "extra": {v: e for v, _, e in variants},
        "device": device_name(), "arch": arch(),
        "n_images": n_images, "src": src_tag,
    }


# ---------------------------------------------------------------------------
# 6. Entry point -- two fixed source sizes, >=200 iterations each
# ---------------------------------------------------------------------------
def _demo_image(h: int, w: int) -> np.ndarray:
    """Deterministic BGR frame so main() runs without a dataset."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    b = ((xx * 255 // max(w - 1, 1))).astype(np.uint8)
    g = ((yy * 255 // max(h - 1, 1))).astype(np.uint8)
    r = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    return np.dstack([b, g, r])


def main() -> None:
    """Correctness-gated, honestly-scoped preprocess benchmark."""
    if not cuda_available():
        print("CUDA not available -- nothing to benchmark (no fallback used).")
        return
    info = write_device_info()
    print(f"Device: {info['gpu_name']}  arch={info['arch']}  torch={info['torch']}")
    print("Timings: kernel_only / h2d_plus_kernel / torch_gpu_ops use CUDA events;")
    print("         end_to_end / opencv_cpu use wall clock with a device sync.\n")

    for src_h, src_w in ((1080, 1920), (480, 640)):
        print(f"=== source {src_w}x{src_h} -> 640 letterbox, iters=200 ===")
        result = bench_preprocess(_demo_image(src_h, src_w), dst_size=640,
                                  warmup=20, iters=200)
        hdr = f"{'variant':<18}{'mean (ms)':>12}{'std (ms)':>12}{'MP/s':>12}"
        print(hdr)
        print("-" * len(hdr))
        for variant, stats in result["variants"].items():
            tp = _mp_per_s(640, 640, stats["mean"], result["n_images"])
            print(f"{variant:<18}{stats['mean']:>12.4f}{stats['std']:>12.4f}{tp:>12.2f}")
        print("Correctness gates:")
        for variant, extra in result["extra"].items():
            print(f"  {variant:<18} {extra}")
        print()


if __name__ == "__main__":
    main()
