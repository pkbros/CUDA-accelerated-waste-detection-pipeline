"""Backend abstractions for the Phase-B inference ladder.

Four rungs, all fed the *same* caller-owned CUDA tensor:

    pt_fp32   PyTorch / ultralytics, FP32 weights
    pt_fp16   PyTorch / ultralytics, FP16 weights
    ort_cuda  ONNX Runtime with the CUDA execution provider
    trt_fp16  TensorRT, FP16 builder config (stretch goal)

Why bother with FP16 at all: the Colab T4 has tensor cores that do FP16 at
roughly 65 TFLOPS versus roughly 8.1 TFLOPS for FP32. A real ladder should
therefore show a measurable gap between pt_fp32 and pt_fp16/trt_fp16. We report
what we actually measure; if a rung is slower we say so rather than hiding it.

CONTRACT -- pipeline.py imports Backend and get_backend from this module, so the
class name, attribute names, method names and signatures below are FIXED.
Additional attributes are additive and safe for that caller.

Heavy third-party imports stay INSIDE functions. ``import backends`` must work
on a machine with no torch / onnxruntime / tensorrt installed; an unavailable
rung becomes an actionable RuntimeError from get_backend(), which infer_bench.py
records as a SKIP instead of a fabricated CSV row.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

try:  # imported as a package: python -m phase_b.infer_bench
    from .common import require_cuda
except ImportError:  # imported as a script: python infer_bench.py from phase_b/
    from common import require_cuda


BACKEND_NAMES: tuple[str, ...] = ("pt_fp32", "pt_fp16", "ort_cuda", "trt_fp16")


# ---------------------------------------------------------------------------
# Shared decode / NMS helper
# ---------------------------------------------------------------------------
_NMS_UNRESOLVED = object()
_nms_fn: Any = _NMS_UNRESOLVED
_nms_error: str | None = None


def _get_nms_fn() -> Any:
    """Resolve ultralytics' public-ish NMS helper once, never at import time.

    We deliberately do NOT hand-write the YOLO head decode + NMS (that would be
    easy to get subtly wrong and would not match ultralytics' own predictions).
    The helper lives in ``ultralytics.utils.nms``. If that import moves or is
    unavailable we fall back to returning the literal raw head tensors -- never
    invented boxes -- and make the fallback loud in the CSV ``extra`` column.
    """
    global _nms_fn, _nms_error
    if _nms_fn is _NMS_UNRESOLVED:
        try:
            from ultralytics.utils.nms import non_max_suppression

            _nms_fn = non_max_suppression
        except Exception as exc:  # noqa: BLE001 - we want the full reason text
            _nms_fn = None
            _nms_error = f"{type(exc).__name__}: {exc}"
    return _nms_fn


def _decode_note() -> str:
    """Short, honest note for infer_bench's CSV ``extra`` field."""
    if _get_nms_fn() is None:
        return (
            "decode+NMS=no (ultralytics helper import failed: "
            f"{_nms_error}; RAW head tensors returned)"
        )
    return "decode+NMS=yes (ultralytics.utils.nms.non_max_suppression)"


def _postprocess(raw: Any, batch: int) -> list:
    """Turn raw YOLO head outputs into the contract's list of detections.

    ``raw`` is a list of CUDA float32 head tensors, each (B, 4+nc, H, W).
    The returned list has length ``batch``; each item is either
    ``(boxes[K,4] xyxy px, scores[K], classes[K])`` or ``None`` when the image
    has no detections. ``.detach().cpu()`` is part of what we time -- the D2H
    copy of the decoded results is a real cost of getting detections to Python.
    """
    tensors = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    nms = _get_nms_fn()
    if nms is None:
        # FALLBACK: return the network's literal output, already on the host.
        # These are feature-space xywh + logits, NOT xyxy pixel boxes, so this
        # path is flagged in the CSV. It is never a fabricated detection.
        return [t.detach().cpu() for t in tensors]

    dets = nms(tensors)  # list[B] of (N, 6): x1, y1, x2, y2, conf, cls
    out: list = []
    for det in dets[:batch]:
        if det is None or getattr(det, "numel", lambda: 0)() == 0:
            out.append(None)
            continue
        det = det.detach().cpu()
        boxes = det[:, :4]        # xyxy, pixels of the imgsz input
        scores = det[:, 4]        # objectness * class confidence
        classes = det[:, 5].long()  # integer class ids
        out.append((boxes, scores, classes))
    # Defensive: the helper always returns batch entries, but never break the
    # length-B contract if a future version changes that.
    while len(out) < batch:
        out.append(None)
    return out


def _shape_desc(result: list) -> str:
    """Compact output-shape string for the CSV ``extra`` column."""
    if not result:
        return "list[B=0]"
    first = result[0]
    if isinstance(first, tuple) and len(first) == 3:
        k = int(first[0].shape[0])
        return f"list[B={len(result)}] of (boxes[{k},4], scores[{k}], classes[{k}])"
    if hasattr(first, "shape"):
        dims = ",".join(str(d) for d in tuple(first.shape))
        return f"raw list[B={len(result)}] of ({dims})"
    return f"list[B={len(result)}] of {type(first).__name__}"


# ---------------------------------------------------------------------------
# Contract base class
# ---------------------------------------------------------------------------
class Backend:
    """Minimal interface every inference rung implements.

    Attributes
    ----------
    name : str
        Rung identifier, e.g. ``"pt_fp32"``.
    imgsz : int
        Input spatial size; every infer() input is (B, 3, imgsz, imgsz).
    """

    name: str = "base"
    imgsz: int = 640
    batch: int = 1

    # Additive bookkeeping used by infer_bench.py for the CSV ``extra`` field.
    output_shape: str = "<unknown>"
    decode_note: str = "decode+NMS=unknown"
    sync_note: str = "D2H=unknown"

    def infer(self, x) -> list:
        """Run one forward pass.

        Parameters
        ----------
        x : torch.Tensor
            (B, 3, imgsz, imgsz) float32 CUDA tensor, NCHW RGB, values already
            normalized to 0..1. Owned by the caller.

        Returns
        -------
        list
            Length B. Each item is a ``(boxes[K,4], scores[K], classes[K])``
            tuple or ``None``. Moving results to the host with
            ``detach().cpu()`` is intentionally part of the measured latency.
        """
        raise NotImplementedError

    def warmup(self, iters: int = 10) -> None:
        """Prime kernels / lazy init on a dummy input before timing."""
        raise NotImplementedError


class _PyTorchBackend(Backend):
    """PyTorch rung, shared by FP32 and FP16 (differ only in weight dtype)."""

    def __init__(
        self,
        name: str,
        model_path: str | None,
        imgsz: int,
        batch: int,
        fp16: bool,
    ) -> None:
        from ultralytics import YOLO

        self.name = name
        self.imgsz = imgsz
        self.batch = batch
        self.fp16 = fp16
        self.decode_note = _decode_note()
        self.sync_note = "D2H=included (detach().cpu())"

        weights = model_path or "yolov8n.pt"
        self.yolo = YOLO(weights)
        self.model = self.yolo.model  # underlying nn.Module, not the wrapper
        self.model.eval()
        self.model.to("cuda")
        if fp16:
            self.model.half()

    def infer(self, x) -> list:
        import torch

        with torch.inference_mode():
            if self.fp16:
                # Cast the input rather than using torch.autocast(): the entire
                # module is already .half(), so autocast would add
                # context-manager overhead inside the timed region without
                # changing a single kernel. One explicit cast is clearer.
                xin = x.half()
            else:
                xin = x
            # yolo.model(x) returns the raw YOLO head, e.g. 3 tensors of shape
            # (B, 4+nc, H, W); it is NOT decoded and we do not decode by hand.
            raw = self.model(xin)
            # NMS is part of getting usable detections, so it belongs inside
            # the measured (and inference-mode) region, not after it.
            result = _postprocess(raw, self.batch)
        self.output_shape = _shape_desc(result)
        return result

    def warmup(self, iters: int = 10) -> None:
        import torch

        x = torch.zeros(
            self.batch, 3, self.imgsz, self.imgsz, device="cuda", dtype=torch.float32
        )
        for _ in range(iters):
            self.infer(x)


class _ORTBackend(Backend):
    """ONNX Runtime rung with the CUDA execution provider (no silent fallback)."""

    def __init__(self, model_path: str | None, imgsz: int, batch: int) -> None:
        import numpy as np  # noqa: F401 - needed by bind_input's element_type
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "ONNX Runtime CUDA backend unavailable: 'CUDAExecutionProvider' "
                f"is not in get_available_providers() -> {available}. Install "
                "onnxruntime-gpu and match its CUDA version to torch's CUDA."
            )

        path = model_path or "models/yolov8n.onnx"
        if not os.path.isfile(path):
            raise RuntimeError(
                f"ONNX model not found at {path!r}. Run export_model.py first."
            )

        import torch  # already loaded by require_cuda(); local per import policy

        opts = ort.SessionOptions()
        # One intra-op thread avoids CPU-thread contention while the GPU runs;
        # on a small batch ORT can otherwise lose time to host scheduling.
        opts.intra_op_num_threads = 1
        # CRITICAL FOR HONEST TIMING: by default ORT's CUDA EP runs on its own
        # stream, so the torch.cuda.Event pair in common.time_call() (recorded
        # on torch's current stream) would NOT bracket ORT's kernels and would
        # under-report latency. Point the EP at torch's current stream. Older
        # ORT builds may reject the "stream" option, hence the fallback.
        stream_ptr = torch.cuda.current_stream().cuda_stream
        try:
            self.sess = ort.InferenceSession(
                path,
                sess_options=opts,
                providers=[
                    ("CUDAExecutionProvider", {"device_id": 0, "stream": stream_ptr})
                ],
            )
            self._stream_aligned = True
        except Exception:  # noqa: BLE001 - fall back to default stream
            self.sess = ort.InferenceSession(
                path, sess_options=opts, providers=["CUDAExecutionProvider"]
            )
            self._stream_aligned = False
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.batch = batch
        self.name = "ort_cuda"
        self.decode_note = _decode_note()
        self.sync_note = (
            "D2H=included (IOBinding CUDA input + copy_outputs_to_cpu, "
            "then H2D for NMS); "
            + (
                "stream=torch current (events valid)"
                if self._stream_aligned
                else "stream=ORT default (events may under-report)"
            )
        )
        self._np = np
        # IOBinding keeps the *input* on the GPU: we bind x.data_ptr() directly,
        # avoiding the x.cpu().numpy() round trip that would make ORT look
        # artificially slow. Outputs are copied to host then back for NMS.
        self._io = self.sess.io_binding()

    def infer(self, x) -> list:
        import torch

        self._io.bind_input(
            self.input_name,
            "cuda",
            0,
            self._np.float32,
            tuple(x.shape),
            x.data_ptr(),
        )
        for out in self.sess.get_outputs():
            self._io.bind_output(out.name, "cuda")
        self.sess.run_with_iobinding(self._io)
        outs_np = self._io.copy_outputs_to_cpu()
        raw = [
            torch.as_tensor(self._np.ascontiguousarray(a), device="cuda") for a in outs_np
        ]
        result = _postprocess(raw, self.batch)
        self.output_shape = _shape_desc(result)
        return result

    def warmup(self, iters: int = 10) -> None:
        import torch

        x = torch.zeros(
            self.batch, 3, self.imgsz, self.imgsz, device="cuda", dtype=torch.float32
        )
        for _ in range(iters):
            self.infer(x)


class _TRTBackend(Backend):
    """TensorRT rung. Load models/*.engine or build it from the ONNX export.

    This is the stretch goal. Because the TRT Python API moved between v8 and
    v10, get_backend() wraps the WHOLE construction in try/except and re-raises
    RuntimeError with the captured traceback text, so infer_bench.py can record
    the failure verbatim instead of a made-up latency.
    """

    def __init__(self, model_path: str | None, imgsz: int, batch: int) -> None:
        import tensorrt as trt
        import torch

        self.name = "trt_fp16"
        self.imgsz = imgsz
        self.batch = batch
        self.decode_note = _decode_note()
        self.sync_note = "D2H=none (results decoded from CUDA tensors)"
        self._torch = torch

        logger = trt.Logger(trt.Logger.WARNING)
        engine_path = model_path or "models/yolov8n.engine"
        onnx_path = os.path.splitext(engine_path)[0] + ".onnx"
        os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)

        if os.path.isfile(engine_path):
            with open(engine_path, "rb") as fh:
                engine_bytes = fh.read()
            self.loaded_from = "cached engine"
        else:
            if not os.path.isfile(onnx_path):
                raise RuntimeError(
                    f"Neither TensorRT engine ({engine_path}) nor ONNX "
                    f"({onnx_path}) exists. Run export_model.py first."
                )
            engine_bytes = self._build(trt, logger, onnx_path, imgsz, batch)
            with open(engine_path, "wb") as fh:
                fh.write(engine_bytes)
            self.loaded_from = "built from ONNX"

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("trt.Runtime.deserialize_cuda_engine returned None")
        context = engine.create_execution_context()
        self._engine = engine
        self._context = context

        major = int(trt.__version__.split(".")[0])
        self._trt10 = major >= 10
        shape = (batch, 3, imgsz, imgsz)
        if self._trt10:
            self._input_name = engine.get_tensor_name(0)
            self._output_names = [
                engine.get_tensor_name(i) for i in range(1, engine.num_io_tensors)
            ]
            context.set_input_shape(self._input_name, shape)
            out_shapes = [
                tuple(context.get_tensor_shape(n)) for n in self._output_names
            ]
        else:  # TensorRT 8.x -- binding-based API
            idxs = range(engine.num_bindings)
            self._input_name = next(
                engine.get_binding_name(i)
                for i in idxs
                if engine.binding_is_input(i)
            )
            self._output_names = [
                engine.get_binding_name(i)
                for i in range(engine.num_bindings)
                if not engine.binding_is_input(i)
            ]
            context.set_binding_shape(
                engine.get_binding_index(self._input_name), shape
            )
            out_shapes = [
                tuple(
                    context.get_binding_shape(engine.get_binding_index(n))
                )
                for n in self._output_names
            ]

        self._out_bufs = [
            torch.empty(sh, device="cuda", dtype=torch.float32) for sh in out_shapes
        ]
        self.output_shape = (
            "["
            + ", ".join("(" + ",".join(str(d) for d in sh) + ")" for sh in out_shapes)
            + "]"
        )

    @staticmethod
    def _build(trt: Any, logger: Any, onnx_path: str, imgsz: int, batch: int) -> bytes:
        """Build a serialized FP16 engine at the fixed batch/imgsz."""
        with trt.Builder(logger) as builder:
            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flag)
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, "rb") as fh:
                onnx_bytes = fh.read()
            if not parser.parse(onnx_bytes):
                msgs = "\n".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise RuntimeError("TensorRT ONNX parse failed:\n" + msgs)

            config = builder.create_builder_config()
            try:  # TRT >= 8 uses memory pools; TRT <= 7 used max_workspace_size
                config.set_memory_pool_limit(
                    trt.MemoryPoolType.WORKSPACE, 4 << 30  # 4 GiB
                )
            except AttributeError:  # pragma: no cover - very old TRT
                config.max_workspace_size = 4 << 30
            # Request FP16 explicitly: on T4 this is what unlocks tensor cores.
            config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            shape = (batch, 3, imgsz, imgsz)
            profile.set_shape(network.get_input(0).name, shape, shape, shape)
            config.add_optimization_profile(profile)

            blob = builder.build_serialized_network(network, config)
            if blob is None:
                raise RuntimeError("TensorRT build_serialized_network returned None")
            return bytes(blob)

    def infer(self, x) -> list:
        torch = self._torch  # bound in __init__, avoids a module-level import
        ctx = self._context
        if self._trt10:
            ctx.set_tensor_address(self._input_name, x.data_ptr())
            for name, buf in zip(self._output_names, self._out_bufs):
                ctx.set_tensor_address(name, buf.data_ptr())
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        else:
            bindings = [x.data_ptr()] + [b.data_ptr() for b in self._out_bufs]
            ctx.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        # TRT outputs stay on the GPU; _postprocess brings detections to host.
        result = _postprocess(list(self._out_bufs), self.batch)
        self.output_shape = _shape_desc(result)
        return result

    def warmup(self, iters: int = 10) -> None:
        import torch

        x = torch.zeros(
            self.batch, 3, self.imgsz, self.imgsz, device="cuda", dtype=torch.float32
        )
        for _ in range(iters):
            self.infer(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_backend(
    name: str,
    model_path: str | None = None,
    imgsz: int = 640,
    batch: int = 1,
) -> Backend:
    """Construct a rung by name.

    Raises
    ------
    RuntimeError
        With an actionable message when CUDA is missing, the name is unknown,
        or an optional dependency (onnxruntime CUDA / tensorrt) is unavailable.
        TensorRT failures include the captured traceback verbatim.
    """
    require_cuda()
    key = str(name).strip().lower()
    if key not in BACKEND_NAMES:
        raise RuntimeError(
            f"Unknown backend {name!r}. Choose from: {', '.join(BACKEND_NAMES)}."
        )

    # Normalize EVERY construction failure to RuntimeError with the captured
    # traceback. infer_bench.py catches RuntimeError and turns it into a SKIP;
    # a bare ModuleNotFoundError (ultralytics / tensorrt not installed) would
    # otherwise crash the whole run instead of being recorded. For trt_fp16 in
    # particular this preserves the full traceback text verbatim.
    try:
        if key in ("pt_fp32", "pt_fp16"):
            return _PyTorchBackend(
                key, model_path, imgsz, batch, fp16=(key == "pt_fp16")
            )
        if key == "ort_cuda":
            return _ORTBackend(model_path, imgsz, batch)
        return _TRTBackend(model_path, imgsz, batch)  # trt_fp16
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - unavailable rung, not our bug
        raise RuntimeError(
            f"{key} backend unavailable ({type(exc).__name__}):\n"
            f"{traceback.format_exc()}"
        ) from exc
