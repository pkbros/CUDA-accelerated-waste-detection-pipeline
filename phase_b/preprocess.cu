// ===========================================================================
// preprocess.cu -- ONE fused kernel for the whole Phase B preprocessing stage.
//
// WHY FUSE (the design this write-up will defend):
//   The naive pipeline is five separate passes over a 640x640 image:
//     1. letterbox resize (uint8 -> uint8, 640x640x3 intermediate)
//     2. BGR -> RGB          (uint8 -> uint8, another full-size intermediate)
//     3. HWC -> CHW          (uint8 -> uint8, another full-size intermediate)
//     4. uint8 -> FP32       (uint8 -> float, 4x the bytes)
//     5. normalize          (float -> float, another full-size pass)
//   Each pass reads AND writes 640*640*3 bytes (or 4x that for FP32), so the
//   naive version moves roughly 10 full-image volumes through DRAM and launches
//   four extra kernels. Here we do ONE pass over the DESTINATION pixels: each
//   thread reads its four source corners (the only global reads) and writes its
//   three CHW outputs. No full-size intermediate ever exists in global memory.
//
//   The arithmetic per output value is ~10 flops (2 lerps + normalize), while
//   we touch ~3 source bytes per channel. At 640x640 that is a few million
//   flops but several MB of traffic, i.e. the ratio is far below any GPU's
//   flops/byte. This kernel is MEMORY-BOUND, not compute-bound: the win comes
//   from eliminating traffic and launches, not from clever math.
//
//   Geometry (scale / pad_x / pad_y) is computed ON THE HOST by Python and
//   passed in. It must be bit-identical to the CPU reference, so the kernel
//   never re-derives it.
// ===========================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cmath>    // floorf / fmaf
#include <vector>

#include <torch/extension.h>
// c10::cuda::getCurrentCUDAStream() lives here (in recent PyTorch it moved out
// of at::cuda). CUDAStream has an explicit .stream() -> cudaStream_t converter.
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>  // header for CUDA exceptions; we use TORCH_CHECK
#include <pybind11/stl.h>            // std::vector<double> <-> Python list

// 1/255 as a single-precision constant: folding the uint8->float conversion
// into one multiply is cheaper than a divide and is accurate to float eps.
static constexpr float kInv255 = 1.0f / 255.0f;

// Small POD passed BY VALUE to the kernel: kernel arguments live in constant
// memory, so every thread in the block reads the same 6 floats for free.
//   mean[c]   is the per-channel mean in [0,1] space,
//   invstd[c] is 1/std[c] (host-precomputed so the kernel never divides).
// The effective normalization is therefore
//   v = (pixel * (1/255) - mean[c]) * invstd[c]  ==  (pixel/255 - mean[c])/std[c].
struct NormParams {
  float mean[3];
  float invstd[3];
};

// Tiny branchless clamp that needs no CUDA header magic and is obvious to a
// learner: keep v inside [lo, hi].
__device__ __forceinline__ int clamp_i(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Convert a fully-computed float to the requested output type. Using a trait
// instead of `static_cast<T>` keeps the FP16 rounding explicit and does not
// depend on the compiler's half-conversion flags. ALL upstream maths is float;
// we convert to half only at the final store (FP16 arithmetic would throw away
// precision for no speed gain on this memory-bound kernel).
template <typename T>
struct StoreConvert;

template <>
struct StoreConvert<float> {
  __device__ __forceinline__ static float run(float v) { return v; }
};

template <>
struct StoreConvert<__half> {
  __device__ __forceinline__ static __half run(float v) { return __float2half(v); }
};

// ---------------------------------------------------------------------------
// The shared sampling body. One thread calls it for one output pixel; it loops
// over the three channels itself. Reasons:
//   * the four source offsets and the two lerp weights are computed ONCE per
//     pixel instead of three times;
//   * the three stores land in three separate CHW planes, so a warp produces
//     three coalesced ~128-byte streams (32 consecutive dx per warp) rather
//     than interleaving channels into one stream.
// `__restrict__` promises the compiler that src and dst never alias, so it may
// keep loads in registers and reorder stores without worrying about a write to
// dst changing a later read of src.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void bilinear_norm_store(
    const uint8_t* __restrict__ src,  // (src_h, src_w, 3), BGR, contiguous
    T* __restrict__ dst,              // (3, dst_h, dst_w), RGB CHW, contiguous
    int src_h, int src_w,
    int dst_h, int dst_w,
    float sx, float sy,               // source coords, half-pixel centres
    NormParams norm,
    int out_xy) {                     // dy * dst_w + dx

  // Floor gives the top-left corner; the fraction is the bilinear weight.
  const int x0f = static_cast<int>(floorf(sx));
  const int y0f = static_cast<int>(floorf(sy));
  const float wx = sx - static_cast<float>(x0f);
  const float wy = sy - static_cast<float>(y0f);

  // Edge clamping: with half-pixel centres the last valid sample can ask for
  // x0+1 == src_w, which must fold back onto src_w-1 (mirror-free clamp).
  const int x0 = clamp_i(x0f, 0, src_w - 1);
  const int x1 = clamp_i(x0f + 1, 0, src_w - 1);
  const int y0 = clamp_i(y0f, 0, src_h - 1);
  const int y1 = clamp_i(y0f + 1, 0, src_h - 1);

  // Byte offsets of the four corners. *3 because the source is HWC with 3
  // channels; consecutive threads (consecutive dx) read consecutive triples,
  // which is what makes the gather coalesced.
  const int base00 = (y0 * src_w + x0) * 3;
  const int base01 = (y0 * src_w + x1) * 3;
  const int base10 = (y1 * src_w + x0) * 3;
  const int base11 = (y1 * src_w + x1) * 3;
  const int plane = dst_h * dst_w;

  // Unroll so the compile-time-known channel index lets the compiler issue the
  // three byte loads of each corner back-to-back. The BGR->RGB reorder is just
  // a register rename: output c reads source channel sc = 2 - c.
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    const int sc = 2 - c;  // c=0<-R(2), c=1<-G(1), c=2<-B(0)

    // The three adjacent byte loads for ONE corner are the "same load" group:
    // they hit the same cache line and the reorder costs no extra memory op.
    const float v00 = static_cast<float>(src[base00 + sc]);
    const float v01 = static_cast<float>(src[base01 + sc]);
    const float v10 = static_cast<float>(src[base10 + sc]);
    const float v11 = static_cast<float>(src[base11 + sc]);

    // Two lerps in float: horizontal along the top/bottom rows, then vertical.
    const float top = fmaf(wx, v01 - v00, v00);
    const float bot = fmaf(wx, v11 - v10, v10);
    const float v = fmaf(wy, bot - top, top);

    // Normalize in FLOAT even on the FP16 path (see StoreConvert comment).
    const float centred = v * kInv255 - norm.mean[c];
    const float normed = centred * norm.invstd[c];

    // CHW destination: channel plane c, then row-major (dy, dx).
    dst[c * plane + out_xy] = StoreConvert<T>::run(normed);
  }
}

// ---------------------------------------------------------------------------
// One thread per destination pixel. A 2-D block over (dst_w, dst_h) means
// neighbouring threads walk neighbouring dx, hence neighbouring source pixels
// -> coalesced bilinear loads. The warp is split across at most two source
// rows, which is fine because the four corner pointers were computed above.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src,
    T* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale, float pad_x, float pad_y,
    NormParams norm) {
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;

  // Guard the tail of the grid when dst_w/dst_h are not multiples of the block.
  if (dx >= dst_w || dy >= dst_h) return;

  // Map destination pixel centre back to the SOURCE image. The +0.5 is the
  // half-pixel centre; -0.5 afterwards makes scale=1/pad=0 the identity, and
  // the whole formula is exactly the inverse of cv2.INTER_LINEAR's
  // src = (dst + 0.5)/scale - 0.5 convention.
  //
  // This coordinate arithmetic is deliberately FLOAT, not double: the host
  // passes scale/pad_x/pad_y already cast to float, and the CPU reference uses
  // the same float32 rounding so floor() picks the same source corner. Doing
  // it in double would (a) disagree with the float32 reference near integer
  // boundaries and (b) be slow on a T4, whose FP64 throughput is tiny. All
  // interpolation and normalization below is float too; only the final store
  // converts to half.
  const float sx = (static_cast<float>(dx) + 0.5f - pad_x) / scale - 0.5f;
  const float sy = (static_cast<float>(dy) + 0.5f - pad_y) / scale - 0.5f;

  const int out_xy = dy * dst_w + dx;
  const int plane = dst_h * dst_w;

  // Letterbox padding: when the mapped source point is outside the real image,
  // write ZERO. In normalized CHW space zero corresponds to the neutral grey
  // border a detector expects; the uint8 letterbox canvas uses pad_value=114,
  // which is the same border expressed before normalization. Writing zero here
  // avoids sampling/clamping the image edge into the pad region.
  if (sx < 0.0f || sx >= static_cast<float>(src_w) ||
      sy < 0.0f || sy >= static_cast<float>(src_h)) {
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
      dst[c * plane + out_xy] = StoreConvert<T>::run(0.0f);
    }
    return;
  }

  bilinear_norm_store<T>(src, dst, src_h, src_w, dst_h, dst_w,
                         sx, sy, norm, out_xy);
}

// ---------------------------------------------------------------------------
// Plain launchers. Kept as ordinary C++ functions so they could be called from
// a non-pybind harness; the pybind wrappers below only add tensor checks.
// ---------------------------------------------------------------------------
template <typename T>
static void launch_preprocess(
    const uint8_t* __restrict__ src,
    T* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale, float pad_x, float pad_y,
    const float mean[3], const float invstd[3],
    cudaStream_t stream) {
  // "-1 safe": never launch a zero/negative-sized grid or divide by scale=0.
  if (src_h <= 0 || src_w <= 0 || dst_h <= 0 || dst_w <= 0 || !(scale > 0.0f)) {
    return;
  }

  NormParams norm;
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    norm.mean[c] = mean[c];
    norm.invstd[c] = invstd[c];
  }

  // 32x8 = 256 threads/block: one warp covers 32 adjacent output columns.
  const dim3 block(32, 8);
  const dim3 grid((dst_w + block.x - 1) / block.x,
                  (dst_h + block.y - 1) / block.y);
  preprocess_kernel<T><<<grid, block, 0, stream>>>(
      src, dst, src_h, src_w, dst_h, dst_w, scale, pad_x, pad_y, norm);
}

void preprocess_u8hwc_to_f32chw(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale, float pad_x, float pad_y,
    const float mean[3], const float invstd[3],
    cudaStream_t stream) {
  launch_preprocess<float>(src, dst, src_h, src_w, dst_h, dst_w,
                           scale, pad_x, pad_y, mean, invstd, stream);
}

void preprocess_u8hwc_to_f16chw(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    int src_h, int src_w,
    int dst_h, int dst_w,
    float scale, float pad_x, float pad_y,
    const float mean[3], const float invstd[3],
    cudaStream_t stream) {
  launch_preprocess<__half>(src, dst, src_h, src_w, dst_h, dst_w,
                            scale, pad_x, pad_y, mean, invstd, stream);
}

// ---------------------------------------------------------------------------
// pybind wrappers: take torch::Tensor directly (cleaner than uintptr_t) and
// validate everything the kernel assumes about layout.
// ---------------------------------------------------------------------------
namespace {

void check_common(torch::Tensor src, torch::Tensor dst,
                  int64_t dst_h, int64_t dst_w,
                  const std::vector<double>& mean,
                  const std::vector<double>& stddev,
                  double scale) {
  TORCH_CHECK(src.device().is_cuda(), "preprocess: src must be a CUDA tensor");
  TORCH_CHECK(src.dtype() == torch::kUInt8, "preprocess: src must be uint8");
  TORCH_CHECK(src.dim() == 3 && src.size(2) == 3,
              "preprocess: src must be HWC with exactly 3 channels (BGR)");
  TORCH_CHECK(src.is_contiguous(), "preprocess: src must be contiguous");
  TORCH_CHECK(src.size(0) > 0 && src.size(1) > 0,
              "preprocess: src must have positive height and width");

  TORCH_CHECK(dst.device().is_cuda() && dst.device() == src.device(),
              "preprocess: dst must be on the same CUDA device as src");
  TORCH_CHECK(dst.is_contiguous(), "preprocess: dst must be contiguous");
  TORCH_CHECK(dst.dim() == 3 && dst.size(0) == 3 &&
                  dst.size(1) == dst_h && dst.size(2) == dst_w,
              "preprocess: dst must be CHW with shape (3, dst_h, dst_w)");

  TORCH_CHECK(mean.size() == 3 && stddev.size() == 3,
              "preprocess: mean and std must each have 3 values");
  TORCH_CHECK(scale > 0.0, "preprocess: scale must be > 0");
  for (int c = 0; c < 3; ++c) {
    TORCH_CHECK(stddev[c] > 0.0, "preprocess: std values must be > 0");
  }
}

}  // namespace

void preprocess_u8hwc_to_f32chw_wrapper(
    torch::Tensor src, torch::Tensor dst,
    int64_t dst_h, int64_t dst_w,
    double scale, double pad_x, double pad_y,
    std::vector<double> mean, std::vector<double> stddev) {
  check_common(src, dst, dst_h, dst_w, mean, stddev, scale);
  TORCH_CHECK(dst.dtype() == torch::kFloat32, "preprocess_f32: dst must be float32");

  float m[3], iv[3];
  for (int c = 0; c < 3; ++c) {
    m[c] = static_cast<float>(mean[c]);
    iv[c] = static_cast<float>(1.0 / stddev[c]);
  }

  // Use PyTorch's current stream so cudaEvent timing in Python observes the
  // kernel on the same stream it records its events on.
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  preprocess_u8hwc_to_f32chw(
      src.data_ptr<uint8_t>(), dst.data_ptr<float>(),
      static_cast<int>(src.size(0)), static_cast<int>(src.size(1)),
      static_cast<int>(dst_h), static_cast<int>(dst_w),
      static_cast<float>(scale), static_cast<float>(pad_x),
      static_cast<float>(pad_y), m, iv, stream);

  // A bad grid or an illegal address surfaces here, not as a mysterious slow
  // kernel or a garbage result later on.
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "preprocess_f32 launch failed: ",
              cudaGetErrorString(err));
}

void preprocess_u8hwc_to_f16chw_wrapper(
    torch::Tensor src, torch::Tensor dst,
    int64_t dst_h, int64_t dst_w,
    double scale, double pad_x, double pad_y,
    std::vector<double> mean, std::vector<double> stddev) {
  check_common(src, dst, dst_h, dst_w, mean, stddev, scale);
  TORCH_CHECK(dst.dtype() == torch::kHalf, "preprocess_f16: dst must be float16");

  float m[3], iv[3];
  for (int c = 0; c < 3; ++c) {
    m[c] = static_cast<float>(mean[c]);
    iv[c] = static_cast<float>(1.0 / stddev[c]);
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  // at::Half has the same 16-bit layout as __half; the cast only changes the
  // C++ type so the store uses __float2half.
  preprocess_u8hwc_to_f16chw(
      src.data_ptr<uint8_t>(),
      reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
      static_cast<int>(src.size(0)), static_cast<int>(src.size(1)),
      static_cast<int>(dst_h), static_cast<int>(dst_w),
      static_cast<float>(scale), static_cast<float>(pad_x),
      static_cast<float>(pad_y), m, iv, stream);

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "preprocess_f16 launch failed: ",
              cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_f32", &preprocess_u8hwc_to_f32chw_wrapper,
        "Fused letterbox + BGR2RGB + normalize + HWC2CHW + u8->FP32 "
        "(src: u8 HWC CUDA, dst: preallocated float32 CHW CUDA, "
        "dst_h, dst_w, scale, pad_x, pad_y, mean[3], std[3])");
  m.def("preprocess_f16", &preprocess_u8hwc_to_f16chw_wrapper,
        "Fused letterbox + BGR2RGB + normalize + HWC2CHW + u8->FP16 "
        "(same arguments; dst is preallocated float16 CHW CUDA)");
}
