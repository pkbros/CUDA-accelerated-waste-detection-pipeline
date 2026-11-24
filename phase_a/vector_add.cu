// ===========================================================================
// vector_add.cu -- Phase A sanity benchmark: C[i] = A[i] + B[i]
//
// WHAT THIS FILE MEASURES
//   Three implementations of the same elementwise add over N floats:
//     cpu_reference : single-threaded host loop (the correctness baseline),
//                     timed with std::chrono::steady_clock because CUDA events
//                     can only timestamp work submitted to the GPU.
//     gpu_scalar    : one CUDA thread per element, 1-D grid of --block threads.
//     gpu_float4    : one thread per FOUR elements using 16-byte float4 loads
//                     and stores, plus a scalar tail for N % 4 leftovers.
//   All three report bytes/s using the same traffic count: 3 * N * 4 bytes
//   (read A, read B, write C). For the CPU row that number is host DRAM
//   bandwidth; it is kept on the same axis so the comparison is honest.
//
// WHY OCCUPANCY IS NOT THE LIMITER HERE
//   Each output element needs ONE floating-point add but moves 12 bytes
//   (4 in from A, 4 in from B, 4 out to C). The arithmetic intensity is
//   therefore about 1/12 add per byte: the arithmetic is finished long before
//   the DRAM can deliver the next cache line. Filling the SMs with more
//   resident warps cannot change that ratio -- it only helps hide latency for
//   compute-bound kernels with many operations per loaded byte. What matters
//   here is memory bandwidth and memory-level parallelism (many loads in
//   flight), so the interesting optimization is how wide each thread's memory
//   access is, not how many blocks we launch.
//
// WHAT THE FLOAT4 VARIANT CHANGES
//   Four scalar threads issue four 32-bit loads from A, four from B and four
//   32-bit stores to C -- twelve narrow accesses for four results. Four
//   float4 threads issue ONE 128-bit load from A, ONE from B and ONE 128-bit
//   store to C each:
//     - fewer instructions (3 wide accesses instead of 12 narrow ones),
//     - fewer memory transactions (a warp requests 512 B per array in one
//       coalesced sweep instead of 128 B per instruction across four issues),
//     - better MLP (more independent bytes requested per issued instruction).
//   The byte count does not change -- we move exactly the same 12 bytes per
//   element -- we just move them with fewer, wider requests.
//
// Correctness: the CPU reference is computed once and every GPU variant is
// copied back and checked element-by-element with compare_gpu_cpu before any
// row is written. A fast wrong kernel is not a result; it aborts the run.
// ===========================================================================

#include "bench_common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 1. Small wall-clock helper for the CPU reference.
//    CudaTimer uses GPU event timestamps and would measure nothing useful
//    around a host loop, so the CPU row needs its own steady_clock timer.
// ---------------------------------------------------------------------------
class WallClockTimer {
 public:
  void start() { start_ = std::chrono::steady_clock::now(); }
  void stop() { stop_ = std::chrono::steady_clock::now(); }

  double ms() const {
    return std::chrono::duration<double, std::milli>(stop_ - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_{};
  std::chrono::steady_clock::time_point stop_{};
};

// The single-threaded host reference. Kept deliberately plain: this is the
// definition of "correct" that the GPU results are compared against.
void cpu_vector_add(const float* a, const float* b, float* c, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

// ---------------------------------------------------------------------------
// 2. GPU kernels. Both use __restrict__ to tell the compiler the three
//    buffers never alias, which is what lets it keep loads/stores wide.
// ---------------------------------------------------------------------------

// One thread = one element. This is the straightforward, readable baseline.
__global__ void vector_add_scalar(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  size_t n) {
  // Which contiguous chunk of the array this block owns.
  const size_t block_start = static_cast<size_t>(blockIdx.x) * blockDim.x;
  // This thread's offset inside its block (0 .. blockDim.x-1).
  const size_t i = block_start + threadIdx.x;

  // The final block may run past N when N is not a multiple of blockDim.x.
  if (i >= n) return;

  // Four bytes in from A, four from B, one add, four bytes out to C.
  c[i] = a[i] + b[i];
}

// One thread = four consecutive elements (one float4 = 16 bytes).
__global__ void vector_add_float4(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  size_t n) {
  // Number of complete groups of four elements.
  const size_t vec_n = n / 4;

  // Reinterpret the flat float buffers as float4 buffers. cudaMalloc returns
  // 256-byte-aligned memory, so every float4 address used below is 16-byte
  // aligned -- the requirement for a single 128-bit LDG/STG instruction.
  const float4* a4 = reinterpret_cast<const float4*>(a);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  float4* c4 = reinterpret_cast<float4*>(c);

  // Global thread index: one thread now owns one float4, not one float.
  const size_t v = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (v < vec_n) {               // bounds check for the last block
    const float4 av = a4[v];     // ONE 128-bit load from A
    const float4 bv = b4[v];     // ONE 128-bit load from B
    float4 cv;                   // four sums assembled component by component
    cv.x = av.x + bv.x;          // output element 4*v + 0
    cv.y = av.y + bv.y;          // output element 4*v + 1
    cv.z = av.z + bv.z;          // output element 4*v + 2
    cv.w = av.w + bv.w;          // output element 4*v + 3
    c4[v] = cv;                  // ONE 128-bit store to C
  }

  // Tail for N % 4: the scalar index of the first leftover element is vec_n*4.
  // Adding the global thread index means only the first <= 3 threads of block
  // zero satisfy i < n, so no element is written twice and no extra launch is
  // needed. This is what makes the fast path correct for any N.
  const size_t i = vec_n * 4 + v;
  if (i < n) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// 3. Correctness gate + row emission, shared by both GPU variants.
//    Printing happens before the abort so a failure is visible in the log;
//    the CSV row is written only when the kernel is numerically correct.
// ---------------------------------------------------------------------------
bool verify_and_emit(const std::string& variant,
                     const std::vector<float>& gpu_result,
                     const std::vector<float>& cpu_result,
                     size_t n,
                     double rtol,
                     double atol,
                     const std::string& csv_path,
                     const Stats& stats,
                     double throughput) {
  const MaxError e = compare_gpu_cpu(gpu_result.data(), cpu_result.data(), n,
                                     rtol, atol);
  std::printf("  [verify] %-13s max_abs_err=%.9g, rel_err=%.9g, ok=%s\n",
              variant.c_str(), e.abs_err, e.rel_err, e.ok ? "true" : "false");

  char extra[128];
  std::snprintf(extra, sizeof(extra), "max_abs_err=%.9g, ok=%s", e.abs_err,
                e.ok ? "true" : "false");

  if (!e.ok) {
    std::fprintf(stderr,
                 "ERROR: %s is numerically wrong (max_abs_err=%.9g, "
                 "rel_err=%.9g, rtol=%g, atol=%g). Aborting: a fast wrong "
                 "kernel is not a benchmark result.\n",
                 variant.c_str(), e.abs_err, e.rel_err, rtol, atol);
    return false;
  }

  print_row(variant, stats, "GB/s", throughput);
  csv_emit(csv_path, "vector_add", variant, static_cast<double>(n), "elements",
           stats, throughput, "GB/s", extra);
  return true;
}

// ---------------------------------------------------------------------------
// 4. main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  // ---- CLI ---------------------------------------------------------------
  const long long n_arg = arg_ll(argc, argv, "n", 50000000LL);
  const int iters = arg_int(argc, argv, "iters", 50);
  const int warmup = arg_int(argc, argv, "warmup", 10);
  const int block = arg_int(argc, argv, "block", 256);
  const std::string csv_path =
      arg_value(argc, argv, "csv", "../results/vector_add.csv");
  const double rtol = arg_double(argc, argv, "tol", 1e-5);
  // The CLI tolerance is the *relative* one. A fixed tiny absolute tolerance
  // is also applied; inputs are small exact decimal fractions, so any real
  // difference shows up in the relative check as well.
  const double atol = 1e-6;

  if (n_arg <= 0 || iters <= 0 || warmup < 0 || block <= 0 || block > 1024 ||
      !(rtol > 0.0)) {
    std::fprintf(stderr,
                 "bad arguments: need --n>0 --iters>0 --warmup>=0 "
                 "0<--block<=1024 --tol>0\n");
    return EXIT_FAILURE;
  }

  const size_t N = static_cast<size_t>(n_arg);
  const size_t bytes = N * sizeof(float);
  const size_t block_sz = static_cast<size_t>(block);
  const size_t grid_scalar = (N + block_sz - 1) / block_sz;
  const size_t vec_n = N / 4;
  // With N < 4 there are no vector elements, but we still need one block so
  // that the scalar tail branch inside the kernel can run.
  const size_t grid_vec = vec_n == 0 ? 1 : (vec_n + block_sz - 1) / block_sz;

  if (grid_scalar > 2147483647ULL || grid_vec > 2147483647ULL) {
    std::fprintf(stderr, "bad arguments: grid exceeds CUDA's 2^31-1 limit\n");
    return EXIT_FAILURE;
  }

  const dim3 threads(static_cast<unsigned int>(block), 1, 1);
  const dim3 grid_s(static_cast<unsigned int>(grid_scalar), 1, 1);
  const dim3 grid_v(static_cast<unsigned int>(grid_vec), 1, 1);

  // Bytes moved per variant: read A + read B + write C. This is a structural
  // count from the algorithm, not an estimate of the measured bandwidth.
  const double traffic = 3.0 * static_cast<double>(bytes);

  const DeviceInfo& dev = device_info();
  std::printf("vector_add: N=%zu floats (%.1f MiB/array), iters=%d, "
              "warmup=%d, block=%d, rtol=%g\n",
              N, static_cast<double>(bytes) / (1024.0 * 1024.0), iters,
              warmup, block, rtol);
  std::printf("device: %s (%s)\n", dev.name.c_str(), dev.arch.c_str());

  // ---- Host data ---------------------------------------------------------
  // Deterministic pseudo-random-style patterns: no rand(), so the inputs are
  // identical on every machine and the reference is reproducible.
  std::vector<float> hA(N), hB(N), hCpu(N), hGpu(N);
  for (size_t i = 0; i < N; ++i) {
    hA[i] = static_cast<float>(i % 1000) * 0.001f;
    hB[i] = static_cast<float>((i * 7 + 13) % 1000) * 0.002f;
  }

  // ---- Device data -------------------------------------------------------
  float* dA = nullptr;
  float* dB = nullptr;
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), bytes));
  // One-time H2D copies, outside the timed region: we measure the kernels,
  // not PCIe transfer. The data stays resident for all iterations.
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK_LAST();

  // ---- Variant 1: CPU reference (wall clock) -----------------------------
  WallClockTimer cpu_timer;
  std::vector<double> cpu_samples;
  cpu_samples.reserve(static_cast<size_t>(iters));
  // Volatile sink: reading one result after each call makes the repeated work
  // observable, so the optimiser cannot fold the timed iterations away.
  volatile double cpu_sink = 0.0;
  for (int i = 0; i < warmup; ++i) {
    cpu_vector_add(hA.data(), hB.data(), hCpu.data(), N);
    cpu_sink += hCpu[N / 2];
  }
  for (int i = 0; i < iters; ++i) {
    cpu_timer.start();
    cpu_vector_add(hA.data(), hB.data(), hCpu.data(), N);
    cpu_timer.stop();
    // Record the sample. Without this push_back the vector stays empty, and
    // stats_of() returns n=0 with a mean of 0 -- a CSV row that claims a
    // measurement which never happened.
    cpu_samples.push_back(cpu_timer.ms());
    cpu_sink += hCpu[N / 2];  // deliberately after stop(): not part of timing
  }
  const Stats cpu_stats = stats_of(cpu_samples);
  // Same byte count as the GPU rows: this is host DRAM bandwidth (and cache
  // effects), useful as a bandwidth axis but not a GPU speedup denominator.
  const double cpu_tp = gbs(traffic, cpu_stats.mean);
  print_row("cpu_reference", cpu_stats, "GB/s", cpu_tp);
  csv_emit(csv_path, "vector_add", "cpu_reference",
           static_cast<double>(N), "elements", cpu_stats, cpu_tp, "GB/s",
           "reference=cpu_single_thread");

  // ---- Variant 2: gpu_scalar ---------------------------------------------
  // Each lambda only *enqueues* one kernel; time_iters wraps the event pair
  // and is responsible for warmup and per-iteration synchronisation.
  const Stats scalar_stats = time_iters(
      [&]() {
        vector_add_scalar<<<grid_s, threads>>>(dA, dB, dC, N);
        CUDA_CHECK_LAST();  // report launch errors after every launch
      },
      warmup, iters);
  // D2H copy is outside the timed region: we time the kernel, not the copy.
  CUDA_CHECK(cudaMemcpy(hGpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
  const double scalar_tp = gbs(traffic, scalar_stats.mean);
  if (!verify_and_emit("gpu_scalar", hGpu, hCpu, N, rtol, atol, csv_path,
                       scalar_stats, scalar_tp)) {
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return EXIT_FAILURE;
  }

  // ---- Variant 3: gpu_float4 ---------------------------------------------
  const Stats float4_stats = time_iters(
      [&]() {
        vector_add_float4<<<grid_v, threads>>>(dA, dB, dC, N);
        CUDA_CHECK_LAST();  // report launch errors after every launch
      },
      warmup, iters);
  CUDA_CHECK(cudaMemcpy(hGpu.data(), dC, bytes, cudaMemcpyDeviceToHost));
  const double float4_tp = gbs(traffic, float4_stats.mean);
  if (!verify_and_emit("gpu_float4", hGpu, hCpu, N, rtol, atol, csv_path,
                       float4_stats, float4_tp)) {
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return EXIT_FAILURE;
  }

  // ---- Teardown ----------------------------------------------------------
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  std::printf("wrote 3 measured rows to %s\n", csv_path.c_str());
  return EXIT_SUCCESS;
}
