// ===========================================================================
// reduction.cu -- array sum on the GPU, four progressively better kernels
//                 plus the CUB "professional" baseline.
//
// WHAT A REDUCTION IS
// -------------------
// Summing N floats is a *tree* problem: pair up neighbours, sum each pair,
// then pair up the results, and repeat until one number is left. The tree has
// log2(N) levels, so a correct parallel reduction needs ~log2(N) steps, not N.
// Every variant below computes the SAME mathematical sum but reaches it with a
// different communication pattern; the point of the benchmark is to show how
// much the communication pattern -- not the arithmetic -- costs.
//
// WHY naive_global IS SLOW
// ------------------------
// It launches ONE block. Only blockDim.x threads (default 256) ever exist, so
// the other ~16M elements are folded into those threads by a strided loop, and
// then the log2(blockDim.x) tree steps write their partial sums back to GLOBAL
// memory. Three problems:
//   (a) every tree step is a global store followed (next step) by a global
//       load -- a full round trip to DRAM per level, ~log2(256)=8 of them;
//   (b) one block = one SM, so occupancy is effectively 1 block and the GPU's
//       other SMs sit idle;
//   (c) __syncthreads() at every level stalls all 256 threads, including the
//       ones already finished at the "tail" (tid >= s). Once s < 32 the active
//       lanes are a subset of one warp, which is the divergence the shuffle
//       version eliminates.
// __syncthreads() is also doing a necessary job here: the CUDA memory model
// guarantees that global writes made before the barrier are visible to every
// thread in the block after it. Without that fence the next level could read
// a stale partial.
//
// WHAT shared_mem FIXES
// ---------------------
// Many blocks (G) each reduce a slice. The per-thread partials live in shared
// memory, which is on-chip and ~100x lower latency than DRAM, so the 8 tree
// steps no longer leave the SM. Only one value per block goes to global memory.
// A second kernel reduces those G partials. The 32-thread boundary matters
// because for s <= 32 only the first warp is active: warp lanes execute in
// lockstep, so they see each other's shared writes without a barrier -- the
// __syncthreads() calls in those last 5 steps are redundant, and that is exactly
// what warp_shuffle exploits.
//
// WHAT warp_shuffle REMOVES
// -------------------------
// __shfl_down_sync(0xffffffff, v, offset) reads a register directly from a
// lane offset away in the same warp. No shared memory at all for the intra-warp
// steps, hence no __syncthreads() inside a warp and no bank conflicts. Each
// warp's lane 0 drops one partial into shared memory, and warp 0 reduces those
// partials with the same 5 shuffles. This is the canonical reduction.
//
// CORRECTNESS / TOLERANCE
// -----------------------
// h_data[i] = 1 + (i % 8) * 0.125. Every element is exactly representable in
// float, one 8-element period sums to exactly 11.5, so the true answer is
// analytic_sum(N) = (N/8)*11.5 + remainder. The CPU reference deliberately
// accumulates in DOUBLE: double holds every integer up to 2^53 exactly, so for
// this pattern the CPU sum is exact while the GPU float sums are not -- once a
// running float partial passes 2^24 its ULP is 2, and small addends get rounded
// away. That is precisely why we need a tolerance (default --tol = 1e-4):
// relative error dominates for a ~2.4e7 sum, while the absolute term 1e-4 is
// sub-ULP and only matters for very small N. See the --tol printout for the
// exact numbers used in a run.
//
// Build: part of phase_a/Makefile -> ../build/reduction
// ===========================================================================

#include "bench_common.h"

#include <chrono>
#include <climits>
#include <cstdint>

// CUB 2.x is header-only and ships with the CUDA toolkit, so no extra library
// is needed. The check is automatic: a toolkit without CUB headers (or a manual
// -DREDUCTION_NO_CUB) skips this variant with a printed reason rather than
// failing the whole build. The other four variants are unaffected.
#ifndef REDUCTION_NO_CUB
#if defined(__has_include)
#if __has_include(<cub/cub.cuh>)
#include <cub/cub.cuh>
#define HAVE_CUB 1
#endif
#endif
#endif

// ---------------------------------------------------------------------------
// Compile-time limits / fixed-size shared buffers
// ---------------------------------------------------------------------------
static constexpr int MAX_THREADS = 1024;             // one full block on most GPUs
static constexpr int WARPS = MAX_THREADS / 32;       // 32 lanes per hardware warp

// ---------------------------------------------------------------------------
// Small host helpers
// ---------------------------------------------------------------------------
static bool is_pow2(int x) {
  return x > 0 && (x & (x - 1)) == 0;
}

// Closed form of sum_{i=0}^{N-1} [1 + (i%8)*0.125].
// One full period (k=0..7) contributes 8 + 0.125*(0+1+...+7) = 8 + 3.5 = 11.5.
static double analytic_sum(size_t n) {
  const size_t periods = n / 8;
  const size_t rem = n % 8;
  double s = static_cast<double>(periods) * 11.5;
  for (size_t k = 0; k < rem; ++k) s += 1.0 + 0.125 * static_cast<double>(k);
  return s;
}

// The reference deliberately uses double precision and a plain host loop: a
// different summation order and a wider accumulator than any GPU variant.
static double cpu_double_sum(const std::vector<float>& a) {
  double acc = 0.0;
  for (float v : a) acc += static_cast<double>(v);
  return acc;
}

// ---------------------------------------------------------------------------
// Device helper: reduce one value across a warp with shuffle-down.
// mask = 0xffffffff is valid because every block is a power of two >= 32, so
// all warps (including the last one) are full: 32 active lanes.
// offsets 16,8,4,2,1 sum all 32 lanes into lane 0.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;  // meaningful in lane 0; other lanes hold partials
}

// ===========================================================================
// VARIANT 2: naive_global -- single block, tree reduction in GLOBAL memory.
// ===========================================================================
__global__ void reduce_naive_global(const float* __restrict__ in,
                                    float* __restrict__ scratch,
                                    float* __restrict__ out, size_t n) {
  const int tid = threadIdx.x;  // 0 .. blockDim.x-1, only blockDim.x threads work

  // Defensive guard: the host validates this, so it can never actually trigger.
  if (blockDim.x > MAX_THREADS) {
    if (tid == 0) *out = 0.0f;
    return;
  }

  // Fold the entire array into blockDim.x per-thread partials. The loop bound
  // `i < n` is the out-of-bounds guard: it works for even, odd, and tiny N.
  // The classic SDK kernel writes into the source array; we use a separate
  // scratch buffer so the input stays intact for the later variants.
  float acc = 0.0f;
  for (size_t i = static_cast<size_t>(tid); i < n; i += blockDim.x) acc += in[i];
  scratch[tid] = acc;  // global store #1 (all blockDim.x slots are written)

  // Make the just-written partials visible to the whole block before reducing.
  __syncthreads();

  // Tree in GLOBAL memory: each step writes scratch[tid] and the NEXT step
  // reads it back -- a global-memory round trip per level. For tid >= s the
  // branch is not taken, yet those threads still reach __syncthreads(): that
  // is the tail divergence / wasted-wait problem described at the top.
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) scratch[tid] += scratch[tid + s];
    __syncthreads();  // execution barrier + global-write visibility fence
  }

  // Thread 0 holds the final sum.
  if (tid == 0) *out = scratch[0];
}

// ===========================================================================
// VARIANT 3a: shared_mem, stage 1 -- G blocks each reduce a slice.
// ===========================================================================
__global__ void reduce_shared_stage1(const float* __restrict__ in,
                                     float* __restrict__ partials, size_t n) {
  __shared__ float sdata[MAX_THREADS];  // fixed-size on-chip buffer
  const int tid = threadIdx.x;

  if (blockDim.x > MAX_THREADS) {  // defensive; host already validated
    if (tid == 0) partials[blockIdx.x] = 0.0f;
    return;
  }

  // CLASSIC TRAP: only zeroing sdata[tid] is fine for the live lanes here, but
  // if a future version lets tid >= s read sdata[tid+s] past blockDim.x it
  // would read uninitialised shared memory (garbage that varies run to run).
  // We therefore initialise ALL MAX_THREADS slots, not just the live ones.
  // (It costs a few stores; correctness first, and the comment marks it.)
  for (int i = tid; i < MAX_THREADS; i += blockDim.x) sdata[i] = 0.0f;
  __syncthreads();

  // Grid-stride loop: block b, thread t visits b*blockDim.x+t, then advances
  // by gridDim.x*blockDim.x. This covers any N (even/odd/large) exactly once.
  float acc = 0.0f;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    acc += in[i];
  }
  sdata[tid] = acc;  // on-chip store, no DRAM round trip for the tree
  __syncthreads();

  // Tree in shared memory. For s <= 32 only warp 0 is active; those lanes run
  // in lockstep, so these last barriers are redundant here (see warp version).
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // One partial per block goes to global memory for stage 2.
  if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// ===========================================================================
// VARIANT 3b: shared_mem, stage 2 -- block 0 reduces the G partials.
// ===========================================================================
__global__ void reduce_shared_stage2(const float* __restrict__ partials,
                                     float* __restrict__ out, int g) {
  __shared__ float sdata[MAX_THREADS];
  const int tid = threadIdx.x;

  if (blockDim.x > MAX_THREADS) {
    if (tid == 0) *out = 0.0f;
    return;
  }

  // Same defensive full initialisation as stage 1.
  for (int i = tid; i < MAX_THREADS; i += blockDim.x) sdata[i] = 0.0f;
  __syncthreads();

  // Host guarantees g <= blockDim.x; threads with tid >= g load 0.0f so the
  // tree still sees well-defined values (no out-of-bounds read of partials).
  sdata[tid] = (tid < g) ? partials[tid] : 0.0f;
  __syncthreads();

  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) *out = sdata[0];
}

// ===========================================================================
// VARIANT 4a: warp_shuffle, stage 1 -- canonical shuffle reduction.
// ===========================================================================
__global__ void reduce_warp_stage1(const float* __restrict__ in,
                                   float* __restrict__ partials, size_t n) {
  __shared__ float warp_sums[WARPS];  // one slot per warp, not per thread
  const int tid = threadIdx.x;
  const int lane = tid & 31;          // 0..31 within this warp
  const int warp = tid >> 5;          // which warp in the block
  const int num_warps = blockDim.x >> 5;

  if (blockDim.x > MAX_THREADS) {
    if (tid == 0) partials[blockIdx.x] = 0.0f;
    return;
  }

  // Initialise all WARPS slots so lanes reading warp_sums[lane] are defined.
  for (int i = tid; i < WARPS; i += blockDim.x) warp_sums[i] = 0.0f;
  __syncthreads();

  // Same grid-stride slice as the shared version; accumulation into a register.
  float acc = 0.0f;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    acc += in[i];
  }

  // Reduce within the warp with 5 shuffles: no shared memory, no barrier,
  // no bank conflicts. Only lane 0 of each warp has the full warp sum.
  acc = warp_reduce_sum(acc);
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();  // one barrier for the whole block, after all warp writes

  // Warp 0 reduces the per-warp partials with the same 5 shuffles.
  float total = 0.0f;
  if (warp == 0) {
    total = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    total = warp_reduce_sum(total);
  }
  if (tid == 0) partials[blockIdx.x] = total;  // lane 0 of warp 0
}

// ===========================================================================
// VARIANT 4b: warp_shuffle, stage 2 -- reduce the G partials in one block.
// ===========================================================================
__global__ void reduce_warp_stage2(const float* __restrict__ partials,
                                   float* __restrict__ out, int g) {
  __shared__ float warp_sums[WARPS];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = blockDim.x >> 5;

  if (blockDim.x > MAX_THREADS) {
    if (tid == 0) *out = 0.0f;
    return;
  }

  for (int i = tid; i < WARPS; i += blockDim.x) warp_sums[i] = 0.0f;
  __syncthreads();

  // Strided accumulation handles any g (host uses g <= blockDim.x here).
  float acc = 0.0f;
  for (int i = tid; i < g; i += blockDim.x) acc += partials[i];

  acc = warp_reduce_sum(acc);
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();

  float total = 0.0f;
  if (warp == 0) {
    total = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    total = warp_reduce_sum(total);
  }
  if (tid == 0) *out = total;
}

// ===========================================================================
// Shared verify -> time -> emit pipeline for every GPU variant.
// We VERIFY FIRST and exit on a wrong answer, so no timing is ever recorded
// for a kernel that returned a bad number.
// ===========================================================================
template <typename Launch>
static void check_time_emit(const std::string& variant, Launch&& launch,
                            float* d_out, double cpu_sum, size_t n,
                            int warmup, int iters, double tol,
                            const std::string& csv_path, const char* bw_note) {
  // ---- 1. correctness (one synchronous, untimed run) ----
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
  launch();
  CUDA_CHECK_LAST();                 // catch bad launch config immediately
  CUDA_CHECK(cudaDeviceSynchronize());  // catch async faults before checking

  float gpu_result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&gpu_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  // compare_gpu_cpu is a per-element helper; for a reduction "n" is 1. The
  // double CPU sum is cast to the nearest float: a float GPU result cannot be
  // more accurate than that (and for our fill pattern the value is exact).
  const float ref = static_cast<float>(cpu_sum);
  const MaxError e = compare_gpu_cpu(&gpu_result, &ref, 1, tol, tol);

  std::printf("  %-16s gpu=%14.4f  cpu=%.4f  err=%.6g  rel=%.3g  ok=%d\n",
              variant.c_str(), static_cast<double>(gpu_result), cpu_sum,
              e.abs_err, e.rel_err, e.ok ? 1 : 0);
  if (!e.ok) {
    std::fprintf(stderr,
                 "FAIL %s: gpu=%.9g cpu=%.9g err=%.6g rel=%.6g tol=%.6g "
                 "-- aborting, no timing recorded\n",
                 variant.c_str(), static_cast<double>(gpu_result), cpu_sum,
                 e.abs_err, e.rel_err, tol);
    std::exit(EXIT_FAILURE);
  }

  // ---- 2. timing (CUDA events, via the shared helper) ----
  const Stats s = time_iters([&] { launch(); }, warmup, iters);

  // ---- 3. throughput + CSV row ----
  // bytes = N*4 for the input read. For the two-stage kernels this is an
  // *effective* bandwidth: the nominal N*4 is divided by a time that also
  // includes writing/reading the G partials and the second kernel launch.
  const double throughput = gbs(static_cast<double>(n) * 4.0, s.mean);
  print_row(variant, s, "GB/s", throughput);

  char extra[256];
  std::snprintf(extra, sizeof(extra),
                "result=%.6f err=%.6g rel=%.6g ok=%d tol=%.3g bw=%s",
                static_cast<double>(gpu_result), e.abs_err, e.rel_err,
                e.ok ? 1 : 0, tol, bw_note);
  csv_emit(csv_path, "reduction", variant, static_cast<double>(n), "elements",
           s, throughput, "GB/s", extra);
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
  const long long n_ll = arg_ll(argc, argv, "n", 16777216LL);   // 2^24
  const int threads = arg_int(argc, argv, "threads", 256);
  const int iters = arg_int(argc, argv, "iters", 50);
  const int warmup = arg_int(argc, argv, "warmup", 10);
  const double tol = arg_double(argc, argv, "tol", 1e-4);
  const std::string csv_path = arg_value(argc, argv, "csv", "../results/reduction.csv");

  if (n_ll < 1) {
    std::fprintf(stderr, "--n must be >= 1\n");
    return EXIT_FAILURE;
  }
  if (!is_pow2(threads) || threads < 32 || threads > MAX_THREADS) {
    std::fprintf(stderr, "--threads must be a power of two in [32, %d]\n", MAX_THREADS);
    return EXIT_FAILURE;
  }
  if (iters < 1 || warmup < 0 || !(tol >= 0.0)) {
    std::fprintf(stderr, "invalid --iters/--warmup/--tol\n");
    return EXIT_FAILURE;
  }
  const size_t N = static_cast<size_t>(n_ll);

  device_info();  // touches CUDA once; csv_emit also records it in every row

  std::printf("reduction: N=%zu threads=%d iters=%d warmup=%d tol=%.6g csv=%s\n",
              N, threads, iters, warmup, tol, csv_path.c_str());

  // ---- deterministic input: exact in float, analytic closed form exists ----
  std::vector<float> h_data(N);
  for (size_t i = 0; i < N; ++i) {
    h_data[i] = 1.0f + static_cast<float>(i & 7) * 0.125f;
  }
  const double analytic = analytic_sum(N);

  // =========================================================================
  // VARIANT 1: cpu_reference -- host loop in double, timed with steady_clock.
  // CUDA events only timestamp device work, so a wall-clock timer is required.
  // =========================================================================
  const double cpu_sum = cpu_double_sum(h_data);
  // Sanity-check the reference itself: for this fill pattern the double sum
  // must equal the closed form (both are exact). If they disagree the input
  // generation or analytic formula is broken and nothing else is trustworthy.
  const double cpu_err = std::fabs(cpu_sum - analytic);
  const bool cpu_ok = cpu_err <= 1e-9 * (std::fabs(analytic) + 1.0);
  std::printf("  analytic=%.6f  cpu_double=%.6f  (diff=%.3g)\n",
              analytic, cpu_sum, cpu_sum - analytic);
  if (!cpu_ok) {
    std::fprintf(stderr, "FAIL cpu_reference: sum != analytic (err=%.6g)\n", cpu_err);
    return EXIT_FAILURE;
  }

  auto cpu_once = [&]() { return cpu_double_sum(h_data); };
  volatile double sink = 0.0;  // stop the optimiser from eliding the timed sum
  for (int i = 0; i < warmup; ++i) sink = cpu_once();
  std::vector<double> cpu_samples;
  cpu_samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const double v = cpu_once();
    const auto t1 = std::chrono::steady_clock::now();
    sink = v;
    cpu_samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  (void)sink;
  const Stats cpu_stats = stats_of(cpu_samples);
  const double cpu_tp = gbs(static_cast<double>(N) * 4.0, cpu_stats.mean);
  std::printf("  %-16s result=%.6f  err=%.6g  ok=%d  (CPU reference, wall-clock)\n",
              "cpu_reference", cpu_sum, cpu_err, cpu_ok ? 1 : 0);
  print_row("cpu_reference", cpu_stats, "GB/s", cpu_tp);
  {
    char extra[128];
    std::snprintf(extra, sizeof(extra),
                  "result=%.6f err=%.6g rel=0 ok=%d bw=host_wall_clock",
                  cpu_sum, cpu_err, cpu_ok ? 1 : 0);
    csv_emit(csv_path, "reduction", "cpu_reference", static_cast<double>(N),
             "elements", cpu_stats, cpu_tp, "GB/s", extra);
  }

  // ---- device buffers (allocated once, reused by all GPU variants) ----
  float* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

  float* d_scratch = nullptr;  // naive_global: one slot per thread
  CUDA_CHECK(cudaMalloc(&d_scratch, static_cast<size_t>(threads) * sizeof(float)));

  // G blocks for the two-stage variants: enough to cover the array, but never
  // more than `threads` so stage 2 can reduce G partials in a single block.
  size_t g_sz = (N + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
  if (g_sz > static_cast<size_t>(threads)) g_sz = static_cast<size_t>(threads);
  if (g_sz == 0) g_sz = 1;
  const int G = static_cast<int>(g_sz);

  float* d_partials = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partials, static_cast<size_t>(G) * sizeof(float)));

  // =========================================================================
  // VARIANT 2: naive_global -- single block, log2(threads) global round trips.
  // =========================================================================
  check_time_emit("naive_global",
                  [&] {
                    reduce_naive_global<<<1, threads>>>(d_data, d_scratch, d_out, N);
                  },
                  d_out, cpu_sum, N, warmup, iters, tol, csv_path,
                  "effective");

  // =========================================================================
  // VARIANT 3: shared_mem -- stage 1 (G blocks) + stage 2 (block 0).
  // =========================================================================
  check_time_emit("shared_mem",
                  [&] {
                    reduce_shared_stage1<<<G, threads>>>(d_data, d_partials, N);
                    reduce_shared_stage2<<<1, threads>>>(d_partials, d_out, G);
                  },
                  d_out, cpu_sum, N, warmup, iters, tol, csv_path,
                  "effective_multi_stage");

  // =========================================================================
  // VARIANT 4: warp_shuffle -- same two stages, shuffles inside each warp.
  // =========================================================================
  check_time_emit("warp_shuffle",
                  [&] {
                    reduce_warp_stage1<<<G, threads>>>(d_data, d_partials, N);
                    reduce_warp_stage2<<<1, threads>>>(d_partials, d_out, G);
                  },
                  d_out, cpu_sum, N, warmup, iters, tol, csv_path,
                  "effective_multi_stage");

  // =========================================================================
  // VARIANT 5: cub_reference -- DeviceReduce::Sum with preallocated scratch.
  // CUB is the "what a professional would ship" baseline: heavily tuned, and
  // the thing to reach for in real code instead of hand-writing reductions.
  // Two calls: the first (null temp storage) only queries the byte count; we
  // allocate once, then time only the real reduction calls.
  // =========================================================================
#ifdef HAVE_CUB
  if (N <= static_cast<size_t>(INT_MAX)) {
    const int n_int = static_cast<int>(N);
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, d_data, d_out, n_int));
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    check_time_emit("cub_reference",
                    [&] {
                      CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data,
                                                        d_out, n_int));
                    },
                    d_out, cpu_sum, N, warmup, iters, tol, csv_path,
                    "effective_device_reduce");
    CUDA_CHECK(cudaFree(d_temp));
  } else {
    std::printf("  cub_reference SKIPPED: N > INT_MAX, CUB query uses int\n");
  }
#else
  std::printf("  cub_reference SKIPPED: CUB headers not available (or REDUCTION_NO_CUB)\n");
#endif

  // ---- cleanup ----
  CUDA_CHECK(cudaFree(d_partials));
  CUDA_CHECK(cudaFree(d_scratch));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_data));

  std::printf("done: rows appended to %s (device info recorded by notebook)\n",
              csv_path.c_str());
  return EXIT_SUCCESS;
}
