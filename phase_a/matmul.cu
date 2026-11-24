// ===========================================================================
// matmul.cu -- single-precision square GEMM, four variants, one benchmark.
//
// WHAT THIS MEASURES AND WHY THE VARIANTS EXIST
// ---------------------------------------------------------------------------
// C = A * B, all row-major N x N. For one output element the naive kernel does
// N multiply-adds but re-reads two length-N vectors (2N floats = 8N bytes)
// straight from global memory. Counting a multiply-add as two flops that is
// ~2N flops for 8N bytes of traffic -- roughly "N useful flops per 2N words
// fetched" -- an arithmetic intensity far below the flop:byte balance of this
// GPU. It is therefore memory bound, and the on-chip FMAs mostly sit idle.
//
// The tiled kernel breaks that: a 32x32 block loads one A tile and one B tile
// into shared memory ONCE per phase, then reuses each loaded value 32 times in
// the inner loop. The 2N global reads per output drop by about 32x, so the
// kernel becomes compute bound and its throughput rises accordingly.
//
// TILE SIZE ON THIS GPU (T4 / sm_75): shared memory is 48-64 KB per SM and a
// block cannot use more than that. Two 32x32 float tiles cost 2*32*32*4 = 8 KB,
// leaving plenty of room for occupancy; much larger tiles (e.g. 128x128 would
// be 128 KB) do not fit at all. 32x32 (blockDim 32x32) is the natural unit here.
//
// EXPECTATIONS, HONESTLY: this tiled kernel still has no register blocking and
// no double buffering of the global->shared pipeline, which every truly fast
// GEMM needs. cuBLAS uses both plus architecture-specific scheduling, so it is
// expected to win by a large factor. That gap is a FINDING of this benchmark,
// not a failure of the implementation -- it is the story the write-up tells.
//
// Every number written to the CSV comes from an actual measurement; nothing is
// estimated or hard-coded. On the host we time with steady_clock; on the GPU we
// use the shared CudaTimer/time_iters helpers (CUDA events, mean and sample std).
// ===========================================================================

#include "bench_common.h"   // CUDA_CHECK, Stats, time_iters, compare_gpu_cpu, csv_emit, CLI helpers
#include <cublas_v2.h>      // cublasCreate / cublasSgemm / cublasDestroy
#include <chrono>           // steady_clock for the CPU baseline

namespace {

// cuBLAS status is a different enum from cudaError_t, and the header only gives
// us CUDA_CHECK, so this tiny host-side helper fills the gap. It never runs on
// the device and adds no GPU time to the measured region.
void check_cublas(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS error %d in %s\n", static_cast<int>(st), what);
    std::exit(EXIT_FAILURE);
  }
}

// 32x32 shared-memory tile; one C tile per block, one thread per output cell.
constexpr int TILE = 32;

// Deterministic pseudo-random fill. The hash mixes i, j and a per-matrix salt
// so A and B are different but reproducible on every run and every machine.
// Values are scaled to O(1/N): an N-term dot product then stays O(1), FP32
// cannot overflow, and an element-wise tolerance of ~1e-4 stays meaningful at
// every size in the sweep (a fixed unscaled magnitude would blow up at 1024).
float host_elem(int i, int j, int n, int salt) {
  unsigned int h = static_cast<unsigned int>(i) * 73856093u ^
                   static_cast<unsigned int>(j) * 19349663u ^
                   static_cast<unsigned int>(salt) * 83492791u;
  h ^= h >> 13;             // Mix the high bits down.
  h *= 2654435761u;         // Multiplicative scramble.
  h ^= h >> 16;             // Final avalanche.
  const int s = static_cast<int>(h % 2001u) - 1000;          // [-1000, 1000]
  return static_cast<float>(s) / (1000.0f * static_cast<float>(n));  // O(1/N)
}

// Fill a row-major n x n matrix with the deterministic, O(1/N) values above.
void fill_matrix(float* m, int n, int salt) {
  for (int i = 0; i < n; ++i) {                 // row index
    for (int j = 0; j < n; ++j) {               // column index
      m[static_cast<size_t>(i) * n + j] = host_elem(i, j, n, salt);
    }
  }
}

// Host reference: the plain i,j,k triple loop over row-major arrays.
// A[i,k] is at i*n+k, B[k,j] at k*n+j, C[i,j] at i*n+j.
void cpu_matmul(const float* a, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) {                 // each row of C
    for (int j = 0; j < n; ++j) {               // each column of C
      float sum = 0.0f;                         // accumulator in FP32, like the GPU
      for (int k = 0; k < n; ++k) {             // the dot product
        sum += a[static_cast<size_t>(i) * n + k] * b[static_cast<size_t>(k) * n + j];
      }
      c[static_cast<size_t>(i) * n + j] = sum;  // store one output element
    }
  }
}

// Split --size-grid on commas AND whitespace, so "256,512,1024" and
// "256 512 1024" (quoted) both work. Non-positive tokens are ignored.
std::vector<int> parse_sizes(const std::string& s) {
  std::vector<int> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\t')) ++i;  // skip separators
    const size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != ' ' && s[i] != '\t') ++i;    // one token
    if (i > start) {
      const int v = std::atoi(s.substr(start, i - start).c_str());
      if (v > 0) out.push_back(v);
    }
  }
  return out;
}

}  // namespace (host helpers above; kernels live at global scope below)

// ---------------------------------------------------------------------------
// Variant 2: naive. One thread computes one C[row,col] with a full dot product.
// WHY IT IS SLOW: every thread re-reads 2*N values (A[row,:] and B[:,col]), so
// the matrix is streamed from global memory O(N) times over. Worse, B is read
// COLUMN-wise: consecutive threads in a warp hold consecutive 'col' values but
// address B[k*n + col], which are n floats apart. Each 4-byte load therefore
// touches a different 32-byte sector -- uncoalesced, wasting ~8x the bandwidth.
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float* a, const float* b, float* c, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // column of C (and of B)
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // row of C (and of A)

  if (row < n && col < n) {                 // guard the ragged tile edge
    float sum = 0.0f;                       // private accumulator in a register
    for (int k = 0; k < n; ++k) {           // full dot product, straight from global
      sum += a[static_cast<size_t>(row) * n + k] *
             b[static_cast<size_t>(k) * n + col];
    }
    c[static_cast<size_t>(row) * n + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// Variant 3: tiled shared-memory GEMM, TILE x TILE threads per C tile.
// For each phase t we load A's tile (rows of this block, columns t) into As and
// B's tile (rows t, columns of this block) into Bs, then reuse them TILE times.
// Out-of-range entries are ZERO-FILLED instead of branching in the inner loop,
// so N not divisible by TILE is correct with an unconditional accumulate.
// ---------------------------------------------------------------------------
__global__ void gemm_tiled(const float* a, const float* b, float* c, int n) {
  __shared__ float As[TILE][TILE];          // A tile: rows of block, columns of phase
  __shared__ float Bs[TILE][TILE];          // B tile: rows of phase, columns of block

  const int col = blockIdx.x * TILE + threadIdx.x;  // global column of C
  const int row = blockIdx.y * TILE + threadIdx.y;  // global row of C
  const int n_phases = (n + TILE - 1) / TILE;       // ceil(N / TILE)
  float acc = 0.0f;                                 // accumulator spans ALL phases

  for (int t = 0; t < n_phases; ++t) {
    // --- load A tile: A[blockY*32 + ty, t*32 + tx] -> As[ty][tx] ---
    const int a_row = blockIdx.y * TILE + threadIdx.y;
    const int a_col = t * TILE + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
        (a_row < n && a_col < n)
            ? a[static_cast<size_t>(a_row) * n + a_col]
            : 0.0f;                                 // zero-fill past the right/bottom edge

    // --- load B tile: B[t*32 + ty, blockX*32 + tx] -> Bs[ty][tx] ---
    const int b_row = t * TILE + threadIdx.y;
    const int b_col = blockIdx.x * TILE + threadIdx.x;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < n && b_col < n)
            ? b[static_cast<size_t>(b_row) * n + b_col]
            : 0.0f;                                 // zero-fill past the right/bottom edge

    __syncthreads();                              // (before) all tile loads visible

    // acc += sum_k A[row, t*32+k] * B[t*32+k, col], using shared memory only.
    // As[ty][k] is A[row, t*32+k]; Bs[k][tx] is B[t*32+k, col]. Out-of-range
    // k entries are zero, so extra terms contribute nothing -- no branch here.
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();                              // (after) no thread reads while others
                                                  // overwrite the tiles for phase t+1
  }

  if (row < n && col < n) {                       // only real cells get stored
    c[static_cast<size_t>(row) * n + col] = acc;
  }
}

int main(int argc, char** argv) {
  // -------------------------------------------------------------------------
  // CLI. --n is the GPU sweep size; --cpu-n is the (much smaller) size at which
  // the CPU reference is computed and the CPU variant is benchmarked. 2*N^3
  // host flops at N=1024 would take far too long, hence the separate cpu_n.
  // -------------------------------------------------------------------------
  int n = arg_int(argc, argv, "n", 1024);
  int cpu_n = arg_int(argc, argv, "cpu-n", 256);
  int iters = arg_int(argc, argv, "iters", 20);
  int warmup = arg_int(argc, argv, "warmup", 5);
  const std::string csv = arg_value(argc, argv, "csv", "../results/matmul.csv");
  const double tol = arg_double(argc, argv, "tol", 1e-4);
  const std::string grid = arg_value(argc, argv, "size-grid", "");

  if (n < 1) n = 1;                              // keep all sizes sane
  if (cpu_n < 1) cpu_n = 1;
  if (iters < 1) iters = 1;
  if (warmup < 0) warmup = 0;

  // Build the list of sizes: an explicit sweep wins, otherwise just --n.
  // cpu_n is always included so the GPU is verified against the CPU there and
  // the CPU baseline is actually measured. Duplicates are dropped.
  std::vector<int> sizes;
  auto add_size = [&sizes](int v) {              // not a template: plain lambda
    if (std::find(sizes.begin(), sizes.end(), v) == sizes.end()) sizes.push_back(v);
  };
  if (!grid.empty()) {
    for (int s : parse_sizes(grid)) add_size(s);
  } else {
    add_size(n);
  }
  add_size(cpu_n);

  const DeviceInfo& dev = device_info();         // device identity for the header line
  std::printf("matmul: device=%s arch=%s  iters=%d warmup=%d tol=%g\n",
              dev.name.c_str(), dev.arch.c_str(), iters, warmup, tol);

  // Handle created ONCE before any timing and destroyed after every size.
  cublasHandle_t handle = nullptr;
  check_cublas(cublasCreate(&handle), "cublasCreate");

  const float alpha = 1.0f;                      // C = 1*A*B + 0*C
  const float beta = 0.0f;
  const dim3 block(TILE, TILE);                  // 32x32 threads for naive and tiled

  for (int N : sizes) {
    std::printf("\nN=%d\n", N);
    const size_t count = static_cast<size_t>(N) * N;
    const size_t bytes = count * sizeof(float);
    const double flops = 2.0 * static_cast<double>(N) * N * N;  // multiply + add per k

    // Deterministic host inputs, filled once per size.
    std::vector<float> h_a(count), h_b(count);
    fill_matrix(h_a.data(), N, 0);               // A uses salt 0
    fill_matrix(h_b.data(), N, 1);               // B uses salt 1

    // Device buffers: A, B are inputs; C is the output every variant overwrites.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 grid_dim((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    std::vector<float> h_back(count), h_cublas(count);  // copy-back and cublas oracle
    std::vector<float> h_cpu;                     // only allocated at cpu_n
    const float* ref = nullptr;                   // correctness oracle for this size
    std::string ref_tag;                          // goes into the CSV 'extra' column

    if (N == cpu_n) {
      h_cpu.resize(count);
      cpu_matmul(h_a.data(), h_b.data(), h_cpu.data(), N);  // the CPU reference
      ref = h_cpu.data();
      ref_tag = "reference=CPU";
    } else {
      ref_tag = "reference=cuBLAS (CPU not run at this N)";
    }

    // --- cuBLAS reference FIRST: it is the oracle when N != cpu_n. -----------
    // Row-major C=A*B is column-major C^T = B^T * A^T, so with both transposes
    // OP_N we pass B as the first matrix and A as the second:
    // cublasSgemm(handle, N, N, N,N,N, &alpha, d_b, N, d_a, N, &beta, d_c, N).
    Stats cublas_stats = time_iters([&]() {
      check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                               &alpha, d_b, N, d_a, N, &beta, d_c, N),
                   "cublasSgemm");
    }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    if (N != cpu_n) ref = h_cublas.data();       // larger N validates against cuBLAS
    const MaxError cublas_err =
        compare_gpu_cpu(h_cublas.data(), ref, count, tol, tol);

    // --- naive: one thread per output, global-memory reads only. -----------
    Stats naive_stats = time_iters([&]() {
      gemm_naive<<<grid_dim, block>>>(d_a, d_b, d_c, N);
      CUDA_CHECK_LAST();                         // host-side launch sanity check
    }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(h_back.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    const MaxError naive_err =
        compare_gpu_cpu(h_back.data(), ref, count, tol, tol);

    // --- tiled: 32x32 shared-memory tiles, zero-filled edges. ---------------
    Stats tiled_stats = time_iters([&]() {
      gemm_tiled<<<grid_dim, block>>>(d_a, d_b, d_c, N);
      CUDA_CHECK_LAST();
    }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(h_back.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    const MaxError tiled_err =
        compare_gpu_cpu(h_back.data(), ref, count, tol, tol);

    // --- CPU baseline: steady_clock, warmup then one sample per iteration. --
    Stats cpu_stats;
    if (N == cpu_n) {
      for (int i = 0; i < warmup; ++i) {
        cpu_matmul(h_a.data(), h_b.data(), h_cpu.data(), N);
      }
      std::vector<double> cpu_samples;
      cpu_samples.reserve(static_cast<size_t>(iters));
      for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        cpu_matmul(h_a.data(), h_b.data(), h_cpu.data(), N);
        const auto t1 = std::chrono::steady_clock::now();
        cpu_samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      cpu_stats = stats_of(cpu_samples);
    }

    // Emit one row per variant. print_row shows the headline GFLOP/s; the line
    // under it reports the measured max error and whether it met the tolerance.
    auto emit = [&](const char* variant, const Stats& st, const MaxError& err,
                    const std::string& extra) {
      const double gf = gflops(flops, st.mean);
      print_row(variant, st, "GFLOPS", gf);
      std::printf("      max abs err %.3e, max rel err %.3e, within tol: %s\n",
                  err.abs_err, err.rel_err, err.ok ? "yes" : "NO");
      csv_emit(csv, "matmul", variant, static_cast<double>(N), "N", st, gf,
               "GFLOPS", extra);
    };

    if (N == cpu_n) {
      // The CPU result IS the reference, so its 'error' is identically zero.
      MaxError cpu_err;           // abs=rel=0 by default; the CPU is the oracle
      cpu_err.ok = true;          // so its comparison is trivially within tol
      emit("cpu_naive", cpu_stats, cpu_err, "reference=CPU (row-major triple loop)");
    }
    emit("naive", naive_stats, naive_err, ref_tag);
    emit("tiled", tiled_stats, tiled_err, ref_tag);
    if (N == cpu_n) {
      emit("cublas", cublas_stats, cublas_err, ref_tag);
    } else {
      emit("cublas", cublas_stats, cublas_err,
           "reference=cuBLAS (self; oracle for this N)");
    }

    // Free this size's device buffers before moving to the next one.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
  }

  check_cublas(cublasDestroy(handle), "cublasDestroy");
  return 0;
}
