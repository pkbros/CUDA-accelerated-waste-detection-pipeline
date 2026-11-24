// ===========================================================================
// streams.cu -- why pinned (page-locked) memory + multiple CUDA streams beat
//               synchronous copies.
//
// THE BIG PICTURE (read once; the code below is just mechanics):
//
//  * A CUDA "stream" is a FIFO queue of work for the device. Commands pushed to
//    the SAME stream execute in issue order. Commands pushed to DIFFERENT
//    streams may execute concurrently, if the hardware has spare resources. A
//    stream is therefore the unit of parallelism we use to overlap a copy with
//    a kernel.
//
//  * The "copy engine" is dedicated DMA hardware on the GPU. It moves bytes
//    between host RAM and device RAM without occupying the SMs that run our
//    kernel. Because it is separate hardware, a host->device (H2D) copy can be
//    in flight at the same moment a kernel executes on the SMs. That is the
//    overlap we are trying to expose.
//
//  * The legacy "default" stream (stream 0) is SYNCHRONISING by default: an
//    operation in it waits for previously issued work in other blocking streams
//    to finish, and later blocking-stream work waits for it. That implicit
//    two-way barrier is why the "sync_*" baselines cannot overlap anything --
//    every H2D, kernel and D2H is fenced against the next. The fast variants
//    avoid the default stream entirely by creating explicit non-blocking
//    streams with cudaStreamCreateWithFlags(...cudaStreamNonBlocking).
//
//  * "Overlap" is easiest to see on a timeline. In the stream variants we
//    enqueue, in order:
//        stream 0: H2D(chunk0) -> kernel(chunk0) -> D2H(chunk0)
//        stream 1: H2D(chunk1) -> kernel(chunk1) -> D2H(chunk1)
//    With two streams the device can run, for example, the H2D of chunk 1 (on
//    the copy engine) while the kernel of chunk 0 is still running on the SMs.
//    The wall clock for the whole batch shrinks even though each individual
//    chunk takes the same time to go from H2D to D2H.
//
//  * Why pinning matters: cudaMallocHost returns page-locked host memory. The
//    driver can hand those physical pages directly to the copy engine, so DMA
//    happens straight to/from them; the copy engine does NOT have to stage the
//    bytes through a temporary pinned "bounce buffer". The CPU can also keep
//    running instead of being tied up doing that staging. Pinned memory lets
//    cudaMemcpyAsync genuinely return immediately, so the CPU can go on to
//    issue the rest of the pipeline. In other words pinning is a PREREQUISITE
//    for the stream win, not an extra bonus on top of it.
//
//  * Two things cap the speedup you will see here:
//      1. There are only 4 chunks, so the pipeline is never more than 4 deep.
//         With 3 streams the 4th chunk lands on a stream whose earlier work is
//         already finished, so it runs alone at the tail with nothing to hide
//         behind. That is why streams3_pinned is NOT a clean 2x over
//         streams2_pinned -- the last stream finishes alone.
//      2. Pinning is a prerequisite rather than a bonus: the "sync_pinned"
//         row already captures most of the single-stream gain. Comparing a
//         stream variant to sync_pageable mixes both effects; comparing it to
//         sync_pinned isolates what streams add on top of pinning.
//
//  * On Colab the PCIe link is shared / virtualised between the notebook and
//    other tenants, so the absolute GB/s printed here will be lower than a
//    bare-metal desktop with a dedicated x16 slot. The RATIO between rows is
//    the meaningful, reproducible number.
// ===========================================================================

#include "../include/bench_common.h"

#include <cstddef>

namespace {

// Four equal chunks of one big host buffer. This is the pipeline depth.
constexpr int kChunks = 4;

// ---------------------------------------------------------------------------
// The workload kernel.
// ---------------------------------------------------------------------------
// One thread per element: y[i] = a[i] * 2.0f + 1.0f.
// NOTE: we are measuring the *pipeline* (copy / launch / overlap), not the ALU.
// This kernel is deliberately tiny so that the copy and launch costs dominate
// the timeline. Do not read the GB/s here as a measure of compute throughput.
__global__ void scale_kernel(const float* __restrict__ a,
                             float* __restrict__ y,
                             size_t n) {
  // Global thread index: block number times block size plus in-block index.
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Guard the tail: the last block may have more threads than there are items.
  if (i < n) {
    // The actual arithmetic. Exact multiply-by-two plus add of one in FP32.
    y[i] = a[i] * 2.0f + 1.0f;
  }
}

// ---------------------------------------------------------------------------
// All buffers live in one struct so setup / teardown / variants share them.
// ---------------------------------------------------------------------------
struct Context {
  size_t chunk_floats = 0;  // elements per chunk (total / kChunks)
  size_t total_floats = 0;  // elements in the whole buffer
  size_t total_bytes = 0;   // bytes in the whole buffer
  int block = 256;          // threads per block for scale_kernel
  int grid = 0;             // blocks per chunk launch

  float* d_a = nullptr;       // device copy of input a (all chunks)
  float* d_y = nullptr;       // device output y (all chunks)

  float* h_page_a = nullptr;  // pageable (malloc) input, all chunks
  float* h_page_y = nullptr;  // pageable (malloc) output, all chunks
  float* h_pin_a = nullptr;   // pinned (cudaMallocHost) input, all chunks
  float* h_pin_y = nullptr;   // pinned (cudaMallocHost) output, all chunks

  // Host reference, computed ONCE from the input before any timing starts.
  std::vector<float> ref;
};

// Pointer to chunk c's input / output regions inside a given host buffer.
inline float* host_chunk(float* base, int chunk, size_t chunk_floats) {
  return base + static_cast<size_t>(chunk) * chunk_floats;
}

// Allocate and fill everything. Called once, outside all timing.
void setup(Context& c, int total_mb) {
  // Treat "--total-mb" as MiB, the usual meaning in a GPU benchmark.
  size_t total_bytes = static_cast<size_t>(total_mb) * 1024ULL * 1024ULL;
  size_t total_floats = total_bytes / sizeof(float);

  // Round DOWN so the kChunks chunks are exactly equal, never ragged.
  c.chunk_floats = total_floats / kChunks;
  total_floats = c.chunk_floats * kChunks;
  total_bytes = total_floats * sizeof(float);
  c.total_floats = total_floats;
  c.total_bytes = total_bytes;

  // Launch geometry: one thread per element, 256 threads per block.
  c.block = 256;
  c.grid = static_cast<int>((c.chunk_floats + c.block - 1) / c.block);

  // Two device buffers, one for a and one for y, split into kChunks regions.
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.d_a), total_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.d_y), total_bytes));

  // Pageable host pair: ordinary malloc memory, the "naive" baseline.
  c.h_page_a = static_cast<float*>(std::malloc(total_bytes));
  c.h_page_y = static_cast<float*>(std::malloc(total_bytes));
  // Pinned host pair: the driver can DMA straight from these pages.
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&c.h_pin_a), total_bytes));
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&c.h_pin_y), total_bytes));

  // Fill the input deterministically so the reference is reproducible.
  for (size_t i = 0; i < total_floats; ++i) {
    c.h_page_a[i] = static_cast<float>((i % 4096) * 0.001f);
  }
  // Make the pinned input bit-identical to the pageable input.
  std::memcpy(c.h_pin_a, c.h_page_a, total_bytes);

  // Compute the CPU reference ONCE. Both variants are checked against it.
  c.ref.resize(total_floats);
  for (size_t i = 0; i < total_floats; ++i) {
    c.ref[i] = 2.0f * c.h_page_a[i] + 1.0f;
  }
}

void teardown(Context& c) {
  // Reverse order of setup: pinned, pageable, device.
  if (c.h_pin_a) CUDA_CHECK(cudaFreeHost(c.h_pin_a));
  if (c.h_pin_y) CUDA_CHECK(cudaFreeHost(c.h_pin_y));
  std::free(c.h_page_a);
  std::free(c.h_page_y);
  if (c.d_a) CUDA_CHECK(cudaFree(c.d_a));
  if (c.d_y) CUDA_CHECK(cudaFree(c.d_y));
}

// ---------------------------------------------------------------------------
// Variant 1 / 2: one chunk at a time on the DEFAULT stream.
//   H2D (synchronous) -> kernel -> D2H (synchronous) -> device sync.
// ---------------------------------------------------------------------------
void run_chunk_sync(const Context& c, int chunk, bool pinned) {
  // Pick the pinned or pageable host pair for this variant.
  const float* h_a = host_chunk(pinned ? c.h_pin_a : c.h_page_a, chunk, c.chunk_floats);
  float* h_y = host_chunk(pinned ? c.h_pin_y : c.h_page_y, chunk, c.chunk_floats);
  // Each chunk owns a disjoint slice of the device buffers (no cross-chunk race).
  float* d_a = host_chunk(c.d_a, chunk, c.chunk_floats);
  float* d_y = host_chunk(c.d_y, chunk, c.chunk_floats);
  const size_t bytes = c.chunk_floats * sizeof(float);

  // Synchronous H2D: blocks the CPU until the copy is complete.
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  // Kernel on the default stream, ordered after the H2D above.
  scale_kernel<<<c.grid, c.block>>>(d_a, d_y, c.chunk_floats);
  // Check for a bad launch immediately. This is the same check as the header's
  // CUDA_CHECK_LAST(); we spell out cudaGetLastError() through CUDA_CHECK so it
  // is robust even where that helper is aliased as an object-like macro.
  CUDA_CHECK(cudaGetLastError());
  // Synchronous D2H: waits for the kernel (same stream), then copies back.
  CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
  // Naive "works but wastes the GPU" barrier at the end of each chunk.
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Variants 3 / 4: N non-blocking streams, chunk i on stream i % N.
// All copies are plain cudaMemcpyAsync on the SAME stream as the kernel, so
// H2D -> kernel -> D2H are strictly ordered within a chunk but free to interleave
// with other chunks on other streams.
// ---------------------------------------------------------------------------
template <int NStreams>
void run_chunk_streams(const Context& c, cudaStream_t* streams, cudaEvent_t* events) {
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    // Round-robin assignment of chunks to streams.
    cudaStream_t s = streams[chunk % NStreams];

    float* d_a = host_chunk(c.d_a, chunk, c.chunk_floats);
    float* d_y = host_chunk(c.d_y, chunk, c.chunk_floats);
    // Stream variants use the PINNED host pair (async memcpy needs pinned RAM).
    const float* h_a = host_chunk(c.h_pin_a, chunk, c.chunk_floats);
    float* h_y = host_chunk(c.h_pin_y, chunk, c.chunk_floats);
    const size_t bytes = c.chunk_floats * sizeof(float);

    // H2D on stream s. Because the memory is pinned, this returns at once and
    // the copy engine can start while we enqueue the rest.
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, s));
    // Kernel on the SAME stream s: starts only after this chunk's H2D finishes.
    scale_kernel<<<c.grid, c.block, 0, s>>>(d_a, d_y, c.chunk_floats);
    // Same launch-error check as CUDA_CHECK_LAST(), spelled out for robustness.
    CUDA_CHECK(cudaGetLastError());
    // D2H on stream s: starts only after this chunk's kernel finishes.
    CUDA_CHECK(cudaMemcpyAsync(h_y, d_y, bytes, cudaMemcpyDeviceToHost, s));
  }

  // Per-stream event at the tail: record after that stream's last enqueued
  // chunk, then wait on the CPU. This is what makes "the pipeline is done".
  for (int s = 0; s < NStreams; ++s) {
    CUDA_CHECK(cudaEventRecord(events[s], streams[s]));
    CUDA_CHECK(cudaEventSynchronize(events[s]));
  }
  // Final device-wide barrier. It is redundant with the per-stream events, but
  // it brackets the enqueued work cleanly for the enclosing CudaTimer (below),
  // so this is a PIPELINE-LEVEL timing, not a per-kernel timing.
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Correctness: check every chunk's D2H output against the precomputed reference.
// ---------------------------------------------------------------------------
bool verify(const Context& c, bool pinned) {
  bool all_ok = true;
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const float* got = host_chunk(pinned ? c.h_pin_y : c.h_page_y, chunk, c.chunk_floats);
    const float* want = c.ref.data() + static_cast<size_t>(chunk) * c.chunk_floats;

    // Per-element worst-case check, rtol = 1e-6 (atol 1e-6 for a tight floor).
    const MaxError e = compare_gpu_cpu(got, want, c.chunk_floats, 1e-6, 1e-6);
    std::printf("    chunk %d: %s  max_abs=%.3g  max_rel=%.3g\n",
                chunk, e.ok ? "PASS" : "FAIL", e.abs_err, e.rel_err);
    all_ok = all_ok && e.ok;
  }
  return all_ok;
}

// ---------------------------------------------------------------------------
// Print and persist one measured row. speedup comes only from measured means.
// ---------------------------------------------------------------------------
void report(const std::string& csv, const std::string& variant,
            const Stats& s, double total_bytes, double baseline_mean, bool ok) {
  // Effective throughput counts BOTH directions of the pipeline: H2D + D2H.
  const double throughput = gbs(2.0 * total_bytes, s.mean);
  // Guard against a pathological zero mean so we never divide by zero.
  const double speedup = (s.mean > 0.0) ? (baseline_mean / s.mean) : 0.0;

  print_row(variant, s, "GB/s", throughput);

  char extra[128];
  std::snprintf(extra, sizeof(extra), "ok=%d,speedup=%.2fx", ok ? 1 : 0, speedup);
  // The CSV "size" is the problem size (one copy of the buffer); the throughput
  // above already accounts for the two copies we perform per pipeline pass.
  csv_emit(csv, "streams", variant, total_bytes, "bytes",
           s, throughput, "GB/s", extra);

  // An explicit, human-readable speedup line derived from the measured means.
  std::printf("    speedup vs sync_pageable = %.2fx\n", speedup);
}

}  // namespace

int main(int argc, char** argv) {
  const int total_mb = arg_int(argc, argv, "total-mb", 64);
  const int warmup = arg_int(argc, argv, "warmup", 5);
  const int iters = arg_int(argc, argv, "iters", 30);
  const std::string csv = arg_value(argc, argv, "csv", "../results/streams.csv");

  // Identify the machine. Note: on Colab this link is shared/virtualised, so
  // absolute GB/s is lower than a bare-metal desktop; trust the ratios.
  const DeviceInfo& dev = device_info();
  std::printf("Device: %s (%s)\n", dev.name.c_str(), dev.arch.c_str());
  std::printf("Total: %d MiB of float, %d chunks -> measured bytes = 2 * total (H2D + D2H)\n\n",
              total_mb, kChunks);

  // One shared set of buffers for all variants (each variant overwrites them).
  Context ctx;
  setup(ctx, total_mb);

  // Create the non-blocking streams and their events BEFORE any copy is issued.
  // This ordering is deliberate: cudaMemcpyAsync on a stream is only meaningful
  // once that stream exists and is known not to sync with the legacy stream.
  constexpr int kMaxStreams = 3;
  cudaStream_t streams[kMaxStreams];
  cudaEvent_t events[kMaxStreams];
  for (int s = 0; s < kMaxStreams; ++s) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&events[s]));
  }

  // Each Stats below is the mean/std of `iters` full-pipeline timings. time_iters
  // wraps each repetition in a CudaTimer (CUDA events), so we report wall time
  // for the WHOLE 64 MB pipeline, not for any single kernel or single copy.
  const double problem_bytes = static_cast<double>(ctx.total_bytes);

  // ---- Variant 1: sync_pageable (baseline) --------------------------------
  std::printf("  %s\n", "sync_pageable (default stream, malloc buffers)");
  Stats t_sync_pageable = time_iters([&]() {
    for (int chunk = 0; chunk < kChunks; ++chunk) {
      run_chunk_sync(ctx, chunk, /*pinned=*/false);
    }
  }, warmup, iters);
  bool ok1 = verify(ctx, /*pinned=*/false);
  report(csv, "sync_pageable", t_sync_pageable, problem_bytes, t_sync_pageable.mean, ok1);
  if (!ok1) { std::fprintf(stderr, "FAILED: sync_pageable\n"); std::exit(EXIT_FAILURE); }
  std::printf("\n");

  // ---- Variant 2: sync_pinned (pinning alone, still one default stream) ----
  std::printf("  %s\n", "sync_pinned (default stream, cudaMallocHost buffers)");
  Stats t_sync_pinned = time_iters([&]() {
    for (int chunk = 0; chunk < kChunks; ++chunk) {
      run_chunk_sync(ctx, chunk, /*pinned=*/true);
    }
  }, warmup, iters);
  bool ok2 = verify(ctx, /*pinned=*/true);
  report(csv, "sync_pinned", t_sync_pinned, problem_bytes, t_sync_pageable.mean, ok2);
  if (!ok2) { std::fprintf(stderr, "FAILED: sync_pinned\n"); std::exit(EXIT_FAILURE); }
  std::printf("\n");

  // ---- Variant 3: streams2_pinned (2 non-blocking streams) ----------------
  std::printf("  %s\n", "streams2_pinned (chunks round-robin over 2 streams)");
  Stats t_streams2 = time_iters([&]() {
    run_chunk_streams<2>(ctx, streams, events);
  }, warmup, iters);
  bool ok3 = verify(ctx, /*pinned=*/true);
  report(csv, "streams2_pinned", t_streams2, problem_bytes, t_sync_pageable.mean, ok3);
  if (!ok3) { std::fprintf(stderr, "FAILED: streams2_pinned\n"); std::exit(EXIT_FAILURE); }
  std::printf("\n");

  // ---- Variant 4: streams3_pinned (3 non-blocking streams) ----------------
  // 4 chunks over 3 streams: stream 0 gets chunks 0 and 3, streams 1 and 2 get
  // one chunk each. The 4th chunk runs alone on stream 0 after its earlier work
  // has drained, so the tail cannot overlap -- this is why wall time is not a
  // clean 2x better than two streams.
  std::printf("  %s\n", "streams3_pinned (4 chunks over 3 streams, uneven tail)");
  Stats t_streams3 = time_iters([&]() {
    run_chunk_streams<3>(ctx, streams, events);
  }, warmup, iters);
  bool ok4 = verify(ctx, /*pinned=*/true);
  report(csv, "streams3_pinned", t_streams3, problem_bytes, t_sync_pageable.mean, ok4);
  if (!ok4) { std::fprintf(stderr, "FAILED: streams3_pinned\n"); std::exit(EXIT_FAILURE); }
  std::printf("\n");

  // No fifth "streams2_pinned_multiengine" variant: the copies in variants 3
  // and 4 are ALREADY cudaMemcpyAsync, so the copy engine and SMs already
  // overlap there, and H2D / D2H use different copy-engine directions. A
  // separate "multi-engine" row would just duplicate streams2/3_pinned with no
  // new mechanism to isolate, so we leave it out rather than invent a number.

  // Clean up every resource we created, in reverse order.
  for (int s = 0; s < kMaxStreams; ++s) {
    CUDA_CHECK(cudaEventDestroy(events[s]));
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }
  teardown(ctx);

  std::printf("Wrote 4 measured rows to %s\n", csv.c_str());
  return 0;
}
