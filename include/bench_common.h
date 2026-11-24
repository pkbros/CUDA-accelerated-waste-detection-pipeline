// ===========================================================================
// bench_common.h -- shared timing / checking / CSV helpers for Phase A.
//
// WHY THIS FILE EXISTS (read this before touching any *.cu):
//   Every benchmark in this repo must be measured the same way, or the numbers
//   in results/summary.md are not comparable. So all of them:
//
//     1. warm up the GPU first (clock ramp + JIT of the context),
//     2. time each iteration individually with cudaEvent pairs,
//     3. report mean and sample standard deviation over N iterations,
//     4. verify the GPU answer against a CPU reference and record the error,
//     5. append one row to a CSV with a single fixed schema:
//
//   bench,variant,size,size_unit,time_ms_mean,time_ms_std,throughput,throughput_unit,extra,device,arch
//
//   The notebook never prints hand-written numbers into the summary. It reads
//   these CSVs. That is the only defence against fabricated results.
//
//   Header-only on purpose: each .cu file is compiled on its own with plain
//   nvcc, so there is no library to build or link.
// ===========================================================================
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// 1. Error checking
// ---------------------------------------------------------------------------
// Every CUDA API call in this repo is wrapped in CUDA_CHECK so a bad launch
// stops the program with file/line instead of silently producing garbage that
// later shows up as a "slow kernel" or a "wrong result".
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                 \
                   cudaGetErrorName(err__), __FILE__, __LINE__,              \
                   cudaGetErrorString(err__));                               \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Same idea, but for the *previous* launch: catches bad grid/block configs and
// illegal memory accesses that only surface after the kernel finishes.
#define CUDA_CHECK_LAST()                                                    \
  do {                                                                       \
    cudaError_t err__ = cudaGetLastError();                                   \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "Kernel launch error %s at %s:%d -> %s\n",        \
                   cudaGetErrorName(err__), __FILE__, __LINE__,              \
                   cudaGetErrorString(err__));                               \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// 2. Statistics
// ---------------------------------------------------------------------------
struct Stats {
  double mean = 0.0;    // arithmetic mean of the per-iteration timings (ms)
  double stddev = 0.0;  // sample standard deviation (n-1 denominator)
  double min = 0.0;
  double max = 0.0;
  size_t n = 0;
};

// Sample stddev, not population: we are estimating the spread of the "true"
// time from a sample of runs. With iters >= 30 the difference is tiny, but the
// choice is stated here so nobody has to guess later.
inline Stats stats_of(const std::vector<double>& v) {
  Stats s;
  s.n = v.size();
  if (v.empty()) return s;
  double sum = 0.0;
  for (double x : v) sum += x;
  s.mean = sum / static_cast<double>(v.size());
  double acc = 0.0;
  for (double x : v) acc += (x - s.mean) * (x - s.mean);
  s.stddev = v.size() > 1 ? std::sqrt(acc / static_cast<double>(v.size() - 1)) : 0.0;
  s.min = *std::min_element(v.begin(), v.end());
  s.max = *std::max_element(v.begin(), v.end());
  return s;
}

// ---------------------------------------------------------------------------
// 3. Timing with CUDA events
// ---------------------------------------------------------------------------
// cudaEventRecord timestamps are written by the GPU itself, so they measure
// device time and are unaffected by CPU scheduling jitter. Wall-clock timers
// around an async launch would only measure the launch, which is why the PRD
// asks for events.
class CudaTimer {
 public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;

  void start() { CUDA_CHECK(cudaEventRecord(start_)); }

  // Synchronises on the stop event, so the value is final when this returns.
  // Synchronising inside the timed region's *end* (not per iteration start of
  // the next) is what keeps iterations independent instead of pipelined.
  void stop() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
  }

  double ms() {
    float value = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&value, start_, stop_));
    return static_cast<double>(value);
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

// Runs `fn` `warmup` times untimed, then `iters` times timed, one measurement
// per iteration. `fn` must only *enqueue* work (a kernel launch, a memcpy, a
// cuBLAS call): the timer isolates iterations for you.
//
// Why not one event pair around the whole loop? Because then you get a single
// number and no stddev, and the PRD requires mean AND std.
template <typename Fn>
Stats time_iters(Fn&& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());  // do not let warmup leak into timing

  CudaTimer timer;
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    timer.start();
    fn();
    timer.stop();
    samples.push_back(timer.ms());
  }
  return stats_of(samples);
}

// ---------------------------------------------------------------------------
// 4. Correctness checking against a CPU reference
// ---------------------------------------------------------------------------
struct MaxError {
  double abs_err = 0.0;   // max |gpu - cpu|
  double rel_err = 0.0;   // max |gpu - cpu| / (|cpu| + 1e-12)
  bool ok = false;        // within rtol/atol
};

// Reports a *stated* tolerance check, as the PRD requires. Note this is a
// per-element check, not an L2 norm: for reductions and GEMM the reviewer wants
// to know the worst element, not an average that can hide a broken tile.
inline MaxError compare_gpu_cpu(const float* gpu, const float* cpu, size_t n,
                                double rtol, double atol) {
  MaxError e;
  for (size_t i = 0; i < n; ++i) {
    const double g = static_cast<double>(gpu[i]);
    const double c = static_cast<double>(cpu[i]);
    const double a = std::fabs(g - c);
    const double r = a / (std::fabs(c) + 1e-12);
    if (a > e.abs_err) e.abs_err = a;
    if (r > e.rel_err) e.rel_err = r;
  }
  for (size_t i = 0; i < n; ++i) {
    const double g = static_cast<double>(gpu[i]);
    const double c = static_cast<double>(cpu[i]);
    if (std::fabs(g - c) > atol + rtol * std::fabs(c)) {
      e.ok = false;
      return e;
    }
  }
  e.ok = true;
  return e;
}

// ---------------------------------------------------------------------------
// 5. Derived throughput metrics
// ---------------------------------------------------------------------------
// bytes moved / seconds, in GB/s (10^9, the convention in vendor datasheets).
inline double gbs(double bytes, double ms) {
  if (ms <= 0.0) return 0.0;
  return (bytes / 1e9) / (ms / 1e3);
}

// FLOPS / seconds, in GFLOP/s. `flops` is a hand-counted operation count, not a
// measured quantity -- document how you counted it next to the kernel.
inline double gflops(double flops, double ms) {
  if (ms <= 0.0) return 0.0;
  return (flops / 1e9) / (ms / 1e3);
}

// ---------------------------------------------------------------------------
// 6. Device info (recorded once per machine, so results are reproducible)
// ---------------------------------------------------------------------------
struct DeviceInfo {
  std::string name;
  int major = 0;
  int minor = 0;
  int sm_count = 0;
  size_t total_mem = 0;
  std::string arch;  // e.g. "sm_75"
};

inline const DeviceInfo& device_info() {
  static DeviceInfo info = [] {
    DeviceInfo d;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    d.name = prop.name;
    d.major = prop.major;
    d.minor = prop.minor;
    d.sm_count = prop.multiProcessorCount;
    d.total_mem = static_cast<size_t>(prop.totalGlobalMem);
    d.arch = "sm_" + std::to_string(prop.major * 10 + prop.minor);
    return d;
  }();
  return info;
}

// ---------------------------------------------------------------------------
// 7. Filesystem + CSV writing (fixed schema, append-only)
// ---------------------------------------------------------------------------
inline void make_one_dir(const std::string& p) {
#ifdef _WIN32
  _mkdir(p.c_str());
#else
  mkdir(p.c_str(), 0755);
#endif
}

inline void make_dirs(const std::string& dir) {
  std::string cur;
  for (size_t i = 0; i < dir.size(); ++i) {
    cur.push_back(dir[i]);
    if (dir[i] == '/' || dir[i] == '\\') {
      if (cur.size() > 1) make_one_dir(cur);
    }
  }
  if (!dir.empty()) make_one_dir(dir);
}

inline std::string csv_escape(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\"\"";
    else out += c;
  }
  out += '"';
  return out;
}

// One header for every Phase A benchmark. Keep this in sync with
// phase_b/common.py CSV_COLUMNS -- the summary generator depends on it.
inline std::string csv_header() {
  return "bench,variant,size,size_unit,time_ms_mean,time_ms_std,throughput,"
         "throughput_unit,extra,device,arch";
}

// `fields` must be given in header order. Missing trailing fields are written
// empty so a kernel that has no natural throughput metric can still emit a row.
inline void csv_append(const std::string& path, const std::vector<std::string>& fields) {
  const size_t kColumns = 11;
  // Create parent directories so `--csv ../results/x.csv` always works.
  const size_t slash = path.find_last_of("/\\");
  if (slash != std::string::npos) make_dirs(path.substr(0, slash));

  bool need_header = true;
  if (FILE* probe = std::fopen(path.c_str(), "rb")) {
    std::fseek(probe, 0, SEEK_END);
    need_header = std::ftell(probe) == 0;
    std::fclose(probe);
  }

  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) {
    std::fprintf(stderr, "cannot open %s for append\n", path.c_str());
    return;
  }
  if (need_header) std::fprintf(f, "%s\n", csv_header().c_str());

  for (size_t i = 0; i < kColumns; ++i) {
    const std::string v = i < fields.size() ? csv_escape(fields[i]) : std::string();
    std::fprintf(f, "%s%s", i ? "," : "", v.c_str());
  }
  std::fprintf(f, "\n");
  std::fclose(f);
}

// Convenience wrapper: the common case of one benchmark row.
inline void csv_emit(const std::string& path, const std::string& bench,
                     const std::string& variant, double size,
                     const std::string& size_unit, const Stats& s,
                     double throughput, const std::string& throughput_unit,
                     const std::string& extra) {
  const DeviceInfo& d = device_info();
  char size_buf[64], t_buf[96], tp_buf[64], mean_buf[64], std_buf[64];
  std::snprintf(size_buf, sizeof(size_buf), "%.0f", size);
  std::snprintf(mean_buf, sizeof(mean_buf), "%.6f", s.mean);
  std::snprintf(std_buf, sizeof(std_buf), "%.6f", s.stddev);
  std::snprintf(t_buf, sizeof(t_buf), "%zu", s.n);
  std::snprintf(tp_buf, sizeof(tp_buf), "%.4f", throughput);

  csv_append(path, {bench, variant, size_buf, size_unit, mean_buf, std_buf,
                    tp_buf, throughput_unit, extra, d.name, d.arch});
}

// Printed so the run is readable while it happens; the CSV is the artifact.
inline void print_row(const std::string& variant, const Stats& s,
                      const std::string& throughput_unit, double throughput) {
  std::printf("  %-22s %9.4f ms  +-%-8.4f  (n=%zu)  %9.2f %s\n", variant.c_str(),
              s.mean, s.stddev, s.n, throughput, throughput_unit.c_str());
}

// Records the machine the numbers came from. The summary generator reads this
// so every table can state its GPU (a T4 result must never be presented as if
// it came from an A100).
inline void write_device_info(const std::string& dir = "../results") {
  make_dirs(dir);
  const DeviceInfo& d = device_info();
  const std::string path = dir + "/device.txt";
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  std::fprintf(f, "gpu_name=%s\n", d.name.c_str());
  std::fprintf(f, "compute_capability=%d.%d\n", d.major, d.minor);
  std::fprintf(f, "arch=%s\n", d.arch.c_str());
  std::fprintf(f, "multiprocessors=%d\n", d.sm_count);
  std::fprintf(f, "global_mem_gib=%.2f\n",
               static_cast<double>(d.total_mem) / (1024.0 * 1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
// 8. Tiny CLI parser: --key value and --key=value
// ---------------------------------------------------------------------------
inline std::string arg_value(int argc, char** argv, const std::string& key,
                             const std::string& fallback) {
  const std::string flag = "--" + key;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == flag && i + 1 < argc) return argv[i + 1];
    if (a.rfind(flag + "=", 0) == 0) return a.substr(flag.size() + 1);
  }
  return fallback;
}

inline int arg_int(int argc, char** argv, const std::string& key, int fallback) {
  const std::string v = arg_value(argc, argv, key, "");
  return v.empty() ? fallback : std::atoi(v.c_str());
}

inline long long arg_ll(int argc, char** argv, const std::string& key,
                        long long fallback) {
  const std::string v = arg_value(argc, argv, key, "");
  return v.empty() ? fallback : std::atoll(v.c_str());
}

inline double arg_double(int argc, char** argv, const std::string& key,
                         double fallback) {
  const std::string v = arg_value(argc, argv, key, "");
  return v.empty() ? fallback : std::atof(v.c_str());
}
