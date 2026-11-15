#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#if defined(_WIN32) && defined(TRELLIS_JOB_SHARED)
# if defined(TRELLIS_JOB_EXPORTS)
#  define TRELLIS_JOB_API __declspec(dllexport)
# else
#  define TRELLIS_JOB_API __declspec(dllimport)
# endif
#else
# define TRELLIS_JOB_API
#endif
#if defined(TRELLIS_JOB_SHARED) && defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL != 0
# error "The Trellis worker SDK requires the matched Release toolchain and CRT"
#endif

namespace trellis {
inline constexpr uint32_t kJobApiVersion = 2;
TRELLIS_JOB_API uint32_t job_api_version() noexcept;
struct JobProgress {
    std::string stage;
    int step = 0;
    int steps = 0;
};
struct StageMetrics {
    std::string stage;
    std::string backend;
    std::string precision;
    double seconds = 0;
    uint64_t sampled_peak_device_bytes = 0; // cudaMemGetInfo; see benchmark.py for device-wide WDDM usage
};
struct JobMetrics {
    std::string device_uuid;
    std::string device_name;
    uint32_t seed = 0;
    double seconds = 0;
    uint64_t idle_device_bytes = 0;
    uint64_t sampled_peak_device_bytes = 0;
    uint64_t idle_after_device_bytes = 0;
    uint64_t repaired_nonfinite = 0;
    uint64_t clamped_guidance = 0;
    std::vector<StageMetrics> stages;
};
enum class JobStatus { Success, Cancelled, Failed };
enum class JobErrorCode { None, InvalidArgument, MissingModels, DeviceUnavailable, Backend, Cancelled, ResourceLimit };
struct JobRequest {
    std::vector<uint8_t> image;  // encoded PNG/JPEG/etc.; borrowed only during run_job
    std::string models = "models/f16";
    std::string profile = "bf16"; // default resident BF16 SS; int8 selects resident quantized flows
    std::string ss_engine;      // empty resolves from profile; profile=native allows explicit legacy selection
    std::string capture_dir;    // optional native forward/noise fixtures, for offline parity
    uint32_t seed = 42;         // explicit, including zero
    int gpu = 0;
    std::string device_uuid;   // optional CUDA UUID; takes precedence over ordinal
    std::string expected_device_name; // optional engine provenance; no cross-device plan fallback
    int resolution = 512;      // first library preset; other values are rejected
    int atlas = 1024;
    int faces = 150000;
    int steps = 12;
    int stream_flow_blocks = 0; // CUDA sparse flows: 0 resident, or a positive divisor of 30
    int ss_weight_budget_mib = -1; // -1 engine default; >=0 requires a streaming-enabled engine
    bool neural_background_removal = true; // false uses existing alpha / white-background threshold
};
struct JobResult {
    JobStatus status = JobStatus::Failed;
    JobErrorCode error_code = JobErrorCode::None;
    std::vector<uint8_t> glb;
    JobMetrics metrics;
    std::string error;
};
// Synchronous worker-thread API. Pipeline calls are serialized because upstream
// GGML runtime flags are process-wide. Callbacks execute on the calling thread;
// they must not start another job. Cancellation is checked between GPU forwards
// and stages, and all in-flight work is drained before returning.
TRELLIS_JOB_API JobResult run_job(const JobRequest& request, const std::atomic_bool* cancel = nullptr,
                  const std::function<void(const JobProgress&)>& progress = {});
}
