#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace trellis {
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
    std::optional<uint64_t> sampled_peak_device_bytes; // CUDA cudaMemGetInfo
    std::optional<uint64_t> sampled_peak_footprint_bytes; // macOS process physical footprint
    double setup_seconds = 0;
    double first_forward_seconds = 0;
    double later_forwards_seconds = 0;
    int forwards = 0;
};
struct JobMetrics {
    uint32_t seed = 0;
    double seconds = 0;
    std::optional<uint64_t> idle_device_bytes;
    std::optional<uint64_t> sampled_peak_device_bytes;
    std::optional<uint64_t> idle_after_device_bytes;
    uint64_t repaired_nonfinite = 0;
    uint64_t clamped_guidance = 0;
    std::vector<StageMetrics> stages;
    std::string device;
    std::optional<uint64_t> idle_footprint_bytes;
    std::optional<uint64_t> sampled_peak_footprint_bytes;
    std::optional<uint64_t> idle_after_footprint_bytes;
};
enum class JobStatus { Success, Cancelled, Failed };
struct JobRequest {
    std::vector<uint8_t> image;  // encoded PNG/JPEG/etc.; borrowed only during run_job
    std::string models = "models/f16";
    std::string ss_engine;      // legacy path, implies TensorRT
    std::string capture_dir;    // optional native forward/noise fixtures, for offline parity
    uint32_t seed = 42;         // explicit, including zero
    int gpu = 0;
    int resolution = 512;      // first library preset; other values are rejected
    int atlas = 1024;
    int faces = 150000;
    int steps = 12;
    bool neural_background_removal = true; // false uses existing alpha / white-background threshold
    std::string device;        // metal, cuda or cpu; default Metal on Metal builds, CUDA otherwise
    std::string ss_backend;    // ggml, tensorrt or mlx; empty infers from ss_engine/ss_weights
    std::string ss_weights;    // original MLX SS safetensors checkpoint
};
struct JobResult {
    JobStatus status = JobStatus::Failed;
    std::vector<uint8_t> glb;
    JobMetrics metrics;
    std::string error;
};
// Synchronous worker-thread API. Pipeline calls are serialized because upstream
// GGML runtime flags are process-wide. Callbacks execute on the calling thread;
// they must not start another job. Cancellation is checked between GPU forwards
// and stages, and all in-flight work is drained before returning.
JobResult run_job(const JobRequest& request, const std::atomic_bool* cancel = nullptr,
                  const std::function<void(const JobProgress&)>& progress = {});
}
