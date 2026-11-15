#include "trellis_job.h"
#include "trellis_run.h"
#include "job_control.h"
#include "trellis_args.h"
#include "stb_image.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>
#ifdef TRELLIS_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace trellis {
namespace {
struct DeviceMonitor {
    std::atomic_bool stop{false};
    std::atomic<uint64_t> peak{0};
    std::atomic<size_t> stage{64};
    std::array<uint64_t, 64> stage_peaks{};
    uint64_t idle = 0;
    int device;
    std::thread thread;
    uint64_t used() {
#ifdef TRELLIS_USE_CUDA
        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) == cudaSuccess) return total - free;
#endif
        return 0;
    }
    explicit DeviceMonitor(int gpu) : device(gpu) {
#ifdef TRELLIS_USE_CUDA
        if (gpu < 0) return;
        if (cudaSetDevice(gpu) != cudaSuccess) throw std::runtime_error("Cannot select CUDA device");
        idle = used(); peak = idle;
        thread = std::thread([this] {
            cudaSetDevice(device);
            while (!stop.load(std::memory_order_relaxed)) {
                const uint64_t current = used();
                peak.store(std::max(peak.load(), current));
                const size_t index = stage.load();
                if (index < stage_peaks.size()) stage_peaks[index] = std::max(stage_peaks[index], current);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
#endif
    }
    void finish() {
        stop.store(true);
        if (thread.joinable()) thread.join();
    }
    ~DeviceMonitor() { finish(); }
    void collect(JobMetrics& metrics) {
        finish();
        metrics.idle_device_bytes = idle;
        metrics.sampled_peak_device_bytes = peak.load();
        metrics.idle_after_device_bytes = used();
        for (size_t i = 0; i < metrics.stages.size() && i < stage_peaks.size(); ++i)
            metrics.stages[i].sampled_peak_device_bytes = stage_peaks[i];
    }
};
}

JobResult run_job(const JobRequest& request, const std::atomic_bool* cancel,
                  const std::function<void(const JobProgress&)>& progress) {
    // Serializing the whole API also makes before/after residency measurements
    // meaningful for consecutive worker jobs. trellis_run serializes CLI/server use.
    static std::mutex job_mutex;
    std::lock_guard<std::mutex> lock(job_mutex);
    const auto start = std::chrono::steady_clock::now();
    JobResult result;
    result.metrics.seed = request.seed;
    RunControl control;
    control.cancel = cancel; control.progress = progress; control.metrics = &result.metrics;
    control.glb = &result.glb; control.capture_dir = request.capture_dir;
    try {
        control.check_cancel();
        if (request.resolution != 512 || request.atlas < kAtlasMin || request.atlas > kAtlasMax ||
            request.faces < kQemFacesMin || request.steps < 1 || request.steps > 1000 || request.gpu < 0 ||
            request.stream_flow_blocks < 0 || request.stream_flow_blocks > 30 ||
            (request.stream_flow_blocks && 30 % request.stream_flow_blocks) || request.ss_weight_budget_mib < -1 ||
            (request.ss_weight_budget_mib >= 0 && request.ss_engine.empty()))
            throw std::invalid_argument("Invalid 512-preset job settings");
        if (request.profile != "bf16" && request.profile != "int8" && request.profile != "native")
            throw std::invalid_argument("profile must be bf16, int8, or native");
        if (request.profile != "native" && (request.stream_flow_blocks || request.ss_weight_budget_mib != -1))
            throw std::invalid_argument("bf16/int8 profiles require resident weights; streaming is disabled");
        int width = 0, height = 0, channels = 0;
        if (request.image.empty() || request.image.size() > INT_MAX ||
            !stbi_info_from_memory(request.image.data(), static_cast<int>(request.image.size()), &width, &height, &channels))
            throw std::invalid_argument("Invalid or incomplete encoded image");
        if (!std::filesystem::is_directory(request.models)) throw std::invalid_argument("Model directory does not exist");
        ImageInput image(request.image.data(), request.image.size());
        control.image = &image;
        TrellisParams params;
        params.models = request.models; params.seed = request.seed; params.gpu = request.gpu;
        params.set_res(512); params.tex = request.atlas; params.faces = request.faces;
        params.steps = request.steps; params.webp = 0; params.ss_engine = request.ss_engine;
        params.profile = request.profile;
        params.stream_flow_blocks = request.stream_flow_blocks;
        params.ss_weight_budget_mib = request.ss_weight_budget_mib;
        params.birefnet = request.neural_background_removal ? -1 : 0;
        params.backend = "CUDA"; params.require_gpu = true;
        DeviceMonitor monitor(request.gpu);
        control.on_stage = [&](size_t index) { monitor.stage = index; };
        try {
            if (trellis_run(params, &control) != 0) throw std::runtime_error("Generation pipeline failed");
            if (result.glb.size() < 20 || std::memcmp(result.glb.data(), "glTF", 4))
                throw std::runtime_error("Pipeline returned an invalid GLB");
            control.check_cancel();
            result.status = JobStatus::Success;
        } catch (...) {
#ifdef TRELLIS_USE_CUDA
            cudaDeviceSynchronize();
#endif
            monitor.collect(result.metrics);
            control.on_stage = {};
            throw;
        }
        monitor.collect(result.metrics);
        control.on_stage = {};
    } catch (const JobCancelled& error) {
        result.status = JobStatus::Cancelled; result.error = error.what(); result.glb.clear();
    } catch (const std::exception& error) {
        result.status = JobStatus::Failed; result.error = error.what(); result.glb.clear();
    } catch (...) {
        result.status = JobStatus::Failed; result.error = "Unknown generation failure"; result.glb.clear();
    }
    control.finish_stage();
    result.metrics.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
}
