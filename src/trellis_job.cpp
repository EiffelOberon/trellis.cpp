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
#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace trellis {
namespace {
enum class MemoryCounter { Unavailable, CudaDevice, ProcessFootprint };

void record_peak(std::optional<uint64_t>& peak, std::optional<uint64_t> sample) {
    if (sample && (!peak || *sample > *peak)) peak = sample;
}

struct DeviceMonitor {
    std::atomic_bool stop{false};
    std::atomic<size_t> stage{64};
    std::array<std::optional<uint64_t>, 64> stage_peaks{};
    std::optional<uint64_t> idle, peak;
    MemoryCounter counter = MemoryCounter::Unavailable;
    std::thread thread;

    std::optional<uint64_t> used() const {
#ifdef TRELLIS_USE_CUDA
        if (counter == MemoryCounter::CudaDevice) {
            size_t free = 0, total = 0;
            if (cudaMemGetInfo(&free, &total) == cudaSuccess) return total - free;
        }
#endif
#ifdef __APPLE__
        if (counter == MemoryCounter::ProcessFootprint) {
            task_vm_info_data_t info{};
            mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
            if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS &&
                count >= TASK_VM_INFO_REV1_COUNT)
                return info.phys_footprint;
        }
#endif
        return std::nullopt;
    }

    DeviceMonitor(const std::string& device, [[maybe_unused]] int gpu) {
#ifdef TRELLIS_USE_CUDA
        if (device == "cuda") {
            if (cudaSetDevice(gpu) != cudaSuccess) throw std::runtime_error("Cannot select CUDA device");
            counter = MemoryCounter::CudaDevice;
        }
#endif
#ifdef __APPLE__
        if (device == "metal" || device == "cpu") counter = MemoryCounter::ProcessFootprint;
#endif
        if (counter == MemoryCounter::Unavailable) return;
        idle = peak = used();
        thread = std::thread([this
#ifdef TRELLIS_USE_CUDA
                              , gpu
#endif
        ] {
#ifdef TRELLIS_USE_CUDA
            if (counter == MemoryCounter::CudaDevice && cudaSetDevice(gpu) != cudaSuccess) return;
#endif
            while (!stop.load(std::memory_order_relaxed)) {
                const size_t index = stage.load();
                auto sample = used();
                record_peak(peak, sample);
                if (index < stage_peaks.size()) record_peak(stage_peaks[index], sample);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }

    void finish() {
        stop.store(true);
        if (thread.joinable()) thread.join();
    }
    ~DeviceMonitor() { finish(); }

    void drain() const {
#ifdef TRELLIS_USE_CUDA
        if (counter == MemoryCounter::CudaDevice) cudaDeviceSynchronize();
#endif
        // GGML drains the owning backend/scheduler before freeing its buffers.
    }

    void collect(JobMetrics& metrics) {
        finish();
        auto after = used();
        record_peak(peak, after);
        const bool cuda = counter == MemoryCounter::CudaDevice;
        if (cuda) {
            metrics.idle_device_bytes = idle;
            metrics.sampled_peak_device_bytes = peak;
            metrics.idle_after_device_bytes = after;
        } else if (counter == MemoryCounter::ProcessFootprint) {
            metrics.idle_footprint_bytes = idle;
            metrics.sampled_peak_footprint_bytes = peak;
            metrics.idle_after_footprint_bytes = after;
        }
        for (size_t i = 0; i < metrics.stages.size() && i < stage_peaks.size(); ++i) {
            auto& stage_metrics = metrics.stages[i];
            (cuda ? stage_metrics.sampled_peak_device_bytes : stage_metrics.sampled_peak_footprint_bytes) = stage_peaks[i];
        }
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
            request.faces < kQemFacesMin || request.steps < 1 || request.steps > 1000 || request.gpu < 0)
            throw std::invalid_argument("Invalid 512-preset job settings");
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
        params.steps = request.steps; params.webp = 0;
        params.ss_engine = request.ss_engine; params.ss_backend = request.ss_backend;
        params.ss_weights = request.ss_weights; params.device = request.device;
        params.birefnet = request.neural_background_removal ? -1 : 0;
        if (params.device.empty()) {
#ifdef TRELLIS_USE_METAL
            params.device = "metal";
#else
            params.device = "cuda";
#endif
        }
        auto validated = params;
        resolve_execution(validated);
        DeviceMonitor monitor(params.device, request.gpu);
        control.on_stage = [&](size_t index) { monitor.stage = index; };
        try {
            if (trellis_run(params, &control) != 0) throw std::runtime_error("Generation pipeline failed");
            if (result.glb.size() < 20 || std::memcmp(result.glb.data(), "glTF", 4))
                throw std::runtime_error("Pipeline returned an invalid GLB");
            control.check_cancel();
            result.status = JobStatus::Success;
        } catch (...) {
            monitor.drain();
            control.finish_stage();
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
