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
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>
#ifdef TRELLIS_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace trellis {
uint32_t job_api_version() noexcept { return kJobApiVersion; }
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
        if (request.image.empty() || request.image.size() > 32u * 1024u * 1024u || request.image.size() > INT_MAX ||
            !stbi_info_from_memory(request.image.data(), static_cast<int>(request.image.size()), &width, &height, &channels))
            throw std::invalid_argument("Invalid or incomplete encoded image");
        if (width < 1 || height < 1 || width > 8192 || height > 8192 || uint64_t(width) * height > 32000000)
            throw std::invalid_argument("Decoded image exceeds the 32 megapixel / 8192 dimension limit");
        if (request.image.size() > 32u * 1024u * 1024u) throw std::invalid_argument("Encoded image exceeds 32 MiB");
        if (!std::filesystem::is_directory(request.models)) {
            result.error_code = JobErrorCode::MissingModels;
            throw std::invalid_argument("Model directory does not exist");
        }
        int gpu = request.gpu;
#ifdef TRELLIS_USE_CUDA
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            result.error_code = JobErrorCode::DeviceUnavailable;
            throw std::runtime_error("No CUDA inference device is available");
        }
        const auto uuid_string = [](const cudaUUID_t& uuid) {
            const auto* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
            char value[41];
            std::snprintf(value, sizeof(value), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
            return std::string(value);
        };
        if (!request.device_uuid.empty()) {
            gpu = -1;
            for (int i = 0; i < count; ++i) {
                cudaDeviceProp property{};
                if (cudaGetDeviceProperties(&property, i) == cudaSuccess && uuid_string(property.uuid) == request.device_uuid) gpu = i;
            }
        }
        if (gpu < 0 || gpu >= count) {
            result.error_code = JobErrorCode::DeviceUnavailable;
            throw std::invalid_argument("Configured CUDA inference device is unavailable");
        }
        cudaDeviceProp property{};
        if (cudaGetDeviceProperties(&property, gpu) != cudaSuccess) throw std::runtime_error("Cannot inspect CUDA device");
        result.metrics.device_uuid = uuid_string(property.uuid);
        result.metrics.device_name = property.name;
        if (!request.expected_device_name.empty() && request.expected_device_name != result.metrics.device_name) {
            result.error_code = JobErrorCode::DeviceUnavailable;
            throw std::invalid_argument("TensorRT engine was built for " + request.expected_device_name +
                "; selected inference device is " + result.metrics.device_name);
        }
        if (cudaSetDevice(gpu) != cudaSuccess) {
            result.error_code = JobErrorCode::DeviceUnavailable;
            throw std::runtime_error("Cannot initialize the configured CUDA device");
        }
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) throw std::runtime_error("Cannot query CUDA memory admission");
        if (free_bytes < (uint64_t(6) << 30)) {
            result.error_code = JobErrorCode::ResourceLimit;
            throw std::runtime_error("Insufficient CUDA memory: the 512 worker requires 6 GiB free at admission, including insertion margin; available bytes=" + std::to_string(free_bytes));
        }
#endif
        ImageInput image(request.image.data(), request.image.size());
        control.image = &image;
        TrellisParams params;
        params.models = request.models; params.seed = request.seed; params.gpu = gpu;
        params.set_res(512); params.tex = request.atlas; params.faces = request.faces;
        params.steps = request.steps; params.webp = 0; params.ss_engine = request.ss_engine;
        params.profile = request.profile;
        params.stream_flow_blocks = request.stream_flow_blocks;
        params.ss_weight_budget_mib = request.ss_weight_budget_mib;
        params.birefnet = request.neural_background_removal ? -1 : 0;
        params.backend = "CUDA"; params.require_gpu = true;
        params.fa_fast = 0; // fixed preset: BF16 K/V and FP32 attention accumulation
        DeviceMonitor monitor(gpu);
        control.on_stage = [&](size_t index) {
            monitor.stage = index;
#ifdef TRELLIS_USE_CUDA
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) throw std::runtime_error("Cannot query stage memory pressure");
            const auto& stage = result.metrics.stages.at(index).stage;
            const bool gpu_model = stage == "preprocess" || stage == "conditioning" || stage == "ss_flow" ||
                stage == "ss_decode" || stage == "shape_flow" || stage == "shape_decode" ||
                stage == "texture_flow" || stage == "texture_decode";
            const uint64_t margin = gpu_model ? (uint64_t(3) << 30) : (uint64_t(256) << 20);
            if (free_bytes < margin) {
                result.error_code = JobErrorCode::ResourceLimit;
                throw std::runtime_error("CUDA memory pressure before " + stage + "; available bytes=" + std::to_string(free_bytes));
            }
#endif
        };
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
        result.error_code = JobErrorCode::Cancelled;
        result.status = JobStatus::Cancelled; result.error = error.what(); result.glb.clear();
    } catch (const std::invalid_argument& error) {
        if (result.error_code == JobErrorCode::None) result.error_code = JobErrorCode::InvalidArgument;
        result.status = JobStatus::Failed; result.error = error.what(); result.glb.clear();
    } catch (const std::exception& error) {
        if (result.error_code == JobErrorCode::None) result.error_code = JobErrorCode::Backend;
        result.status = JobStatus::Failed; result.error = error.what(); result.glb.clear();
    } catch (...) {
        result.error_code = JobErrorCode::Backend;
        result.status = JobStatus::Failed; result.error = "Unknown generation failure"; result.glb.clear();
    }
    control.finish_stage();
    result.metrics.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
}
