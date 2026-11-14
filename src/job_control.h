#pragma once
#include "trellis_job.h"
#include "preprocess.h"
#include <chrono>
#include <stdexcept>

namespace trellis {
struct JobCancelled : std::runtime_error {
    JobCancelled() : std::runtime_error("Generation cancelled") {}
};
struct RunControl {
    const std::atomic_bool* cancel = nullptr;
    std::function<void(const JobProgress&)> progress;
    JobMetrics* metrics = nullptr;
    const ImageInput* image = nullptr;
    std::vector<uint8_t>* glb = nullptr;
    std::string capture_dir;
    std::string current_stage;
    std::chrono::steady_clock::time_point stage_start;
    std::function<void(size_t)> on_stage;
    void check_cancel() const {
        if (cancel && cancel->load(std::memory_order_relaxed)) throw JobCancelled();
    }
    void check(int step = 0, int steps = 0) {
        check_cancel();
        if (progress) progress({current_stage, step, steps});
        check_cancel();
    }
    void finish_stage() {
        if (metrics && !current_stage.empty())
            metrics->stages.back().seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stage_start).count();
        current_stage.clear();
    }
    void stage(const char* name, const char* backend = "native-cpu", const char* precision = "fp32") {
        finish_stage();
        current_stage = name;
        stage_start = std::chrono::steady_clock::now();
        if (metrics) metrics->stages.push_back({name, backend, precision, 0});
        if (on_stage && metrics) on_stage(metrics->stages.size() - 1);
        check();
    }
};
}
