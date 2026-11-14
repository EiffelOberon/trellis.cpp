// Standalone consumer of exactly the worker-thread API intended for Glint.
#include "trellis_job.h"
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <thread>

static std::string quote_json(const std::string& value) {
    const char* hex = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { result += '\\'; result += char(ch); }
        else if (ch < 32) { result += "\\u00"; result += hex[ch >> 4]; result += hex[ch & 15]; }
        else result += char(ch);
    }
    return result + '"';
}

static void metrics(const std::string& path, const trellis::JobResult& result) {
    std::ofstream out(path);
    const auto& m = result.metrics;
    out << "{\n  \"status\": " << quote_json(result.status == trellis::JobStatus::Success ? "success" :
        result.status == trellis::JobStatus::Cancelled ? "cancelled" : "failed")
        << ",\n  \"error\": " << quote_json(result.error) << ",\n  \"seed\": " << m.seed
        << ",\n  \"seconds\": " << m.seconds << ",\n  \"idle_device_bytes\": " << m.idle_device_bytes
        << ",\n  \"sampled_peak_device_bytes\": " << m.sampled_peak_device_bytes
        << ",\n  \"idle_after_device_bytes\": " << m.idle_after_device_bytes
        << ",\n  \"repaired_nonfinite\": " << m.repaired_nonfinite
        << ",\n  \"clamped_guidance\": " << m.clamped_guidance << ",\n  \"stages\": [";
    for (size_t i = 0; i < m.stages.size(); ++i) {
        const auto& stage = m.stages[i];
        if (i) out << ',';
        out << "\n    {\"stage\": " << quote_json(stage.stage) << ", \"backend\": " << quote_json(stage.backend)
            << ", \"precision\": " << quote_json(stage.precision) << ", \"seconds\": " << stage.seconds
            << ", \"sampled_peak_device_bytes\": " << stage.sampled_peak_device_bytes << '}';
    }
    out << "\n  ]\n}\n";
    if (!out) throw std::runtime_error("Cannot write job metrics");
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: trellis-job IMAGE OUTPUT.glb MODELS [--ss-engine ENGINE] [--capture-dir DIR] [--repeat N] [--exercise-lifecycle]\n");
        return 2;
    }
    try {
        trellis::JobRequest request;
        std::ifstream image(argv[1], std::ios::binary);
        if (!image) throw std::runtime_error("Cannot open input image");
        request.image.assign(std::istreambuf_iterator<char>(image), {});
        request.models = argv[3];
        bool lifecycle = false; int repeats = 1;
        for (int i = 4; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--exercise-lifecycle") lifecycle = true;
            else if (arg == "--ss-engine" && i + 1 < argc) request.ss_engine = argv[++i];
            else if (arg == "--capture-dir" && i + 1 < argc) request.capture_dir = argv[++i];
            else if (arg == "--repeat" && i + 1 < argc) repeats = std::stoi(argv[++i]);
            else throw std::invalid_argument("Unknown/incomplete argument: " + arg);
        }
        if (repeats < 1) throw std::invalid_argument("Repeat count must be positive");
        int exit_code = 0;
        std::thread worker([&] {
            try {
                const int count = lifecycle ? 4 : repeats;
                for (int i = 0; i < count; ++i) {
                    auto job = request;
                    std::atomic_bool cancel{false};
                    const bool cancel_job = lifecycle && i == 1;
                    const bool fail_job = lifecycle && i == 2;
                    if (fail_job) job.ss_engine = "missing-engine-for-lifecycle-test.plan";
                    auto result = trellis::run_job(job, &cancel, [&](const trellis::JobProgress& p) {
                        fprintf(stderr, "[job %d] %s %d/%d\n", i, p.stage.c_str(), p.step, p.steps);
                        if (cancel_job && p.stage == "ss_flow" && p.step >= 1) cancel = true;
                    });
                    const std::string stem = std::string(argv[2]) + ".job" + std::to_string(i);
                    metrics(stem + ".json", result);
                    const auto expected = cancel_job ? trellis::JobStatus::Cancelled : fail_job ? trellis::JobStatus::Failed : trellis::JobStatus::Success;
                    if (result.status != expected) throw std::runtime_error("Unexpected job status: " + result.error);
                    if (result.status == trellis::JobStatus::Success) {
                        std::ofstream output(i == 0 ? std::string(argv[2]) : stem + ".glb", std::ios::binary);
                        output.write(reinterpret_cast<const char*>(result.glb.data()), result.glb.size());
                        if (!output) throw std::runtime_error("Cannot write output GLB");
                    }
                    fprintf(stderr, "[job %d] completed in %.3fs; idle_after=%llu, peak=%llu bytes\n", i,
                            result.metrics.seconds, (unsigned long long)result.metrics.idle_after_device_bytes,
                            (unsigned long long)result.metrics.sampled_peak_device_bytes);
                }
            } catch (const std::exception& error) {
                fprintf(stderr, "%s\n", error.what()); exit_code = 1;
            }
        });
        worker.join();
        return exit_code;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
