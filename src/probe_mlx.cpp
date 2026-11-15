#include "mlx_flow.h"
#include "npy.h"
#include <mlx/mlx.h>
#include <mlx/backend/metal/metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

static void probe_attention(const std::string& path, int keys, const std::string& precision) {
    namespace mx = mlx::core;
    if (keys != 4096 && keys != 1029) throw std::invalid_argument("Expected 4096 or 1029 attention keys");
    mx::Dtype dtype = mx::bfloat16;
    if (precision == "fp32") dtype = mx::float32;
    else if (precision == "fp16") dtype = mx::float16;
    else if (precision != "bf16") throw std::invalid_argument("Expected bf16, fp16 or fp32");
    auto stream = mx::new_stream(mx::Device::gpu);
    constexpr float scale = 1.0f / 11.313708499f;
    auto q = mx::full({1,4096,12,128}, mx::array(0.1f), dtype, stream);
    auto k = mx::full({1,keys,12,128}, mx::array(-0.1f), dtype, stream);
    auto v = mx::ones({1,keys,12,128}, dtype, stream);
    mx::eval(q, k, v);
    mx::synchronize(stream);
    auto head_major = [&](const mx::array& input) { return mx::transpose(input, {0,2,1,3}, stream); };
    mx::clear_cache();
    mx::reset_peak_memory();
    const size_t baseline = mx::get_active_memory();
    const bool capture = path != "-";
    if (capture) mx::metal::start_capture(std::filesystem::absolute(path).string());
    auto attention = mx::fast::scaled_dot_product_attention(head_major(q), head_major(k), head_major(v),
                                                              scale, "", {}, {}, true, stream);
    attention.eval();
    mx::synchronize(stream);
    if (capture) mx::metal::stop_capture();
    const size_t additional = mx::get_peak_memory() - baseline;
    const size_t scores = size_t(12) * 4096 * keys * dtype.size();
    const float mean = mx::mean(mx::astype(attention, mx::float32, stream), false, stream).item<float>();
    printf("keys=%d dtype=%s baseline=%zu additional_peak=%zu dense_scores=%zu mean=%.6f\n",
           keys, precision.c_str(), baseline, additional, scores, mean);
    if (additional >= scores || !std::isfinite(mean) || std::abs(mean - 1.f) > 0.005f)
        throw std::runtime_error("Attention probe exceeded scratch or numerical gate");
}

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "attention") {
        try {
            probe_attention(argv[2], std::stoi(argv[3]), argv[4]);
            return 0;
        } catch (const std::exception& error) {
            fprintf(stderr, "%s\n", error.what());
            return 1;
        }
    }
    if (argc != 6) {
        fprintf(stderr, "usage: trellis-probe-mlx WEIGHTS FIXTURE_DIR OUTPUT_DIR bf16|fp16|fp32 block|full\n");
        fprintf(stderr, "       trellis-probe-mlx attention TRACE.gputrace|- 4096|1029 bf16|fp16|fp32\n");
        return 2;
    }
    try {
        namespace fs = std::filesystem;
        const fs::path fixture = argv[2], out = argv[3];
        const std::string precision = argv[4], mode = argv[5];
        fs::create_directories(out);
        const auto latent = npy::load((fixture / "latent.npy").string());
        const auto timestep = npy::load((fixture / "timestep.npy").string());
        const auto cond = npy::load((fixture / "cond.npy").string());
        if (latent.shape != std::vector<int64_t>{1,4096,8} || timestep.shape != std::vector<int64_t>{1} ||
            cond.shape != std::vector<int64_t>{1,1029,1024})
            throw std::runtime_error("Unexpected MLX probe input shape");
        const auto start = std::chrono::steady_clock::now();
        trellis::MlxFlowRunner runner(argv[1], precision);
        const auto loaded = std::chrono::steady_clock::now();
        if (mode == "block") runner.probe_block(latent.data, timestep.data[0], cond.data.data(), out.string());
        else if (mode == "full") {
            auto velocity = runner.forward(latent.data, timestep.data[0], cond.data.data());
            npy::save((out / "velocity.npy").string(), velocity.data(), {1,4096,8});
        } else throw std::invalid_argument("MLX probe mode must be block or full");
        const auto done = std::chrono::steady_clock::now();
        printf("load_seconds=%.6f probe_seconds=%.6f\n",
               std::chrono::duration<double>(loaded-start).count(),
               std::chrono::duration<double>(done-loaded).count());
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
