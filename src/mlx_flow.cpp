#include "mlx_flow.h"
#include "npy.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <stdexcept>

namespace trellis {
namespace {
namespace mx = mlx::core;
using mx::array;
constexpr int tokens = 4096, channels = 8, ctx_tokens = 1029, ctx_channels = 1024;
constexpr int width = 1536, heads = 12, head_width = 128, ff_width = 8192, block_count = 30;
constexpr float attention_scale = 1.0f / 11.313708499f;

// MLX has no single-stream destruction API. Reuse two private streams even when
// consecutive jobs arrive on different threads. The lease covers pending work.
std::mutex mlx_mutex;
mx::Stream inference_stream() {
    static const auto stream = mx::new_thread_unsafe_stream(mx::Device::gpu);
    return stream;
}

mx::Stream load_stream() {
    static const auto stream = mx::new_thread_unsafe_stream(mx::Device::cpu);
    return stream;
}

struct Linear {
    array weights, bias;
};

struct Block {
    array modulation, norm_weight, norm_bias;
    Linear self_qkv, self_out, cross_q, cross_kv, cross_out, ff_in, ff_out;
    array self_q_norm, self_k_norm, cross_q_norm, cross_k_norm;
};

struct WeightReader {
    std::unordered_map<std::string, array> tensors;
    std::vector<array> prepared;
    mx::Stream stream;
    mx::Dtype compute_dtype;

    array take(const std::string& key, const mx::Shape& shape, mx::Dtype target) {
        auto entry = tensors.extract(key);
        if (entry.empty()) throw std::runtime_error("Missing MLX SS tensor: " + key);
        const auto& value = entry.mapped();
        if (value.shape() != shape || value.dtype() != mx::bfloat16)
            throw std::runtime_error("Unexpected shape or dtype for MLX SS tensor: " + key);
        auto result = mx::astype(value, target, stream);
        prepared.push_back(result);
        return result;
    }

    Linear linear(const std::string& name, int in, int out, bool torso = true) {
        const auto target = torso ? compute_dtype : mx::float32;
        auto matrix = take(name + ".weight", {out, in}, target);
        return {mx::transpose(matrix, stream), take(name + ".bias", {out}, target)};
    }

    Block block(int index) {
        const auto prefix = "blocks." + std::to_string(index) + ".";
        auto f32 = [&](const char* name, const mx::Shape& shape) {
            return take(prefix + name, shape, mx::float32);
        };
        return {
            f32("modulation", {6 * width}), f32("norm2.weight", {width}), f32("norm2.bias", {width}),
            linear(prefix + "self_attn.to_qkv", width, 3 * width),
            linear(prefix + "self_attn.to_out", width, width),
            linear(prefix + "cross_attn.to_q", width, width),
            linear(prefix + "cross_attn.to_kv", ctx_channels, 2 * width),
            linear(prefix + "cross_attn.to_out", width, width),
            linear(prefix + "mlp.mlp.0", width, ff_width),
            linear(prefix + "mlp.mlp.2", ff_width, width),
            f32("self_attn.q_rms_norm.gamma", {heads, head_width}),
            f32("self_attn.k_rms_norm.gamma", {heads, head_width}),
            f32("cross_attn.q_rms_norm.gamma", {heads, head_width}),
            f32("cross_attn.k_rms_norm.gamma", {heads, head_width})
        };
    }

    void evaluate() {
        mx::eval(prepared);
        prepared.clear();
    }
};

mx::Dtype parse_precision(const std::string& name) {
    if (name == "bf16") return mx::bfloat16;
    if (name == "fp16") return mx::float16;
    if (name == "fp32") return mx::float32;
    throw std::invalid_argument("MLX precision must be bf16, fp16 or fp32");
}
}

struct MlxFlowRunner::Impl {
    std::unique_lock<std::mutex> lease{mlx_mutex};
    mx::Stream stream = inference_stream();
    const mx::Dtype dtype;
    const size_t previous_cache_limit;
    std::optional<Linear> input, time_in, time_out, time_mod, output;
    std::vector<Block> blocks;
    std::optional<array> rope_cos, rope_sin;

    explicit Impl(mx::Dtype precision)
        : dtype(precision), previous_cache_limit(mx::set_cache_limit(512ull << 20)) {
        mx::reset_peak_memory();
    }

    ~Impl() {
        try { mx::synchronize(stream); mx::synchronize(load_stream()); }
        catch (const std::exception& error) { fprintf(stderr, "[mlx] drain: %s\n", error.what()); }
        fprintf(stderr, "[mlx] active=%zu cache=%zu peak=%zu bytes before SS release\n",
                mx::get_active_memory(), mx::get_cache_memory(), mx::get_peak_memory());
        blocks.clear();
        input.reset(); time_in.reset(); time_out.reset(); time_mod.reset(); output.reset();
        rope_cos.reset(); rope_sin.reset();
        mx::clear_cache();
        mx::set_cache_limit(previous_cache_limit);
        fprintf(stderr, "[mlx] active=%zu cache=%zu bytes after SS release\n",
                mx::get_active_memory(), mx::get_cache_memory());
    }

    void load(const std::string& path) {
        WeightReader weights{mx::load_safetensors(path, load_stream()).first, {}, stream, dtype};
        input = weights.linear("input_layer", channels, width, false);
        time_in = weights.linear("t_embedder.mlp.0", 256, width, false);
        time_out = weights.linear("t_embedder.mlp.2", width, width, false);
        time_mod = weights.linear("adaLN_modulation.1", width, 6 * width, false);
        output = weights.linear("out_layer", width, channels, false);
        weights.evaluate();
        for (int i = 0; i < block_count; ++i) {
            blocks.push_back(weights.block(i));
            // Bound the temporary BF16 source allocations during FP16 conversion.
            weights.evaluate();
        }
        if (!weights.tensors.empty())
            throw std::runtime_error("Unexpected MLX SS tensor: " + weights.tensors.begin()->first);
        mx::clear_cache();

        // The original 3D RoPE packs 21 complex frequencies per axis, then one
        // identity phase. Tokens are x/y/z meshgrid order with z contiguous.
        std::vector<float> cosine(tokens * head_width / 2), sine(cosine.size());
        constexpr int freqs = head_width / 2 / 3;
        for (int token = 0; token < tokens; ++token) {
            const int xyz[] = {token / 256, token / 16 % 16, token % 16};
            for (int j = 0; j < head_width / 2; ++j) {
                const float phase = j < 3 * freqs ?
                    xyz[j / freqs] / std::pow(10000.0f, float(j % freqs) / freqs) : 0;
                cosine[token * head_width / 2 + j] = std::cos(phase);
                sine[token * head_width / 2 + j] = std::sin(phase);
            }
        }
        const mx::Shape shape{1, tokens, 1, head_width / 2};
        rope_cos.emplace(cosine.data(), shape);
        rope_sin.emplace(sine.data(), shape);
    }

    array linear(const array& x, const Linear& p) const {
        return mx::addmm(p.bias, x, p.weights, 1, 1, stream);
    }
    array silu(const array& x) const {
        return mx::multiply(x, mx::sigmoid(x, stream), stream);
    }
    array gelu(const array& x) const {
        auto f = mx::astype(x, mx::float32, stream);
        auto cubic = mx::multiply(mx::square(f, stream), f, stream);
        auto inner = mx::add(f, mx::multiply(array(0.044715f), cubic, stream), stream);
        auto cdf = mx::tanh(mx::multiply(array(0.7978845608f), inner, stream), stream);
        auto result = mx::multiply(mx::multiply(f, array(0.5f), stream),
                                   mx::add(array(1.f), cdf, stream), stream);
        return mx::astype(result, x.dtype(), stream);
    }
    array norm(const array& x, const std::optional<array>& weight = {},
               const std::optional<array>& bias = {}, float eps = 1e-6f) const {
        return mx::astype(mx::fast::layer_norm(mx::astype(x, mx::float32, stream), weight, bias, eps, stream),
                          x.dtype(), stream);
    }
    array qk_norm(const array& x, const array& gamma) const {
        auto f = mx::astype(x, mx::float32, stream);
        auto energy = mx::sum(mx::square(f, stream), -1, true, stream);
        auto scale = mx::rsqrt(mx::maximum(energy, array(1e-24f), stream), stream);
        auto unit = mx::multiply(f, scale, stream);
        auto weighted = mx::multiply(unit, gamma, stream);
        return mx::astype(mx::multiply(weighted, array(11.313708499f), stream), dtype, stream);
    }
    array rope(const array& x) const {
        auto pairs = mx::reshape(mx::astype(x, mx::float32, stream), {1,tokens,heads,head_width/2,2}, stream);
        auto xy = mx::unstack(pairs, -1, stream);
        auto even = mx::subtract(mx::multiply(xy[0], *rope_cos, stream),
                                 mx::multiply(xy[1], *rope_sin, stream), stream);
        auto odd = mx::add(mx::multiply(xy[0], *rope_sin, stream),
                           mx::multiply(xy[1], *rope_cos, stream), stream);
        auto packed = mx::reshape(mx::stack({even, odd}, -1, stream), x.shape(), stream);
        return mx::astype(packed, dtype, stream);
    }
    array attention(const array& q, const array& k, const array& v) const {
        auto head_major = [&](const array& a) { return mx::transpose(a, {0,2,1,3}, stream); };
        auto result = mx::fast::scaled_dot_product_attention(head_major(q), head_major(k), head_major(v),
                                                            attention_scale, "", {}, {}, true, stream);
        return mx::reshape(head_major(result), {1, tokens, width}, stream);
    }
    array self_attention(const array& x, const Block& b) const {
        auto qkv = mx::unstack(mx::reshape(linear(x, b.self_qkv), {1,tokens,3,heads,head_width}, stream), 2, stream);
        return linear(attention(rope(qk_norm(qkv[0], b.self_q_norm)), rope(qk_norm(qkv[1], b.self_k_norm)),
                                qkv[2]), b.self_out);
    }
    array cross_attention(const array& x, const array& cond, const Block& b) const {
        auto q = mx::reshape(linear(x, b.cross_q), {1,tokens,heads,head_width}, stream);
        auto kv = mx::unstack(mx::reshape(linear(cond, b.cross_kv), {1,ctx_tokens,2,heads,head_width}, stream), 2, stream);
        return linear(attention(qk_norm(q, b.cross_q_norm), qk_norm(kv[0], b.cross_k_norm), kv[1]), b.cross_out);
    }
    array modulation(float timestep) const {
        std::array<float, 256> embedding{};
        for (int i = 0; i < 128; ++i) {
            const float phase = timestep * std::exp(-std::log(10000.0f) * i / 128.f);
            embedding[i] = std::cos(phase);
            embedding[i+128] = std::sin(phase);
        }
        auto t = linear(silu(linear(array(embedding.data(), {1,256}), *time_in)), *time_out);
        return mx::astype(linear(silu(t), *time_mod), dtype, stream);
    }
    array block(const array& x, const array& cond, const array& mod, const Block& b,
                std::vector<array>* probe = nullptr) const {
        auto controls = mx::astype(mx::add(mod, b.modulation, stream), dtype, stream);
        auto m = mx::split(controls, 6, -1, stream);
        auto modulate = [&](const array& h, int i) {
            return mx::add(mx::multiply(norm(h), mx::add(array(1.f, dtype), m[i+1], stream), stream),
                           m[i], stream);
        };
        auto self = self_attention(modulate(x, 0), b);
        auto after_self = mx::add(x, mx::multiply(self, m[2], stream), stream);
        auto cross = cross_attention(norm(after_self, b.norm_weight, b.norm_bias), cond, b);
        auto after_cross = mx::add(after_self, cross, stream);
        auto ff = linear(gelu(linear(modulate(after_cross, 3), b.ff_in)), b.ff_out);
        auto result = mx::add(after_cross, mx::multiply(ff, m[5], stream), stream);
        if (probe) *probe = {self, cross, ff, result};
        return result;
    }

    std::pair<array,array> inputs(const std::vector<float>& latent, float timestep, const FlowCond& cond) const {
        if (latent.size() != tokens * channels || !cond.cond || cond.proj || !std::isfinite(timestep))
            throw std::invalid_argument("Invalid MLX SS forward input");
        auto h = linear(array(latent.data(), {1,tokens,channels}), *input);
        return {mx::astype(h, dtype, stream),
                mx::astype(array(cond.cond, {1,ctx_tokens,ctx_channels}), dtype, stream)};
    }

    std::vector<float> read(const array& value) const {
        auto flat = mx::contiguous(mx::astype(value, mx::float32, stream), false, stream);
        flat.eval();
        mx::synchronize(stream);
        const auto* ptr = flat.data<float>();
        if (!std::all_of(ptr, ptr + flat.size(), [](float f) { return std::isfinite(f); }))
            throw std::runtime_error("Nonfinite MLX SS prediction");
        return {ptr, ptr + flat.size()};
    }
};

MlxFlowRunner::MlxFlowRunner(const std::string& weights, const std::string& precision)
    : impl_(std::make_unique<Impl>(parse_precision(precision))) {
    impl_->load(weights);
}
MlxFlowRunner::~MlxFlowRunner() = default;

std::vector<float> MlxFlowRunner::forward(const std::vector<float>& latent, float timestep, const FlowCond& cond) {
    auto& p = *impl_;
    auto [hidden, context] = p.inputs(latent, timestep, cond);
    auto mod = p.modulation(timestep);
    // Submit each block independently to bound live intermediate activations.
    for (const auto& block : p.blocks) {
        hidden = p.block(hidden, context, mod, block);
        hidden.eval();
    }
    return p.read(p.linear(p.norm(mx::astype(hidden, mx::float32, p.stream), {}, {}, 1e-5f), *p.output));
}

void MlxFlowRunner::probe_block(const std::vector<float>& latent, float timestep, const FlowCond& cond,
                                const std::string& output_dir) {
    auto& p = *impl_;
    auto [hidden, context] = p.inputs(latent, timestep, cond);
    std::vector<array> values;
    p.block(hidden, context, p.modulation(timestep), p.blocks.front(), &values);
    mx::eval(values);
    const char* names[] = {"self", "cross", "ff", "block"};
    for (size_t i = 0; i < values.size(); ++i) {
        auto data = p.read(values[i]);
        npy::save(output_dir + "/" + names[i] + ".npy", data.data(), {1, tokens, width});
    }
}

const char* MlxFlowRunner::precision() const {
    if (impl_->dtype == mx::float16) return "fp16 torso, fp32 IO/norm/RoPE";
    if (impl_->dtype == mx::float32) return "fp32 arrays";
    return "bf16 torso, fp32 IO/norm/RoPE";
}
}
