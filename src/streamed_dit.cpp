#include "streamed_dit.h"
#include "trellis_model.h"
#include "trellis_sched.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace trellis {
namespace {
struct ContextFree { void operator()(ggml_context* c) const { if (c) ggml_free(c); } };
struct BufferFree { void operator()(ggml_backend_buffer* b) const { if (b) ggml_backend_buffer_free(b); } };
using Context = std::unique_ptr<ggml_context, ContextFree>;
using Buffer = std::unique_ptr<ggml_backend_buffer, BufferFree>;
Context context(size_t nodes) {
    Context c(ggml_init({ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20), nullptr, true}));
    if (!c) throw std::runtime_error("streamed DiT: metadata allocation failed");
    return c;
}
struct Graph {
    Context ctx;
    std::unique_ptr<GraphExec> exec;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* output = nullptr;
    ggml_tensor* mod = nullptr;
    void alloc(const Model& m, size_t nodes) {
        graph = ggml_new_graph_custom(ctx.get(), nodes, false);
        ggml_build_forward_expand(graph, output);
        ggml_set_output(output);
        if (mod) { ggml_build_forward_expand(graph, mod); ggml_set_output(mod); }
        exec = std::make_unique<GraphExec>(m);
        if (!exec->alloc(graph)) throw std::runtime_error("streamed DiT: graph allocation failed");
    }
    void compute() {
        if (exec->compute(graph, "dit.streamed") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("streamed DiT: compute failed");
    }
};
}

struct StreamedDit::Impl {
    // Declare buffers before their tensor metadata/graphs so destruction drains graphs first.
    Buffer weights, state;
    Model aliases;
    Context state_ctx;
    Graph input, blocks, output;
    struct Packed { Buffer buffer; size_t size; void* data; };
    std::vector<Packed> packed;
    ggml_tensor *raw = nullptr, *x = nullptr, *tf = nullptr, *h = nullptr, *mod = nullptr,
                *cond = nullptr, *proj = nullptr, *cos = nullptr, *sin = nullptr;
    int group_count = 0;

    Impl(const Model& m, const DiTParams& p, int N, int Lc,
         const std::vector<float>& rcos, const std::vector<float>& rsin, int count) {
        if (!m.host_weights || m.sched || count < 1 || count > p.n_blocks || p.n_blocks % count)
            throw std::invalid_argument("streamed DiT requires host weights, direct CUDA, and a block count dividing 30");
        group_count = p.n_blocks / count;
        aliases.backend = m.backend; aliases.on_gpu = true;
        aliases.meta = ggml_init({ggml_tensor_overhead() * (m.tensors.size() + 8), nullptr, true});
        if (!aliases.meta) throw std::runtime_error("streamed DiT: weight metadata allocation failed");
        const auto buft = ggml_backend_get_default_buffer_type(m.backend);
        const size_t align = ggml_backend_buft_get_alignment(buft);
        struct Entry { std::string name; ggml_tensor* tensor; size_t offset; };
        std::vector<Entry> entries[3];
        size_t sizes[3] = {};
        for (const auto& [name, src] : m.tensors) {
            int phase = 0;
            if (name.rfind("blocks.", 0) == 0) {
                const int index = std::stoi(name.substr(7));
                if (index >= count) continue;
                phase = 1;
            } else if (name.rfind("out_layer.", 0) == 0) phase = 2;
            auto* dst = ggml_dup_tensor(aliases.meta, src);
            ggml_set_name(dst, name.c_str());
            aliases.tensors[name] = dst;
            entries[phase].push_back({name, dst, sizes[phase]});
            sizes[phase] += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, dst), align);
        }
        const size_t capacity = std::max({sizes[0], sizes[1], sizes[2]});
        weights.reset(ggml_backend_buft_alloc_buffer(buft, capacity));
        if (!weights) throw std::runtime_error("streamed DiT: weight buffer allocation failed");
        ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto& phase : entries) for (auto& e : phase) {
            if (ggml_backend_tensor_alloc(weights.get(), e.tensor,
                    static_cast<uint8_t*>(ggml_backend_buffer_get_base(weights.get())) + e.offset) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("streamed DiT: weight tensor allocation failed");
        }
        // Pinned host storage permits full PCIe bandwidth. Fall back to ordinary host storage
        // only on backends without a host buffer type (the mode currently requires CUDA).
        auto host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(m.backend));
        if (!host_buft) host_buft = ggml_backend_cpu_buffer_type();
        // Pack each group once; one transfer replaces dozens of tiny copies.
        for (int part = 0; part < group_count + 2; ++part) {
            const int phase = part == 0 ? 0 : part == group_count + 1 ? 2 : 1;
            Buffer buffer(ggml_backend_buft_alloc_buffer(host_buft, sizes[phase]));
            if (!buffer) throw std::runtime_error("streamed DiT: host weight allocation failed");
            void* data = ggml_backend_buffer_get_base(buffer.get());
            std::memset(data, 0, sizes[phase]);
            packed.push_back({std::move(buffer), sizes[phase], data});
            for (const auto& e : entries[phase]) {
                std::string source = e.name;
                if (phase == 1) {
                    const size_t dot = source.find('.', 7);
                    source = "blocks." + std::to_string(std::stoi(source.substr(7, dot - 7)) + (part - 1) * count) + source.substr(dot);
                }
                auto* src = m.get(source);
                if (src->type != e.tensor->type || !ggml_are_same_shape(src, e.tensor))
                    throw std::runtime_error("streamed DiT: unequal block weight layout: " + source);
                ggml_backend_tensor_get(src, static_cast<uint8_t*>(data) + e.offset, 0, ggml_nbytes(src));
            }
        }
        raw = ggml_new_tensor_1d(aliases.meta, GGML_TYPE_I8, capacity);
        if (ggml_backend_tensor_alloc(weights.get(), raw, ggml_backend_buffer_get_base(weights.get())) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("streamed DiT: raw upload view allocation failed");

        // Persistent device state prevents any hidden-state round trips through the CPU.
        state_ctx = context(64);
        auto tensor = [&](int d, int n) {
            auto* t = ggml_new_tensor_2d(state_ctx.get(), GGML_TYPE_F32, d, n);
            ggml_set_input(t); return t;
        };
        x = tensor(p.in_ch, N); tf = tensor(256, 1); h = tensor(p.d_model, N);
        mod = tensor(6 * p.d_model, 1); cond = tensor(p.d_cond, Lc);
        if (p.proj_mode) proj = tensor(p.proj_ch, N);
        cos = ggml_new_tensor_4d(state_ctx.get(), GGML_TYPE_F32, 1, p.head_dim / 2, 1, N); ggml_set_input(cos);
        sin = ggml_dup_tensor(state_ctx.get(), cos); ggml_set_input(sin);
        state.reset(ggml_backend_alloc_ctx_tensors(state_ctx.get(), m.backend));
        if (!state) throw std::runtime_error("streamed DiT: state allocation failed");
        ggml_backend_tensor_set(cos, rcos.data(), 0, rcos.size() * sizeof(float));
        ggml_backend_tensor_set(sin, rsin.data(), 0, rsin.size() * sizeof(float));

        input.ctx = context(256);
        input.output = build_dit_input(input.ctx.get(), aliases, p, x, tf, &input.mod);
        input.alloc(aliases, 256);
        // The reusable block graph binds blocks 0..count-1; later groups supply identical layouts.
        blocks.ctx = context(65536);
        blocks.output = build_dit_blocks(blocks.ctx.get(), aliases, p, 0, count, h, mod, cond, proj, cos, sin);
        blocks.alloc(aliases, 65536);
        output.ctx = context(128);
        output.output = build_dit_output(output.ctx.get(), aliases, p, h);
        output.alloc(aliases, 128);
        fprintf(stderr, "[dit-stream] %d blocks/group; weights %.1f MiB, state+compute %.1f MiB; unchanged weight precision\n",
                count, capacity / 1048576.0, (ggml_backend_buffer_get_size(state.get()) + input.exec->buffer_size() +
                blocks.exec->buffer_size() + output.exec->buffer_size()) / 1048576.0);
    }
    std::vector<float> forward(const std::vector<float>& xt, const std::vector<float>& timestep, const FlowCond& c) {
        if (xt.size() * sizeof(float) != ggml_nbytes(x) || timestep.size() != 256 || !c.cond)
            throw std::invalid_argument("streamed DiT: invalid forward inputs");
        ggml_backend_tensor_set(x, xt.data(), 0, xt.size() * sizeof(float));
        ggml_backend_tensor_set(tf, timestep.data(), 0, timestep.size() * sizeof(float));
        ggml_backend_tensor_set(cond, c.cond, 0, ggml_nbytes(cond));
        if (proj) {
            if (c.proj) ggml_backend_tensor_set(proj, c.proj, 0, ggml_nbytes(proj));
            else ggml_backend_tensor_memset(proj, 0, 0, ggml_nbytes(proj));
        }
        for (int part = 0; part < group_count + 2; ++part) {
            ggml_backend_tensor_set(raw, packed[part].data, 0, packed[part].size);
            Graph& g = part == 0 ? input : part == group_count + 1 ? output : blocks;
            g.compute();
            if (part == 0) ggml_backend_tensor_copy(input.mod, mod);
            if (part <= group_count) ggml_backend_tensor_copy(g.output, h);
        }
        return tensor_to_f32(output.output);
    }
};

StreamedDit::StreamedDit(const Model& m, const DiTParams& p, int N, int Lc,
                         const std::vector<float>& cos, const std::vector<float>& sin, int blocks)
    : impl_(std::make_unique<Impl>(m, p, N, Lc, cos, sin, blocks)) {}
StreamedDit::~StreamedDit() = default;
std::vector<float> StreamedDit::forward(const std::vector<float>& x, const std::vector<float>& t, const FlowCond& c) {
    return impl_->forward(x, t, c);
}
}
