// Compare the unchanged whole-block decoder graph with bounded row chunks.
#include "trellis_model.h"
#include "trellis_sched.h"
#include "sparse.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

static void budget(const char* mb) {
#ifdef _WIN32
    _putenv_s("TRELLIS_BLOCK_CHUNK_MB", mb);
#else
    setenv("TRELLIS_BLOCK_CHUNK_MB", mb, 1);
#endif
}
int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: trellis-test-sparse-chunk SHAPE_DEC.gguf\n"); return 2; }
    try {
        constexpr int N = 65667, C = 128;
        auto m = trellis::Model::load(argv[1]);
        std::vector<std::array<int,3>> coords(N);
        for (int i = 0; i < N; ++i) coords[i] = {i / (41 * 41), (i / 41) % 41, i % 41};
        const auto neighbors = trellis::build_neighbor_table(coords);
        std::vector<float> input(C * N);
        for (size_t i = 0; i < input.size(); ++i) input[i] = 2 * std::sin(float(i) * 0.073f);
        auto run = [&](int count, const char* mb) {
            budget(mb);
            auto* ctx = ggml_init({ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false) + (1<<20), nullptr, true});
            try {
                trellis::GraphExec exec(m);
                auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, N); ggml_set_input(x);
                auto* nbr = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N, 27); ggml_set_input(nbr);
                auto* h = x;
                for (int i = 0; i < count; ++i)
                    h = trellis::sparse_convnext(ctx, m, "blocks.3." + std::to_string(i), h, nbr, N);
                ggml_set_output(h);
                auto* g = ggml_new_graph_custom(ctx, 65536, false); ggml_build_forward_expand(g, h);
                if (!exec.alloc(g)) throw std::runtime_error("alloc failed");
                ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                ggml_backend_tensor_set(nbr, neighbors.data(), 0, neighbors.size() * sizeof(int32_t));
                if (exec.compute(g, "sparse.chunk.parity") != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
                auto result = trellis::tensor_to_f32(h);
                ggml_free(ctx);
                return result;
            } catch (...) { ggml_free(ctx); throw; }
        };
        for (int count : {1,4}) {
            const auto ref = run(count, "1500");
            const auto out = run(count, "128"); // one full tile and a short tail
            double diff = 0, scale = 0, maxabs = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = double(out[i]) - ref[i];
                diff += d * d; scale += double(ref[i]) * ref[i];
                maxabs = std::max(maxabs, std::abs(d));
                if (!std::isfinite(d)) throw std::runtime_error("non-finite decoder output");
            }
            const double rel = std::sqrt(diff / std::max(scale, 1e-30));
            printf("blocks=%d relL2=%.9g maxabs=%.9g\n", count, rel, maxabs);
            if (rel > 1e-5) throw std::runtime_error("decoder chunk parity failed");
        }
        puts("PASS: whole/chunked sparse decoder parity");
    } catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
