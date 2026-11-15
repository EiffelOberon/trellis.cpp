// Real-checkpoint regression: padded attention, changing timestep/conditioning,
// repeated uploads, and several block group sizes must preserve resident outputs.
#include "flow_runner.h"
#include "trellis_model.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: trellis-test-streamed-flow MODEL_DIRECTORY\n"); return 2; }
    try {
        constexpr int N = 257, Lc = 1029;
        std::vector<std::array<int,3>> coords(N);
        for (int i = 0; i < N; ++i) coords[i] = {i / 64, (i / 8) % 8, i % 8};
        std::vector<float> cond[2];
        cond[0].resize(Lc * 1024); cond[1].resize(Lc * 1024);
        for (size_t i = 0; i < cond[0].size(); ++i) cond[0][i] = 0.3f * std::sin(float(i) * 0.013f);
        const float times[] = {1000, 500, 10, 500};
        for (int channels : {32, 64}) {
            const std::string path = std::string(argv[1]) + (channels == 32 ? "/shape_flow_512.gguf" : "/tex_flow_512.gguf");
            trellis::DiTParams p; p.in_ch = channels; p.out_ch = 32;
            std::vector<float> x(channels * N);
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * 0.07f);
            std::vector<std::vector<float>> reference;
            {
                auto m = trellis::Model::load(path);
                std::unique_ptr<trellis::DitRunner> run(trellis::make_sparse_runner(m, p, coords, Lc));
                for (int j = 0; j < 4; ++j) reference.push_back(run->forward(x, times[j], cond[j % 2].data()));
            }
            for (int group : {1, 5, 30}) {
                auto m = trellis::Model::load(path, 0, true);
                std::unique_ptr<trellis::DitRunner> run(trellis::make_sparse_runner(m, p, coords, Lc, group));
                for (int j = 0; j < 4; ++j) {
                    auto out = run->forward(x, times[j], cond[j % 2].data());
                    double error = 0, scale = 0, maxabs = 0;
                    for (size_t k = 0; k < out.size(); ++k) {
                        if (!std::isfinite(out[k])) throw std::runtime_error("non-finite streamed output");
                        const double d = double(out[k]) - reference[j][k];
                        error += d * d; scale += double(reference[j][k]) * reference[j][k];
                        maxabs = std::max(maxabs, std::abs(d));
                    }
                    const double rel = std::sqrt(error / std::max(scale, 1e-30));
                    printf("channels=%d group=%d pass=%d relL2=%.9g maxabs=%.9g\n", channels, group, j, rel, maxabs);
                    if (rel > 1e-5 || maxabs > 1e-4) throw std::runtime_error("streamed output differs from resident output");
                }
            }
        }
        puts("PASS: resident/streamed shape and texture parity");
    } catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
