#include "trt_flow.h"
#include "npy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: trellis-test-trt ENGINE FIXTURE_DIR [L2_RTOL=0.02] [ATOL=0.02] [MAX_RTOL=0.05]\n");
        return 2;
    }
    try {
        trellis::TrtFlowRunner runner(argv[1], 0, true);
        const std::string dir = argv[2];
        auto latent = npy::load(dir + "/latent.npy");
        auto time = npy::load(dir + "/timestep.npy");
        auto cond = npy::load(dir + "/cond.npy");
        auto expected = npy::load(dir + "/velocity.npy");
        if (cond.data.size() != 1029 * 1024 || time.data.size() != 1) throw std::runtime_error("Bad fixture dimensions");
        auto actual = runner.forward(latent.data, time.data[0], cond.data.data());
        if (actual.size() != expected.data.size()) throw std::runtime_error("Output size mismatch");
        const double rtol = argc > 3 ? std::stod(argv[3]) : .02, atol = argc > 4 ? std::stod(argv[4]) : .02;
        const double max_rtol = argc > 5 ? std::stod(argv[5]) : .05;
        double maxabs = 0, sum = 0, ref = 0, refmax = 0; size_t bad = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double delta = std::abs(double(actual[i]) - expected.data[i]);
            if (!std::isfinite(expected.data[i]) || delta > atol + rtol * std::abs(expected.data[i])) ++bad;
            maxabs = std::max(maxabs, delta); sum += delta * delta; ref += double(expected.data[i]) * expected.data[i];
            refmax = std::max(refmax, std::abs(double(expected.data[i])));
        }
        printf("max_abs=%.8g relative_l2=%.8g out_of_tolerance=%zu/%zu\n", maxabs, std::sqrt(sum / std::max(ref, 1e-24)), bad, actual.size());
        npy::save(dir + "/trt_velocity.npy", actual.data(), {1,4096,8});
        const double relative_l2 = std::sqrt(sum / std::max(ref, 1e-24));
        return relative_l2 <= rtol && maxabs <= atol + max_rtol * refmax ? 0 : 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
