// Decode two captured token-major SS latents with the same native decoder.
#include "trellis_model.h"
#include "ss_decoder.h"
#include "npy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: trellis-compare-ss SS_DEC.gguf A.npy B.npy [MIN_IOU=0.95]\n"); return 2; }
    try {
        auto model = trellis::Model::load(argv[1]);
        auto decode = [&](const char* path) {
            auto latent = npy::load(path);
            if (latent.data.size() != 8*4096) throw std::runtime_error("Bad SS latent dimensions");
            std::vector<float> chw(8*4096);
            for (int c = 0; c < 8; ++c) for (int t = 0; t < 4096; ++t) chw[c*4096+t] = latent.data[t*8+c];
            auto result = trellis::ss_decode(model, chw);
            for (float x : result) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite decoded occupancy");
            return result;
        };
        auto a = decode(argv[2]), b = decode(argv[3]);
        if (a.size() != b.size()) throw std::runtime_error("Occupancy dimensions differ");
        size_t both=0, either=0, agree=0;
        double square=0;
        for (size_t i=0; i<a.size(); ++i) {
            const bool x=a[i]>0, y=b[i]>0;
            both += x&&y; either += x||y; agree += x==y;
            const double d=double(a[i])-b[i]; square+=d*d;
        }
        const double iou=either?double(both)/either:1;
        printf("{\"occupancy_iou\": %.9g, \"agreement\": %.9g, \"logits_rmse\": %.9g}\n",
               iou, double(agree)/a.size(), std::sqrt(square/a.size()));
        return iou >= (argc>4?std::stod(argv[4]):.95) ? 0 : 1;
    } catch (const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return 1; }
}
