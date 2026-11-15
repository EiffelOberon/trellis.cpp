#pragma once
#include "flow_runner.h"

namespace trellis {
// FP16/BF16 bytes are copied unchanged. Only one block group resides on the GPU.
class StreamedDit {
public:
    StreamedDit(const Model& model, const DiTParams& params, int tokens, int cond_tokens,
                const std::vector<float>& cos, const std::vector<float>& sin, int blocks);
    ~StreamedDit();
    std::vector<float> forward(const std::vector<float>& x, const std::vector<float>& timestep,
                               const FlowCond& cond);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
