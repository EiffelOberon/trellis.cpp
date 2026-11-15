#pragma once

#include "flow_runner.h"
#include <memory>
#include <string>

namespace trellis {

// Dense SS only. The sampler boundary remains host FP32, token major.
class MlxFlowRunner {
public:
    explicit MlxFlowRunner(const std::string& weights, const std::string& precision = "fp16");
    ~MlxFlowRunner();
    std::vector<float> forward(const std::vector<float>& latent, float timestep, const FlowCond& cond);
    // Runs block zero with the production input projection and modulation.
    void probe_block(const std::vector<float>& latent, float timestep, const FlowCond& cond,
                     const std::string& output_dir);
    const char* precision() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
