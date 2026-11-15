#pragma once
#include "flow_runner.h"
#include <memory>
#include <string>

namespace trellis {
// Fixed 512-preset SS flow. The host-vector ABI deliberately matches FlowFwd;
// transfers and synchronization are included in forward timing.
class TrtFlowRunner {
public:
    explicit TrtFlowRunner(const std::string& engine_path, int device = 0, bool allow_probe = false,
                           int weight_budget_mib = -1, bool require_resident_engine = false);
    ~TrtFlowRunner();
    TrtFlowRunner(const TrtFlowRunner&) = delete;
    TrtFlowRunner& operator=(const TrtFlowRunner&) = delete;
    std::vector<float> forward(const std::vector<float>& latent, float timestep, const FlowCond& cond);
    std::string precision() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
