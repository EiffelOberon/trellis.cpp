#pragma once
#include "trellis_args.h"

// Run the full TRELLIS.2 image->3D pipeline described by `p`: reconstruct
// `p.image` into a textured GLB at `p.output`. Returns 0 on success. All
// behavior (resolution, bg removal, texture, guidance, ...) comes from `p`,
// which trellis-cli / trellis-server populate via trellis::parse_args.
namespace trellis {
struct RunControl;
// Resolve and validate explicit device / SS selections before loading model weights.
void resolve_execution(TrellisParams& params);
}
int trellis_run(const trellis::TrellisParams& p, trellis::RunControl* control = nullptr);
