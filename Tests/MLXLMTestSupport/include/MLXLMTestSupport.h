#pragma once

/// Returns 1 once the loader has set MLX_ENABLE_TF32=0 for this process.
int mlxlm_float32_matmul_pinned(void);
