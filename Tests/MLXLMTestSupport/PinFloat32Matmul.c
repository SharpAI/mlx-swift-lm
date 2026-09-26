#include "MLXLMTestSupport.h"

#include <stdlib.h>

static int pinned = 0;

// M5 GPUs run float32 matmuls as TF32 unless MLX_ENABLE_TF32=0. MLX reads it
// once, so set it at load time, before any test runs, as MLX's own tests do.
__attribute__((constructor)) static void pin_float32_matmul(void) {
    pinned = setenv("MLX_ENABLE_TF32", "0", 1) == 0;
}

int mlxlm_float32_matmul_pinned(void) { return pinned; }
