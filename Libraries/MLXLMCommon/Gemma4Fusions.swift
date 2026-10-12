import MLX

private let compiledGemma4Softcap = compile(shapeless: true) {
    (logits: MLXArray, cap: MLXArray) in tanh(logits / cap) * cap
}

package func gemma4LogitSoftcap(_ logits: MLXArray, _ cap: Float) -> MLXArray {
    guard Device.defaultDevice().deviceType == .gpu else { return tanh(logits / cap) * cap }
    // Swift scalars use the logits' dtype; a float32 tensor would promote low-precision logits.
    return compiledGemma4Softcap(logits, cap.asMLXArray(dtype: logits.dtype))
}

package func gemma4LogitSoftcap(_ logits: MLXArray, _ cap: MLXArray) -> MLXArray {
    guard Device.defaultDevice().deviceType == .gpu, cap.dtype == .float32 else {
        return tanh(logits / cap) * cap
    }
    // Promote before tracing: mixed float16/float32 division can round differently inside compile.
    if logits.dtype == .float16 || logits.dtype == .bfloat16 {
        return compiledGemma4Softcap(logits.asType(.float32), cap)
    }
    return compiledGemma4Softcap(logits, cap)
}
