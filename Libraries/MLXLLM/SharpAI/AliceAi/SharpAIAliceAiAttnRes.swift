// SharpAI fork overlay: alice_ai attention residual mixing. Not part of upstream mlx-swift-lm.
//
// Same maths as the fallback path of vLLM's kimi_k3 attn_res kernel (used by alice_ai.py):
// logit_s = dot(v_s, norm * qk) * rsqrt(mean(v_s^2) + eps), softmax over sources, weighted sum of raw v_s.

import Foundation
import MLX

/// Mixes residual sources (completed blocks plus the open partial) into one vector per position.
/// Sources are shaped `[..., D]`; the result has the same shape as one source.
public func sharpAIAliceAiAttnRes(
    sources: [MLXArray], normWeight: MLXArray, qkWeight: MLXArray, eps: Float
) -> MLXArray {
    precondition(!sources.isEmpty, "AliceAI AttnRes needs at least one source")
    if sources.count == 1 { return sources[0] }

    let dtype = sources[0].dtype
    let values = MLX.stacked(sources, axis: -2).asType(.float32)  // [..., S, D]
    let weight = (normWeight * qkWeight).asType(.float32)  // [D]
    let rstd = MLX.rsqrt(MLX.mean(MLX.square(values), axis: -1) + MLXArray(eps))  // [..., S]
    let logits = MLX.sum(values * weight, axis: -1) * rstd  // [..., S]
    let probs = MLX.softmax(logits, axis: -1, precise: true)  // [..., S]
    let mixed = MLX.sum(MLX.expandedDimensions(probs, axis: -1) * values, axis: -2)
    return mixed.asType(dtype)
}
