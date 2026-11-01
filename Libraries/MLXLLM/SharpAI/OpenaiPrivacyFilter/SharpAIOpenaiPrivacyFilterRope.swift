// SharpAI fork overlay: openai_privacy_filter YaRN rotary frequencies. Not part of upstream mlx-swift-lm.
//
// Follows transformers _compute_yarn_parameters, including truncate: false. MLXLMCommon's
// YarnRoPE always rounds the correction range, so it is not reused here.
// The layout is interleaved pairs, so callers pass traditional: true to MLXFast.RoPE.

import Foundation
import MLX

enum SharpAIOpenaiPrivacyFilterRope {

    /// Wavelengths for the `freqs` argument of MLXFast.RoPE, and the YaRN attention factor.
    static func yarn(_ config: SharpAIOpenaiPrivacyFilterConfiguration) -> (
        freqs: MLXArray, attentionFactor: Float
    ) {
        let params = config.rope
        let dim = Double(config.headDimensions)
        let base = Double(params.ropeTheta)
        let factor = Double(params.factor)
        let half = config.headDimensions / 2

        func correctionDim(_ rotations: Double) -> Double {
            dim
                * log(Double(params.originalMaxPositionEmbeddings) / (rotations * 2 * Double.pi))
                / (2 * log(base))
        }

        var low = correctionDim(Double(params.betaFast))
        var high = correctionDim(Double(params.betaSlow))
        if params.truncate {
            low = floor(low)
            high = ceil(high)
        }
        low = max(low, 0)
        high = min(high, dim - 1)
        if low == high {
            high += 0.001
        }

        var wavelengths: [Float] = []
        wavelengths.reserveCapacity(half)
        for i in 0 ..< half {
            let posFreq = pow(base, Double(2 * i) / dim)
            let extrapolation = 1.0 / posFreq
            let interpolation = 1.0 / (factor * posFreq)
            let ramp = min(max((Double(i) - low) / (high - low), 0), 1)
            let invFreq = interpolation * ramp + extrapolation * (1 - ramp)
            wavelengths.append(Float(1.0 / invFreq))
        }

        let attentionFactor = factor > 1 ? 0.1 * log(factor) + 1 : 1
        return (MLXArray(wavelengths), Float(attentionFactor))
    }
}
