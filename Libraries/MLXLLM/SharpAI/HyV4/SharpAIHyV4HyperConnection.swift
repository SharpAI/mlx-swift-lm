//
//  SharpAIHyV4HyperConnection.swift
//  MLXLLM
//
//  Independent hyper-connections (iHC) for hy_v4: the residual is kept as `hc_mult`
//  parallel streams, collapsed to one stream before each block, and expanded back
//  after it. Port of HYV4HyperConnection and HYV4HyperHead in modular_hy_v4.py,
//  which is based on DeepseekV4HyperConnection and DeepseekV4HyperHead.
//
//  Differences from the DeepSeek-V4 form. There is no Sinkhorn `comb` matrix, so the
//  residual is a plain sum of the streams. The post gate is scaled by `hc_magnitude`.
//  The mix projection is fp32.
//
//  Norm: the reference writes `F.linear(flat, fn) * input_norm(flat)`. For that to
//  broadcast, `input_norm` has to be the inverse RMS scalar per token, which is what
//  HYV4UnweightedRMSNorm returns. This port uses that form. Because the scalar is
//  per token, the result equals `F.linear(flat * rsqrt, fn)`.
//
//  UNVERIFIED: checkpoint keys (`attn_hc.*`, `ffn_hc.*`, `hc_head.*`) follow the
//  transformers parameter names. No checkpoint index is on disk.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One block's stream mixing. Returns `post` [B, L, N] for the block output and the
/// collapsed block input [B, L, D].
class SharpAIHyV4HyperConnection: Module {
    let streams: Int
    let magnitude: Float
    let eps: Float
    let normEps: Float

    /// [2N, N*D]. Rows 0..<N give pre logits, rows N..<2N give post logits.
    @ParameterInfo(key: "fn") var fn: MLXArray
    /// [2N]
    @ParameterInfo(key: "base") var base: MLXArray
    /// [2]: the pre scale and the post scale.
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ config: SharpAIHyV4Configuration) {
        let n = config.resolvedHcMult
        self.streams = n
        self.magnitude = config.resolvedHcMagnitude
        self.eps = config.resolvedHcEps
        self.normEps = config.rmsNormEps

        // Initialization follows HYV4PreTrainedModel._init_weights: fn ~ N(0, 0.006),
        // scale 0.01, and the pre bias set to -log(max(N - 1, 1)).
        _fn.wrappedValue = MLXRandom.normal([2 * n, n * config.hiddenSize]) * Float(0.006)
        let preBias = -log(Float(max(n - 1, 1)))
        _base.wrappedValue = concatenated([ones([n]) * preBias, zeros([n])], axis: 0)
        _scale.wrappedValue = ones([2]) * Float(0.01)
    }

    /// `streams` is [B, L, N, D]. Returns (post [B, L, N], collapsed [B, L, D]), both fp32
    /// for post and the input dtype for collapsed.
    func callAsFunction(_ streams: MLXArray) -> (post: MLXArray, collapsed: MLXArray) {
        let (B, L) = (streams.dim(0), streams.dim(1))
        let flat = streams.reshaped(B, L, -1).asType(.float32)
        let invRms = rsqrt(flat.square().mean(axis: -1, keepDims: true) + normEps)
        let mixes = matmul(flat, fn.asType(.float32).T) * invRms  // [B, L, 2N]

        let mixParts = split(mixes, indices: [self.streams], axis: -1)
        let baseParts = split(base.asType(.float32), indices: [self.streams], axis: 0)
        let pre = sigmoid(mixParts[0] * scale[0] + baseParts[0]) + eps
        let post = sigmoid(mixParts[1] * scale[1] + baseParts[1]) * magnitude + eps

        let collapsed = (pre.expandedDimensions(axis: -1) * streams.asType(.float32))
            .sum(axis: 2)
        return (post, collapsed.asType(streams.dtype))
    }
}

/// Final stream collapse before the model norm (HYV4HyperHead).
class SharpAIHyV4HyperHead: Module {
    let eps: Float
    let normEps: Float

    /// [N, N*D]
    @ParameterInfo(key: "hc_fn") var hcFn: MLXArray
    /// [N]
    @ParameterInfo(key: "hc_base") var hcBase: MLXArray
    /// [1]
    @ParameterInfo(key: "hc_scale") var hcScale: MLXArray

    init(_ config: SharpAIHyV4Configuration) {
        let n = config.resolvedHcMult
        self.eps = config.resolvedHcEps
        self.normEps = config.rmsNormEps

        _hcFn.wrappedValue = MLXRandom.normal([n, n * config.hiddenSize]) * Float(0.006)
        let preBias = -log(Float(max(n - 1, 1)))
        _hcBase.wrappedValue = ones([n]) * preBias
        _hcScale.wrappedValue = ones([1]) * Float(0.01)
    }

    /// `x` is [B, L, N, D]. Returns [B, L, D].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let flat = x.reshaped(B, L, -1).asType(.float32)
        let invRms = rsqrt(flat.square().mean(axis: -1, keepDims: true) + normEps)
        let mixes = matmul(flat, hcFn.asType(.float32).T) * invRms  // [B, L, N]
        let pre = sigmoid(mixes * hcScale.asType(.float32) + hcBase.asType(.float32)) + eps
        let out = (pre.expandedDimensions(axis: -1) * x.asType(.float32)).sum(axis: 2)
        return out.asType(x.dtype)
    }
}
