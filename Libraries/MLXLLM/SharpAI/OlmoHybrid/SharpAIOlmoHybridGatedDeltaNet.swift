// SharpAI fork overlay: olmo_hybrid GatedDeltaNet block. Not part of upstream mlx-swift-lm.
//
// Port of OlmoHybridGatedDeltaNet and OlmoHybridRMSNormGated in transformers' modeling_olmo_hybrid.py.
// Separate q/k/v/a/b/g projections, depthwise conv over q|k|v, and beta scaled by 2 when
// linear_allow_neg_eigval is set. The recurrence runs in SharpAIOverlayGatedDeltaKDA, which takes
// explicit beta and a per-key decay, so the scalar decay is broadcast over the key dimension.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Gated RMSNorm: weight * rmsnorm(x) * silu(gate), computed in float32 and cast back.
final class SharpAIOlmoHybridRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let inputDType = x.dtype
        let x32 = x.asType(.float32)
        let variance = mean(square(x32), axis: -1, keepDims: true)
        let normed = (x32 * rsqrt(variance + MLXArray(eps))).asType(inputDType)
        let scaled = weight * normed
        return (scaled.asType(.float32) * silu(gate.asType(.float32))).asType(inputDType)
    }
}

final class SharpAIOlmoHybridGatedDeltaNet: Module {
    let numKHeads: Int
    let numVHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int
    let allowNegEigval: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "a_proj") var aProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "o_norm") var oNorm: SharpAIOlmoHybridRMSNormGated

    init(_ config: SharpAIOlmoHybridConfiguration) {
        numKHeads = config.linearNumKeyHeads
        numVHeads = config.linearNumValueHeads
        headKDim = config.linearKeyHeadDim
        headVDim = config.linearValueHeadDim
        keyDim = headKDim * numKHeads
        valueDim = headVDim * numVHeads
        convKernelSize = config.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        allowNegEigval = config.linearAllowNegEigval

        _qProj.wrappedValue = Linear(config.hiddenSize, keyDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, keyDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, valueDim, bias: false)
        _aProj.wrappedValue = Linear(config.hiddenSize, numVHeads, bias: false)
        _bProj.wrappedValue = Linear(config.hiddenSize, numVHeads, bias: false)
        _gProj.wrappedValue = Linear(config.hiddenSize, valueDim, bias: false)
        _oProj.wrappedValue = Linear(valueDim, config.hiddenSize, bias: false)

        // UNVERIFIED: one fused conv1d over q|k|v. The HF checkpoint may instead use separate
        // q/k/v conv modules. Check linear_attn.conv1d keys in the safetensors index before loading.
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        // Placeholders. Checkpoint values replace them at load time.
        _aLog.wrappedValue = MLXArray.zeros([numVHeads])
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])

        // HF hard-codes eps=1e-5 for the output norm (FLA default).
        _oNorm.wrappedValue = SharpAIOlmoHybridRMSNormGated(dimensions: headVDim, eps: 1e-5)

        super.init()
    }

    /// `mask` is the [B, S] padding mask (1 for real tokens), or nil. Padded inputs are zeroed first, as in HF.
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: MambaCache?
    ) -> MLXArray {
        let zeroConv = MLXArray.zeros([x.dim(0), convKernelSize - 1, convDim], dtype: x.dtype)
        let (output, newConvState, newRecState) = forward(
            x,
            convState: cache?[0] ?? zeroConv,
            recState: cache?[1],
            mask: mask)
        if let cache {
            cache[0] = newConvState
            cache[1] = newRecState
            cache.advance(x.dim(1))
        }
        return output
    }

    func forward(
        _ input: MLXArray, convState: MLXArray, recState: MLXArray?, mask: MLXArray?
    ) -> (output: MLXArray, convState: MLXArray, recState: MLXArray) {
        let B = input.dim(0)
        let S = input.dim(1)

        var x = input
        if let mask {
            x = MLX.where(expandedDimensions(mask, axis: -1), x, MLXArray.zeros(like: x))
        }

        let qkv = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)
        let convInput = concatenated([convState, qkv], axis: 1)
        let convOut = silu(conv1d(convInput))
        let newConvState = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])

        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = parts[0].reshaped(B, S, numKHeads, headKDim)
        var k = parts[1].reshaped(B, S, numKHeads, headKDim)
        let v = parts[2].reshaped(B, S, numVHeads, headVDim)

        // HF: q = l2norm(q) * Dk^-0.5, k = l2norm(k). The l2 eps (1e-6) is added to the sum of squares.
        // rmsNorm adds eps to the mean, so pass eps / Dk to get the same result.
        let invScale = pow(Float(headKDim), -0.5)
        let qkEps = 1e-6 / Float(headKDim)
        q =
            MLXArray(invScale * invScale).asType(x.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: qkEps)
        k =
            MLXArray(invScale).asType(x.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: qkEps)

        if numVHeads > numKHeads {
            let repeatFactor = numVHeads / numKHeads
            q = repeated(q, count: repeatFactor, axis: 2)
            k = repeated(k, count: repeatFactor, axis: 2)
        }

        let betaScale: Float = allowNegEigval ? 2 : 1
        let beta = sigmoid(bProj(x).asType(.float32)) * MLXArray(betaScale)

        // Decay exp(-exp(A_log) * softplus(a + dt_bias)), per value head.
        let biased = aProj(x).asType(.float32) + dtBias.asType(.float32)
        let logDecay = MLXArray(Float(-1)) * exp(aLog.asType(.float32)) * softplus(biased)
        let decay = exp(logDecay)
        let g = broadcast(
            expandedDimensions(decay, axis: -1), to: [B, S, numVHeads, headKDim])

        let (y, newRecState) = SharpAIOverlayGatedDeltaKDA.update(
            q: q, k: k, v: v, g: g, beta: beta,
            state: recState, mask: nil, useKernel: !training)

        let gate = gProj(x).reshaped(B, S, numVHeads, headVDim)
        let normed = oNorm(y, gate: gate)
        return (oProj(normed.reshaped(B, S, -1)), newConvState, newRecState)
    }
}
