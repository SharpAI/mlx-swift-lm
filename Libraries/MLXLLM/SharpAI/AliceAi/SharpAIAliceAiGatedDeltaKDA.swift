// SharpAI fork overlay: alice_ai Kimi Delta Attention block. Not part of upstream mlx-swift-lm.
//
// Port of vLLM's KimiGatedDeltaNetAttention with use_full_rank_gate=False.
// Decay: g = exp(-exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)); beta = sigmoid(b_proj(x)).
// The recurrence runs in SharpAIOverlayGatedDeltaKDA (MLXLMCommon).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIAliceAiGatedDeltaKDA: Module {
    let numHeads: Int
    let headDim: Int
    let projectionSize: Int
    let convKernelSize: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    /// Depthwise conv over the concatenated q, k, v channels. Sanitize merges the q/k/v checkpoint weights.
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ args: SharpAIAliceAiConfiguration) {
        numHeads = args.linearNumValueHeads
        headDim = args.linearValueHeadDim
        projectionSize = numHeads * headDim
        convKernelSize = args.linearConvKernelDim
        scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, projectionSize, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, projectionSize, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, projectionSize, bias: false)
        _bProj.wrappedValue = Linear(args.hiddenSize, numHeads, bias: false)
        _fAProj.wrappedValue = Linear(args.hiddenSize, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projectionSize, bias: false)
        _gAProj.wrappedValue = Linear(args.hiddenSize, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectionSize, bias: false)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: 3 * projectionSize,
            outputChannels: 3 * projectionSize,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 3 * projectionSize,
            bias: false
        )

        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionSize, args.hiddenSize, bias: false)

        _aLog.wrappedValue = log(MLXRandom.uniform(low: 1, high: 16, [numHeads]))
        _dtBias.wrappedValue = MLXArray.zeros([projectionSize])

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: MambaCache? = nil) -> MLXArray {
        let zeroConv = MLXArray.zeros(
            [x.dim(0), convKernelSize - 1, 3 * projectionSize], dtype: x.dtype)
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
        _ x: MLXArray, convState: MLXArray, recState: MLXArray?, mask: MLXArray?
    ) -> (output: MLXArray, convState: MLXArray, recState: MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)
        let P = projectionSize
        let H = numHeads
        let D = headDim

        var qkv = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)
        if let mask {
            qkv = MLX.where(expandedDimensions(mask, axis: -1), qkv, MLXArray.zeros(like: qkv))
        }
        let convInput = concatenated([convState, qkv], axis: 1)
        let convOut = silu(conv1d(convInput))
        let newConvState = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])

        let parts = MLX.split(convOut, indices: [P, 2 * P], axis: -1)
        let q = l2Normalize(parts[0].reshaped(B, S, H, D)) * MLXArray(scale).asType(x.dtype)
        let k = l2Normalize(parts[1].reshaped(B, S, H, D))
        let v = parts[2].reshaped(B, S, H, D)

        let biased = (fBProj(fAProj(x)).reshaped(B, S, H, D) + dtBias.reshaped(H, D))
            .asType(.float32)
        let softplus = MLX.where(
            MLX.greater(biased, MLXArray(Float(20))),
            biased,
            MLX.log(MLX.exp(biased) + MLXArray(Float(1))))
        let aExp = MLX.exp(aLog.asType(.float32)).reshaped(1, 1, H, 1)
        let decay = MLX.exp(MLXArray(Float(-1)) * aExp * softplus)  // [B, S, H, D]
        let beta = sigmoid(bProj(x).asType(.float32))  // [B, S, H]

        let (y, newRecState) = SharpAIOverlayGatedDeltaKDA.update(
            q: q, k: k, v: v, g: decay, beta: beta,
            state: recState, mask: mask, useKernel: !training)

        let gate = gBProj(gAProj(x)).reshaped(B, S, H, D)
        let normed = oNorm(y.asType(x.dtype)) * sigmoid(gate)
        return (oProj(normed.reshaped(B, S, -1)), newConvState, newRecState)
    }

    /// Row-wise L2 norm with eps added to the sum of squares, as in the KDA kernel.
    private func l2Normalize(_ x: MLXArray) -> MLXArray {
        let sumSquares = MLX.sum(MLX.square(x.asType(.float32)), axis: -1, keepDims: true)
        return (x.asType(.float32) * MLX.rsqrt(sumSquares + MLXArray(Float(1e-6)))).asType(x.dtype)
    }
}
