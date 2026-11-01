//
//  SharpAIBailingMoeV3KDA.swift
//  MLXLLM
//
//  Kimi Delta Attention (KDA) linear-attention layer for bailing_moe_v3 (Ling-3.0), with
//  no_kda_lora = true (full-rank f_proj / g_proj / b_proj). Follows the HF reference
//  BailingMoeV3KimiDeltaAttention and vLLM bailing_moe_v3:
//
//  - q, k, v: full-rank projections, each followed by a depthwise causal short conv
//    (kernel = short_conv_kernel_size) and SiLU. q and k are L2-normalized; q is scaled by d^-0.5.
//  - Log-decay per key dimension, from f = f_proj(x):
//      kda_safe_gate:   g = kda_lower_bound * sigmoid(exp(A_log) * (f + dt_bias))
//      otherwise:       g = -exp(A_log) * softplus(f + dt_bias)
//  - beta = sigmoid(b_proj(x)), per head.
//  - Gated delta rule (fla naive_recurrent_kda): S = S * exp(g); S += outer(beta * k, v - k^T S);
//    o = q^T S. The recurrence runs in SharpAIOverlayGatedDeltaKDA (MLXLMCommon).
//  - Output: o_norm(o) * sigmoid(g_proj(x)) per head (RMSNorm over head_dim), then o_proj.
//
//  The attention mask is ignored, as in the HF reference (attention_mask is set to None).
//  Cache: ArraysCache with 4 slots: conv state for q, k, v, then the recurrent state.
//  UNVERIFIED: the decoder attribute is `attention` for both MLA and KDA layers (HF
//  reference), and the ShortConvolution has no bias. Neither was checked against a real checkpoint.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIBailingMoeV3KDA: Module, SharpAIBailingMoeV3TokenMixer {
    /// Cache slots: 0 = q conv state, 1 = k conv state, 2 = v conv state, 3 = recurrent state.
    static let cacheSlotCount = 4

    let numHeads: Int
    let headDim: Int
    let projectionSize: Int
    let convKernelSize: Int
    let safeGate: Bool
    let lowerBound: Float
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_conv1d") var qConv1d: Conv1d
    @ModuleInfo(key: "k_conv1d") var kConv1d: Conv1d
    @ModuleInfo(key: "v_conv1d") var vConv1d: Conv1d
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "f_proj") var fProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ config: SharpAIBailingMoeV3Configuration) {
        numHeads = config.attentionHeads
        headDim = config.headDim
        projectionSize = numHeads * headDim
        convKernelSize = config.shortConvKernelSize
        safeGate = config.kdaSafeGate
        lowerBound = config.kdaLowerBound
        scale = pow(Float(headDim), -0.5)

        let hidden = config.hiddenSize
        let p = projectionSize
        _qProj.wrappedValue = Linear(hidden, p, bias: false)
        _kProj.wrappedValue = Linear(hidden, p, bias: false)
        _vProj.wrappedValue = Linear(hidden, p, bias: false)
        _qConv1d.wrappedValue = Conv1d(
            inputChannels: p, outputChannels: p, kernelSize: convKernelSize, stride: 1,
            padding: 0, dilation: 1, groups: p, bias: false)
        _kConv1d.wrappedValue = Conv1d(
            inputChannels: p, outputChannels: p, kernelSize: convKernelSize, stride: 1,
            padding: 0, dilation: 1, groups: p, bias: false)
        _vConv1d.wrappedValue = Conv1d(
            inputChannels: p, outputChannels: p, kernelSize: convKernelSize, stride: 1,
            padding: 0, dilation: 1, groups: p, bias: false)
        _bProj.wrappedValue = Linear(hidden, numHeads, bias: false)
        _fProj.wrappedValue = Linear(hidden, p, bias: false)
        _gProj.wrappedValue = Linear(hidden, p, bias: false)
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _oProj.wrappedValue = Linear(p, hidden, bias: false)

        _aLog.wrappedValue = log(MLXRandom.uniform(low: 1, high: 16, [numHeads]))
        _dtBias.wrappedValue = MLXArray.zeros([p], dtype: .float32)

        super.init()
    }

    /// Log-decay per key dimension, `[B, T, H, d]` float32.
    /// Safe gate: `lowerBound * sigmoid(exp(A_log) * (f + dt_bias))`.
    /// Otherwise: `-exp(A_log) * softplus(f + dt_bias)`.
    static func logDecay(
        f: MLXArray, aLog: MLXArray, dtBias: MLXArray, numHeads: Int, headDim: Int,
        safeGate: Bool, lowerBound: Float
    ) -> MLXArray {
        let (B, T) = (f.dim(0), f.dim(1))
        let aExp = MLX.exp(aLog.asType(.float32)).reshaped(1, 1, numHeads, 1)
        let biased =
            f.asType(.float32).reshaped(B, T, numHeads, headDim)
            + dtBias.asType(.float32).reshaped(numHeads, headDim)
        if safeGate {
            return MLXArray(lowerBound) * sigmoid(aExp * biased)
        }
        // softplus with the same threshold as the usual stable form.
        let softplus = MLX.where(
            MLX.greater(biased, MLXArray(Float(20))),
            biased,
            MLX.log(MLX.exp(biased) + MLXArray(Float(1))))
        return MLXArray(Float(-1)) * aExp * softplus
    }

    func mix(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // The mask is not used by KDA (HF reference sets attention_mask = None).
        let arrays: ArraysCache? = cache.map { c in
            guard let arrays = c as? ArraysCache else {
                preconditionFailure("KDA layer needs an ArraysCache (\(Self.cacheSlotCount) slots)")
            }
            return arrays
        }
        let (B, T) = (x.dim(0), x.dim(1))
        let (H, D, P) = (numHeads, headDim, projectionSize)

        let (qConv, qState) = shortConv(qProj(x), conv: qConv1d, state: arrays?[0])
        let (kConv, kState) = shortConv(kProj(x), conv: kConv1d, state: arrays?[1])
        let (vConv, vState) = shortConv(vProj(x), conv: vConv1d, state: arrays?[2])

        let q = l2Normalize(qConv.reshaped(B, T, H, D)) * MLXArray(scale).asType(x.dtype)
        let k = l2Normalize(kConv.reshaped(B, T, H, D))
        let v = vConv.reshaped(B, T, H, D)

        let logG = Self.logDecay(
            f: fProj(x), aLog: aLog, dtBias: dtBias, numHeads: H, headDim: D,
            safeGate: safeGate, lowerBound: lowerBound)
        let decay = MLX.exp(logG)  // per-key multiplier, [B, T, H, D]
        let beta = sigmoid(bProj(x).asType(.float32))  // [B, T, H]

        let (y, recState) = SharpAIOverlayGatedDeltaKDA.update(
            q: q, k: k, v: v, g: decay, beta: beta,
            state: arrays?[3], mask: nil, useKernel: !training)

        let gate = gProj(x).reshaped(B, T, H, D)
        let normed = oNorm(y.asType(x.dtype)) * sigmoid(gate)
        let out = oProj(normed.reshaped(B, T, P))

        if let arrays {
            arrays[0] = qState
            arrays[1] = kState
            arrays[2] = vState
            arrays[3] = recState
            arrays.offset += T
        }
        return out
    }

    /// Causal depthwise conv over the time axis with SiLU. `state` holds the last
    /// `kernel - 1` inputs of earlier calls. Returns the activated output and the new state.
    private func shortConv(_ x: MLXArray, conv: Conv1d, state: MLXArray?) -> (MLXArray, MLXArray) {
        let B = x.dim(0)
        let prefix = state ?? MLXArray.zeros([B, convKernelSize - 1, x.dim(2)], dtype: x.dtype)
        let input = concatenated([prefix, x], axis: 1)
        let out = silu(conv(input))
        let newState = contiguous(
            input[0..., (input.dim(1) - (convKernelSize - 1))..., 0...])
        return (out, newState)
    }

    /// Row-wise L2 norm with eps added to the sum of squares, as in the KDA kernel.
    private func l2Normalize(_ x: MLXArray) -> MLXArray {
        let sumSquares = MLX.sum(MLX.square(x.asType(.float32)), axis: -1, keepDims: true)
        return (x.asType(.float32) * MLX.rsqrt(sumSquares + MLXArray(Float(1e-6)))).asType(x.dtype)
    }
}
