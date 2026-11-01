// SharpAI fork overlay: unlimited_ocr text decoder. Not part of upstream mlx-swift-lm.
//
// A DeepSeek-V2 decoder with standard multi-head attention (no MLA), a dense first layer and
// mixture-of-experts layers with shared experts. The forward pass is full causal attention with
// rotate-half RoPE, which is what the reference forward computes once config.sliding_window is
// cleared (the reference does this in generate paths).
// Verified against: https://huggingface.co/baidu/Unlimited-OCR/raw/main/modeling_deepseekv2.py
//   (DeepseekV2MLP, MoEGate with topk_method greedy and norm_topk_prob false, DeepseekV2MoE with
//   shared_experts, DeepseekV2DecoderLayer, mha_eager attention).
// NOT ported: the ring-buffer decode in SlidingWindowLlamaAttention, which only runs when
// config._ring_window is set by the reference infer() path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIUnlimitedOcrAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIUnlimitedOcrTextConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        // Llama-style RoPE with rotate-half pairing, base rope_theta (10000 by default).
        self.rope = initializeRope(
            dims: headDim, base: config.ropeTheta, traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class SharpAIUnlimitedOcrDenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Greedy top-k router: softmax over all routed experts, then the top-k scores, no renormalization
/// unless norm_topk_prob is set. Matches MoEGate.forward in modeling_deepseekv2.py.
final class SharpAIUnlimitedOcrMoEGate: Module {
    let topK: Int
    let routedScalingFactor: Float
    let normTopKProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ config: SharpAIUnlimitedOcrTextConfiguration) {
        self.topK = config.numExpertsPerTok
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopKProb = config.normTopKProb
        _weight.wrappedValue = zeros([config.nRoutedExperts, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Router logits are computed in float32, as in MoEGate.forward.
        let gates = x.asType(.float32).matmul(weight.T.asType(.float32))
        let scores = softmax(gates, axis: -1, precise: true)
        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if topK > 1 && normTopKProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor
        return (inds, weights.asType(x.dtype))
    }
}

final class SharpAIUnlimitedOcrMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: SharpAIUnlimitedOcrMoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIUnlimitedOcrDenseMLP?

    init(_ config: SharpAIUnlimitedOcrTextConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts)
        _gate.wrappedValue = SharpAIUnlimitedOcrMoEGate(config)
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAIUnlimitedOcrDenseMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        // switchMLP returns (B, L, k, hidden); weight each selected expert and sum over k.
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

final class SharpAIUnlimitedOcrDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIUnlimitedOcrAttention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIUnlimitedOcrTextConfiguration, layerIdx: Int) {
        _selfAttn.wrappedValue = SharpAIUnlimitedOcrAttention(config)
        if config.nRoutedExperts > 0,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = SharpAIUnlimitedOcrMoE(config)
        } else {
            self.mlp = SharpAIUnlimitedOcrDenseMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}
