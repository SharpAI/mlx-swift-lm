// SharpAI fork overlay: kolibri1 (Aleph Alpha Kolibri-1) text model. Not part of upstream mlx-swift-lm.
//
// Architecture, from the transformers `kolibri1` modeling file (modular_kolibri1.py) as
// summarized from the Hugging Face source. The file was read through a summarizer, so the
// exact tensor layout should be re-checked against the raw source before relying on it:
// - Attention: bias-free q/k/v/o. Per-head RMSNorm (q_norm, k_norm, width head_dim) runs
//   before RoPE.
// - RoPE is applied only when sliding_window is unset or the layer is sliding_attention.
//   full_attention layers are NoPE. There is one rotary embedding shared by all layers.
// - Masks follow layer_types: sliding layers use a windowed causal mask, full layers a
//   plain causal mask.
// - Decoder layer uses sandwich norms:
//     h = x + post_self_attn_layernorm(self_attn(input_layernorm(x)))
//     h = h + post_mlp_layernorm(mlp(post_attention_layernorm(h)))
// - MoE router: logits = x @ gate.weight^T in fp32. Top-k is taken over
//   logits + gate.e_score_correction_bias. Weights are sigmoid(logits) at the selected
//   indices, divided by their sum when norm_topk_prob is set.
// - A shared expert (plain SwiGLU, no gate) is added to the routed output.
// - Checkpoint experts are fused (mlp.experts.gate_up_proj, mlp.experts.down_proj). The
//   sanitizer splits them into switch_mlp.{gate,up,down}_proj.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Gated MLP for the shared expert.
class SharpAIKolibri1MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Router for the MoE block. Its parameters are `gate.weight` and
/// `gate.e_score_correction_bias`.
class SharpAIKolibri1Router: Module {
    let topK: Int
    let normTopkProb: Bool
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: SharpAIKolibri1Configuration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self._weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.numExperts])
    }

    /// Returns the selected expert indices and their routing weights, both shaped [..., topK].
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = x.asType(.float32).matmul(weight.T)
        let selection = logits + eScoreCorrectionBias
        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = sigmoid(takeAlong(logits, inds, axis: -1))
        if normTopkProb {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (inds, scores.asType(x.dtype))
    }
}

/// Routed experts (SwitchGLU) plus the shared expert.
class SharpAIKolibri1MoE: Module {
    @ModuleInfo(key: "gate") var gate: SharpAIKolibri1Router
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIKolibri1MLP

    init(_ config: SharpAIKolibri1Configuration) {
        self._gate.wrappedValue = SharpAIKolibri1Router(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            activation: silu
        )
        self._sharedExperts.wrappedValue = SharpAIKolibri1MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.sharedExpertIntermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)  // [B, L, topK, H]
        y = (y * scores.expandedDimensions(axis: -1)).sum(axis: -2)
        return y + sharedExperts(x)
    }
}

class SharpAIKolibri1Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    /// RoPE is applied to this layer.
    let useRope: Bool
    /// Window for the causal mask. Nil for full-attention layers.
    let windowSize: Int?
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAIKolibri1Configuration, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        let isSliding = config.isSlidingAttentionLayer(layerIdx)
        self.useRope = !config.useSlidingWindow || isSliding
        self.windowSize = isSliding ? config.slidingWindow : nil

        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)

        self._qProj.wrappedValue = Linear(
            config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(
            numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qNorm(qProj(x).reshaped(B, L, numHeads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if useRope {
            let offset = cache?.ropeOffset
            q = applyRotaryPosition(rope, to: q, offset: offset)
            k = applyRotaryPosition(rope, to: k, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

class SharpAIKolibri1DecoderLayer: Module {
    let isSliding: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIKolibri1Attention
    var mlp: SharpAIKolibri1MoE
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayernorm: RMSNorm

    init(_ config: SharpAIKolibri1Configuration, layerIdx: Int) {
        self.isSliding = config.isSlidingAttentionLayer(layerIdx)
        self._selfAttn.wrappedValue = SharpAIKolibri1Attention(config, layerIdx: layerIdx)
        self.mlp = SharpAIKolibri1MoE(config)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postSelfAttnLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postMlpLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = x + postSelfAttnLayernorm(selfAttn(inputLayernorm(x), mask: mask, cache: cache))
        h = h + postMlpLayernorm(mlp(postAttentionLayernorm(h)))
        return h
    }
}

public class SharpAIKolibri1ModelInner: Module {
    let config: SharpAIKolibri1Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [SharpAIKolibri1DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIKolibri1Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            SharpAIKolibri1DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)

        for (i, layer) in layers.enumerated() {
            // Each layer builds its mask from its own cache. Sliding layers use their window.
            let layerCache = cache?[i]
            let mask = createAttentionMask(
                h: h, cache: layerCache, windowSize: layer.isSliding ? config.slidingWindow : nil)
            h = layer(h, mask: mask, cache: layerCache)
        }

        return norm(h)
    }
}

public class SharpAIKolibri1Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]

    let args: SharpAIKolibri1Configuration
    public var model: SharpAIKolibri1ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIKolibri1Configuration) {
        self.args = configuration
        self.kvHeads = Array(repeating: configuration.numKeyValueHeads, count: configuration.numHiddenLayers)
        self.model = SharpAIKolibri1ModelInner(configuration)
        if configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { (layer) throws -> KVCache in
            if layer.isSliding {
                return try makeSlidingWindowKVCache(parameters: parameters, window: args.slidingWindow)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    /// Maps HF checkpoint keys to module keys. See `SharpAIKolibri1Sanitizer`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = SharpAIKolibri1Sanitizer.sanitize(weights: weights, config: args)
        if args.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }
        return sanitized
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
