// SharpAI fork overlay: cohere2_moe text model. Not part of upstream mlx-swift-lm.
//
// Port of transformers v5.10.2 modeling_cohere2_moe.py:
// https://github.com/huggingface/transformers/blob/v5.10.2/src/transformers/models/cohere2_moe/modeling_cohere2_moe.py
//
// - One pre-norm feeds both branches: h = x + attn(norm(x)) + mlp(norm(x)).
// - Sliding-attention layers use the window and RoPE. Full-attention layers skip RoPE unless
//   they are dense prefix layers with prefix_dense_sliding_window_pattern == 1.
// - Router: top-k on raw logits first, then softmax (or sigmoid) over the selected scores.
// - Shared experts are combined by averaging (default) or summing.
// Weight keys after sanitize: switch_mlp.{gate,up,down}_proj.weight are stacked per expert.
// UNVERIFIED against a real checkpoint: the per-expert input layout is the sanitizer's assumption.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// RMSNorm, as Cohere2MoeRMSNorm in the source. Computes in float32.
class SharpAICohere2MoeRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = x.asType(.float32)
        let variance = MLX.mean(input * input, axis: -1, keepDims: true)
        let normed = input * rsqrt(variance + eps)
        return (normed.asType(x.dtype)) * weight
    }
}

/// LayerNorm without bias, as Cohere2MoeLayerNorm in the source (weight only).
class SharpAICohere2MoeLayerNorm: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let centered = x - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        return centered * rsqrt(variance + eps) * weight
    }
}

/// The norm the source picks: RMSNorm when rms_norm_eps is set, otherwise LayerNorm.
func sharpAICohere2MoeNorm(_ config: SharpAICohere2MoeConfiguration) -> Module & UnaryLayer {
    if let eps = config.rmsNormEps {
        return SharpAICohere2MoeRMSNorm(dimensions: config.hiddenSize, eps: eps)
    }
    return SharpAICohere2MoeLayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
}

class SharpAICohere2MoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class SharpAICohere2MoeAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    /// RoPE runs on sliding-attention layers and on forced dense prefix layers.
    let useRope: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ config: SharpAICohere2MoeConfiguration, useRope: Bool) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        self.useRope = useRope

        _wq.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _wk.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _wv.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)

        // Interleaved pairs, matching HF's repeat_interleave of the frequencies.
        self.rope = RoPE(dimensions: headDim, traditional: true, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        if useRope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

/// Sparse MoE block: router, routed experts, optional shared experts.
class SharpAICohere2MoeSparseMoE: Module, UnaryLayer {
    let topK: Int
    let expertSelectionFn: String
    let normTopkProb: Bool
    let sharedCombination: String

    /// Router, `mlp.gate.weight` in HF: (num_experts, hidden_size), no bias.
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAICohere2MoeMLP?

    init(_ config: SharpAICohere2MoeConfiguration, numExperts: Int) {
        self.topK = config.numExpertsPerTok
        self.expertSelectionFn = config.expertSelectionFn
        self.normTopkProb = config.normTopkProb
        self.sharedCombination = config.sharedExpertCombinationStrategy

        _gate.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        // HF routed experts use intermediate_size, not moe_intermediate_size.
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: numExperts
        )
        if config.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAICohere2MoeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize * config.numSharedExperts)
        } else {
            _sharedExperts.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x)
        // Top-k on the raw logits first. Scores are computed over the selected entries only.
        let inds = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let selected = takeAlong(logits, inds, axis: -1).asType(.float32)

        var scores: MLXArray
        if expertSelectionFn == "sigmoid" {
            scores = sigmoid(selected)
            if normTopkProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }
        } else {
            scores = softmax(selected, axis: -1)
        }
        scores = scores.asType(x.dtype)

        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, scores)
        if let sharedExperts {
            let combined = y + sharedExperts(x)
            y = sharedCombination == "sum" ? combined : combined * 0.5
        }
        return y
    }
}

class SharpAICohere2MoeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAICohere2MoeAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Module & UnaryLayer

    /// Window for this layer's mask. nil on full-attention layers.
    let slidingWindow: Int?

    init(_ config: SharpAICohere2MoeConfiguration, layerIndex: Int) {
        let isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        let isDensePrefix = layerIndex < config.firstKDenseReplace
        let window: Int? = isSliding ? config.slidingWindow : nil
        let forceRope = isDensePrefix && config.prefixDenseSlidingWindowPattern == 1
        self.slidingWindow = window

        _attention.wrappedValue = SharpAICohere2MoeAttention(
            config, useRope: window != nil || forceRope)

        let mlpModule: Module & UnaryLayer
        if let numExperts = config.numExperts, !isDensePrefix {
            mlpModule = SharpAICohere2MoeSparseMoE(config, numExperts: numExperts)
        } else {
            let denseSize =
                isDensePrefix
                ? (config.prefixDenseIntermediateSize ?? config.intermediateSize)
                : config.intermediateSize
            mlpModule = SharpAICohere2MoeMLP(
                hiddenSize: config.hiddenSize, intermediateSize: denseSize)
        }
        _mlp.wrappedValue = mlpModule
        _inputLayerNorm.wrappedValue = sharpAICohere2MoeNorm(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Parallel residual: both branches read the same normed input.
        let normed = inputLayerNorm(x)
        let attn = attention(normed, mask: mask, cache: cache)
        return x + attn + mlp(normed)
    }
}

class SharpAICohere2MoeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAICohere2MoeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Module & UnaryLayer

    init(_ config: SharpAICohere2MoeConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAICohere2MoeDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = sharpAICohere2MoeNorm(config)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            // Each layer builds its mask from its own window: full layers get none.
            let mask = createAttentionMask(
                h: h, cache: layerCache, windowSize: layer.slidingWindow)
            h = layer(h, mask: mask, cache: layerCache)
        }
        return norm(h)
    }
}

class SharpAICohere2MoeModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAICohere2MoeModelInner
    let configuration: SharpAICohere2MoeConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAICohere2MoeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAICohere2MoeModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        var logits: MLXArray
        if let lmHead {
            logits = lmHead(out)
        } else {
            logits = model.embedTokens.asLinear(out)
        }
        return logits * configuration.logitScale
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAICohere2MoeSanitizer.sanitize(weights, configuration: configuration)
    }
}

extension SharpAICohere2MoeModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
