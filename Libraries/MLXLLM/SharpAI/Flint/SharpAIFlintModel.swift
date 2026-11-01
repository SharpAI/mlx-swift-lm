// SharpAI fork overlay: flint (Qwen3-MoE style decoder with sinks and hybrid attention).
// Not part of upstream mlx-swift-lm.
//
// Ref: vllm/models/flint/model.py. Sliding layers use windowed caches and masks; full layers
// use plain caches. Weight keys match the HF names, so sanitize only stacks expert weights.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIFlintDecoderLayer: Module {
    let isSlidingWindow: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIFlintAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let mlp: UnaryLayer

    init(_ config: SharpAIFlintConfiguration, layerIdx: Int) {
        self.isSlidingWindow = config.layerTypes[layerIdx] == "sliding_attention"

        _selfAttn.wrappedValue = SharpAIFlintAttention(
            config, usesRope: !config.noRopeLayerIndices.contains(layerIdx))
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.mlpOnlyLayers.contains(layerIdx) {
            self.mlp = SharpAIFlintMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        } else {
            self.mlp = SharpAIFlintSparseMoeBlock(config)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let residual = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

public class SharpAIFlintModelInner: Module, LayerPartitionable, StreamableMoE {

    // LayerPartitionable
    public var gpuLayerCount: Int?
    public var totalLayerCount: Int { layers.count }

    // StreamableMoE
    public var streamExperts: Bool = false

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIFlintDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    /// Nil when sliding windows are disabled. Sliding layers then attend over the full prefix.
    let slidingWindow: Int?
    let fullAttentionLayerIndex: Int
    let slidingAttentionLayerIndex: Int

    init(_ config: SharpAIFlintConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { index in
            SharpAIFlintDecoderLayer(config, layerIdx: index)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.slidingWindow = config.effectiveSlidingWindow
        self.fullAttentionLayerIndex =
            config.layerTypes.firstIndex { $0 != "sliding_attention" } ?? 0
        self.slidingAttentionLayerIndex =
            config.layerTypes.firstIndex(of: "sliding_attention") ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(
            h: h, cache: cache?[fullAttentionLayerIndex])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingAttentionLayerIndex], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let usesSlidingMask = layer.isSlidingWindow && slidingWindow != nil
            let mask = usesSlidingMask ? slidingMask : fullMask
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts) {
                layer(h, mask: mask, cache: cache?[i])
            }
        }

        return norm(h)
    }
}

public class SharpAIFlintModel: Module, LLMModel {
    public let vocabularySize: Int
    public let model: SharpAIFlintModelInner
    let configuration: SharpAIFlintConfiguration

    // Always present, as in the vLLM reference (tie_word_embeddings is not applied).
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: SharpAIFlintConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.model = SharpAIFlintModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIFlintSanitizer.sanitize(weights, configuration: configuration)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { layer in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.effectiveSlidingWindow,
                usesSlidingWindow: layer.isSlidingWindow)
        }
    }
}

// MARK: - LoRA

extension SharpAIFlintModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
