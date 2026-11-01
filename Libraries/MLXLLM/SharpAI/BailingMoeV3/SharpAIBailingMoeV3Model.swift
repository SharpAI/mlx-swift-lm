//
//  SharpAIBailingMoeV3Model.swift
//  MLXLLM
//
//  Text model for bailing_moe_v3. Layers are MLA (full attention) or KDA (linear attention),
//  chosen by config.isKDALayer. Configs with q_lora_rank, low-rank KDA (no_kda_lora=false)
//  or SwiGLU clamps are rejected by validateModelConfiguration().
//  Weight keys follow the HF checkpoint: model.word_embeddings, model.layers.N.*,
//  model.norm, lm_head. Both layer kinds sit under model.layers.N.attention.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Token mixer shared by MLA and KDA layers. `cache` is a KV cache for MLA layers and an
/// ArraysCache for KDA layers.
protocol SharpAIBailingMoeV3TokenMixer {
    func mix(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?)
        -> MLXArray
}

extension SharpAIBailingMoeV3Attention: SharpAIBailingMoeV3TokenMixer {
    func mix(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?)
        -> MLXArray
    {
        callAsFunction(x, mask: mask, cache: cache)
    }
}

class SharpAIBailingMoeV3DecoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: Module & SharpAIBailingMoeV3TokenMixer
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    /// True for KDA linear-attention layers.
    let isLinear: Bool

    init(_ config: SharpAIBailingMoeV3Configuration, layerIndex: Int) {
        isLinear = config.isKDALayer(layerIndex)
        if isLinear {
            _attention.wrappedValue = SharpAIBailingMoeV3KDA(config)
        } else {
            _attention.wrappedValue = SharpAIBailingMoeV3Attention(config)
        }
        if layerIndex >= config.firstKDenseReplace {
            _mlp.wrappedValue = SharpAIBailingMoeV3SparseMoE(config)
        } else {
            _mlp.wrappedValue = SharpAIBailingMoeV3MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                bias: config.useBias
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention.mix(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAIBailingMoeV3ModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var embedTokens: Embedding
    let layers: [SharpAIBailingMoeV3DecoderLayer]
    let norm: RMSNorm
    /// Index of the first MLA layer, whose KV cache defines the attention mask.
    let faIdx: Int?

    init(_ config: SharpAIBailingMoeV3Configuration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        let builtLayers = (0 ..< config.hiddenLayers).map {
            SharpAIBailingMoeV3DecoderLayer(config, layerIndex: $0)
        }
        self.layers = builtLayers
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.faIdx = builtLayers.firstIndex { !$0.isLinear }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let attentionCache = faIdx.flatMap { cache?[$0] }
        let mask = createAttentionMask(h: h, cache: attentionCache)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

class SharpAIBailingMoeV3Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAIBailingMoeV3ModelInner
    let configuration: SharpAIBailingMoeV3Configuration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIBailingMoeV3Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.attentionHeads }
        self.model = SharpAIBailingMoeV3ModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    /// MLA layers get the standard attention KV cache. KDA layers get an ArraysCache with
    /// conv states for q, k, v and the recurrent state.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { layer -> KVCache in
            if layer.isLinear {
                return ArraysCache(size: SharpAIBailingMoeV3KDA.cacheSlotCount)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    /// Drops MTP layers and rotary caches, then stacks routed experts into `switch_mlp`.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        let layerCount = configuration.hiddenLayers
        sanitized = sanitized.filter {
            if $0.key.contains("rotary_emb.inv_freq") { return false }
            guard let index = SharpAIBailingMoeV3Model.layerIndex(of: $0.key) else { return true }
            return index < layerCount
        }

        for l in 0 ..< layerCount where l >= configuration.firstKDenseReplace {
            let prefix = "model.layers.\(l)"
            for n in ["gate_proj", "up_proj", "down_proj"] {
                for k in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).mlp.experts.0.\(n).\(k)"] != nil else { continue }
                    let toJoin = (0 ..< configuration.numExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")
                    }
                    sanitized["\(prefix).mlp.switch_mlp.\(n).\(k)"] = MLX.stacked(toJoin)
                }
            }
        }

        // HF ShortConvolution weights are [C, 1, K]; MLX Conv1d wants [C, K, 1].
        // Kernel size is at least 2 in practice, so a middle axis of 1 identifies the HF layout.
        for key in Array(sanitized.keys) where key.hasSuffix("_conv1d.weight") {
            guard let weight = sanitized[key], weight.ndim == 3, weight.dim(1) == 1 else {
                continue
            }
            sanitized[key] = weight.movedAxis(source: 2, destination: 1)
        }

        return sanitized
    }

    /// Layer index for keys like `model.layers.12.x`, nil otherwise.
    static func layerIndex(of key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        let digits = key.dropFirst(prefix.count).prefix { $0.isNumber }
        return Int(digits)
    }
}

extension SharpAIBailingMoeV3Model: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
