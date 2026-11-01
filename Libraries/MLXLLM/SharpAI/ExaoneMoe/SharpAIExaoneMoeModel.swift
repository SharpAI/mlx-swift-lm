//
//  SharpAIExaoneMoeModel.swift
//  MLXLLM
//
//  Text model for `exaone_moe` (K-EXAONE). Pre-norm decoder: attention, then a dense
//  MLP or a sparse MoE block per layer (mlp_layer_types). Weight keys follow the HF
//  checkpoint names. Sanitize drops MTP weights, moves the router bias onto the MoE
//  block, and stacks routed experts into `switch_mlp`.
//  UNVERIFIED: no real checkpoint was run. The layout comes from vLLM exaone_moe.py.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIExaoneMoeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIExaoneMoeAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let slidingWindow: Int?

    /// `mtp: true` builds an MTP layer. MTP layers always use a dense MLP (vLLM `is_mtp`).
    init(_ config: SharpAIExaoneMoeConfiguration, layer: Int, mtp: Bool) {
        _attention.wrappedValue = SharpAIExaoneMoeAttention(config, layer: layer, mtp: mtp)

        if !mtp && config.isSparseLayer(layer) {
            _mlp.wrappedValue = SharpAIExaoneMoeSparseMoE(config, layer: layer)
        } else {
            _mlp.wrappedValue = SharpAIExaoneMoeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                bias: config.mlpBias,
                // UNVERIFIED: MTP index is not a main-layer index, so MTP gets no clamp.
                swigluLimit: mtp ? nil : config.swigluLimit(layer: layer))
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.slidingWindow = config.windowSize(layer: layer, mtp: mtp)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAIExaoneMoeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIExaoneMoeDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIExaoneMoeConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIExaoneMoeDecoderLayer(config, layer: $0, mtp: false)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            // Each layer builds its own mask. Sliding layers use their window, and
            // full-attention layers use none.
            let layerCache = cache?[i]
            let mask = createAttentionMask(h: h, cache: layerCache, windowSize: layer.slidingWindow)
            h = layer(h, mask: mask, cache: layerCache)
        }
        return norm(h)
    }
}

class SharpAIExaoneMoeModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAIExaoneMoeModelInner
    let configuration: SharpAIExaoneMoeConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIExaoneMoeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIExaoneMoeModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Drops MTP and rotary buffers, drops the tied LM head, then stacks routed experts
    /// into `switch_mlp` for each sparse layer.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        let layerCount = configuration.hiddenLayers
        sanitized = sanitized.filter {
            // MTP weights belong to the drafter, not this model (vLLM maps `mtp.` to None).
            if $0.key.hasPrefix("mtp.") { return false }
            if $0.key.contains("rotary_emb.inv_freq") { return false }
            guard let index = SharpAIExaoneMoeModel.layerIndex(of: $0.key) else { return true }
            return index < layerCount
        }

        let numExperts = configuration.numExperts
        for l in 0 ..< layerCount where configuration.isSparseLayer(l) {
            let prefix = "model.layers.\(l).mlp"
            // UNVERIFIED: checkpoint key location. Accept the gate path and move it onto the MoE block.
            if let bias = sanitized.removeValue(forKey: "\(prefix).gate.e_score_correction_bias") {
                sanitized["\(prefix).e_score_correction_bias"] = bias
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for k in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).experts.0.\(proj).\(k)"] != nil else { continue }
                    // Missing experts are skipped here. A short checkpoint fails at load time.
                    let toJoin = (0 ..< numExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(proj).\(k)")
                    }
                    sanitized["\(prefix).switch_mlp.\(proj).\(k)"] = MLX.stacked(toJoin)
                }
            }
        }

        return sanitized
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { (layer) throws -> KVCache in
            if let window = layer.slidingWindow {
                return try makeSlidingWindowKVCache(parameters: parameters, window: window)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    /// Layer index for keys like `model.layers.12.x`, nil otherwise.
    static func layerIndex(of key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        let digits = key.dropFirst(prefix.count).prefix { $0.isNumber }
        return Int(digits)
    }
}

extension SharpAIExaoneMoeModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
