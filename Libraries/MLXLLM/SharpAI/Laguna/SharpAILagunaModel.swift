//
//  SharpAILagunaModel.swift
//  MLXLLM
//
//  Text model for `laguna` (LagunaForCausalLM). Pre-norm decoder: attention, then a dense
//  MLP or a sparse MoE block per layer (mlp_layer_types). Sliding layers use their window
//  and full-attention layers use none. Weight keys follow the HF checkpoint names.
//  Sanitize stacks routed experts into `switch_mlp` and moves the router bias.
//  UNVERIFIED: no real checkpoint was run. The layout comes from vLLM laguna.py.
//  The DFlash drafter (DFlashLagunaForCausalLM) is not implemented and is rejected by validation.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAILagunaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAILagunaAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let slidingWindow: Int?

    init(_ config: SharpAILagunaConfiguration, layer: Int) {
        _attention.wrappedValue = SharpAILagunaAttention(config, layer: layer)

        if config.isSparseLayer(layer) {
            _mlp.wrappedValue = SharpAILagunaSparseMoE(config)
        } else {
            _mlp.wrappedValue = SharpAILagunaMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.slidingWindow = config.windowSize(layer: layer)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAILagunaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAILagunaDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAILagunaConfiguration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAILagunaDecoderLayer(config, layer: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            // Each layer builds its own mask. Sliding layers use their window.
            let layerCache = cache?[i]
            let mask = createAttentionMask(h: h, cache: layerCache, windowSize: layer.slidingWindow)
            h = layer(h, mask: mask, cache: layerCache)
        }
        return norm(h)
    }
}

class SharpAILagunaModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAILagunaModelInner
    let configuration: SharpAILagunaConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAILagunaConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAILagunaModelInner(config)

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

    /// Drops rotary buffers and the tied LM head. Moves the router bias onto the gate, then
    /// stacks routed experts into `switch_mlp` for each sparse layer.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        sanitized = sanitized.filter { !$0.key.contains("rotary_emb.inv_freq") }

        let numExperts = configuration.numExperts
        for l in 0 ..< configuration.hiddenLayers where configuration.isSparseLayer(l) {
            let prefix = "model.layers.\(l).mlp"

            if let bias = sanitized.removeValue(forKey: "\(prefix).experts.e_score_correction_bias")
            {
                sanitized["\(prefix).gate.e_score_correction_bias"] = bias
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
}

extension SharpAILagunaModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
