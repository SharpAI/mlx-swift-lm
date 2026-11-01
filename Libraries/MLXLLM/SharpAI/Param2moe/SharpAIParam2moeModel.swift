//
//  SharpAIParam2moeModel.swift
//  MLXLLM
//
//  Text model for `param2moe` (Param2MoEForCausalLM). Pre-norm decoder: attention, then a dense
//  MLP for the first `first_k_dense_replace` layers and a sparse MoE block after that.
//  Weight keys follow the HF checkpoint names. Sanitize stacks routed experts into `switch_mlp`
//  and zero-means the router bias, as vLLM does at load time.
//  UNVERIFIED: no checkpoint was run. Per-expert names (experts.{e}.gate_proj etc.) follow the
//  vLLM default mapping. MTP layers are not implemented (the fixture has none).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIParam2moeDecoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: SharpAIParam2moeAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIParam2moeConfiguration, layerIdx: Int) {
        _attention.wrappedValue = SharpAIParam2moeAttention(config)

        if config.isSparseLayer(layerIdx) {
            _mlp.wrappedValue = SharpAIParam2moeSparseMoE(config)
        } else {
            _mlp.wrappedValue = SharpAIParam2moeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                bias: config.useBias)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAIParam2moeModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var embedTokens: Embedding
    let layers: [SharpAIParam2moeDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIParam2moeConfiguration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIParam2moeDecoderLayer(config, layerIdx: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

class SharpAIParam2moeModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAIParam2moeModelInner
    let configuration: SharpAIParam2moeConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIParam2moeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIParam2moeModelInner(config)

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

    /// Drops rotary buffers and the tied LM head. Zero-means the router bias, then stacks
    /// routed experts into `switch_mlp` for each sparse layer.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        sanitized = sanitized.filter { !$0.key.contains("rotary_emb.inv_freq") }

        for l in 0 ..< configuration.hiddenLayers where configuration.isSparseLayer(l) {
            let prefix = "model.layers.\(l).mlp"

            // A constant shift keeps the top-k order, so this only matches vLLM's load step.
            if let bias = sanitized["\(prefix).gate.expert_bias"] {
                sanitized["\(prefix).gate.expert_bias"] =
                    bias - bias.mean(axis: -1, keepDims: true)
            }

            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for k in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).experts.0.\(proj).\(k)"] != nil else { continue }
                    // Missing experts are skipped here. A short checkpoint fails at load time.
                    let toJoin = (0 ..< configuration.numExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(proj).\(k)")
                    }
                    sanitized["\(prefix).switch_mlp.\(proj).\(k)"] = MLX.stacked(toJoin)
                }
            }
        }

        return sanitized
    }
}

extension SharpAIParam2moeModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
