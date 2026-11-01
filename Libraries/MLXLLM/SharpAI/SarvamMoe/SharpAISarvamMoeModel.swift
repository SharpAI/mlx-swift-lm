//
//  SharpAISarvamMoeModel.swift
//  MLXLLM
//
//  Text model for sarvam_moe. Layers below first_k_dense_replace use a dense MLP,
//  the rest use the sparse MoE block. Weight keys follow the HF checkpoint names
//  (`model.word_embeddings`, `model.layers.N.attention.*`, `model.layers.N.mlp.*`,
//  `model.norm`, `lm_head`), checked against sarvamai/sarvam-30b's
//  model.safetensors.index.json. sanitize() maps the raw per-expert layout onto switch_mlp.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAISarvamMoeDecoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: SharpAISarvamMoeAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAISarvamMoeConfiguration, layerIndex: Int) {
        _attention.wrappedValue = SharpAISarvamMoeAttention(config)
        if layerIndex >= config.firstKDenseReplace && config.numExperts > 0 {
            _mlp.wrappedValue = SharpAISarvamMoeSparseMoE(config)
        } else {
            _mlp.wrappedValue = SharpAISarvamMoeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize
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
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAISarvamMoeModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var embedTokens: Embedding
    let layers: [SharpAISarvamMoeDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAISarvamMoeConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAISarvamMoeDecoderLayer(config, layerIndex: $0)
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

class SharpAISarvamMoeModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAISarvamMoeModelInner
    let configuration: SharpAISarvamMoeConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAISarvamMoeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAISarvamMoeModelInner(config)
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

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAISarvamMoeSanitize.sanitize(weights, configuration: configuration)
    }
}

extension SharpAISarvamMoeModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
