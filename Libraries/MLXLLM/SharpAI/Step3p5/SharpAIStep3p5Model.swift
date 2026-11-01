//
//  SharpAIStep3p5Model.swift
//  MLXLLM
//
//  Text model for `step3p5` (Step3p5ForCausalLM). Pre-norm decoder: attention, then a dense
//  SwiGLU MLP or a sparse MoE block per layer (moe_layers_enum). Weight keys follow the HF
//  checkpoint names. Sanitize handles key remapping and expert stacking.
//  The MTP draft layers (index >= num_hidden_layers) are not built, and sanitize drops their keys.
//  UNVERIFIED: no real checkpoint was run. The layout comes from vLLM step3p5.py.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIStep3p5DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIStep3p5Attention
    @ModuleInfo(key: "mlp") var mlp: SharpAIStep3p5MLP?
    @ModuleInfo(key: "moe") var moe: SharpAIStep3p5MoE?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIStep3p5RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: SharpAIStep3p5RMSNorm

    let slidingWindow: Int?

    init(_ config: SharpAIStep3p5Configuration, layer: Int) {
        _attention.wrappedValue = SharpAIStep3p5Attention(config, layer: layer)

        if config.isMoELayer(layer) {
            _moe.wrappedValue = SharpAIStep3p5MoE(config, layer: layer)
            _mlp.wrappedValue = nil
        } else {
            _mlp.wrappedValue = SharpAIStep3p5MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                limit: config.sharedSwigluLimit(layer: layer))
            _moe.wrappedValue = nil
        }

        _inputLayerNorm.wrappedValue = SharpAIStep3p5RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIStep3p5RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.slidingWindow = config.windowSize(layer: layer)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        let normed = postAttentionLayerNorm(h)
        if let moe {
            return h + moe(normed)
        }
        guard let mlp else {
            preconditionFailure("step3p5 layer has neither a dense MLP nor a MoE block")
        }
        return h + mlp(normed)
    }
}

class SharpAIStep3p5ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIStep3p5DecoderLayer]
    let norm: SharpAIStep3p5RMSNorm

    init(_ config: SharpAIStep3p5Configuration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIStep3p5DecoderLayer(config, layer: $0)
        }
        self.norm = SharpAIStep3p5RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
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

class SharpAIStep3p5Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAIStep3p5ModelInner
    let configuration: SharpAIStep3p5Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIStep3p5Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.attentionGroups }
        self.model = SharpAIStep3p5ModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        } else {
            _lmHead.wrappedValue = nil
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIStep3p5Sanitizer.sanitize(weights, configuration: configuration)
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

extension SharpAIStep3p5Model: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
