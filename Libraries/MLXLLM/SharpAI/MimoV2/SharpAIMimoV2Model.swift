//
//  SharpAIMimoV2Model.swift
//  MLXLLM
//
//  Text model for mimo_v2 (MiMoV2ForCausalLM). Layers alternate between
//  sliding-window and global attention by hybrid_layer_pattern. Dense and MoE
//  blocks follow moe_layer_freq.
//  The MTP layer is built only when retainMTP is set, and it is not wired into
//  generation. Omni towers are rejected at config decode.
//  UNVERIFIED: no checkpoint weights or real config were loaded.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIMimoV2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMimoV2Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    let mlp: UnaryLayer
    let isSlidingWindow: Bool

    init(_ config: SharpAIMimoV2Configuration, layerIndex: Int) {
        let slidingWindow = config.isSlidingWindowLayer(layerIndex)
        self.isSlidingWindow = slidingWindow

        _selfAttn.wrappedValue = SharpAIMimoV2Attention(config, isSlidingWindow: slidingWindow)

        if config.isMoELayer(layerIndex) {
            self.mlp = SharpAIMimoV2MoE(config)
        } else {
            self.mlp = SharpAIMimoV2MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let residual = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

class SharpAIMimoV2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIMimoV2DecoderLayer]
    let norm: RMSNorm
    let hybridLayerPattern: [Int]
    let slidingWindowSize: Int

    @ModuleInfo(key: "mtp") var mtp: SharpAIMimoV2MTPLayers?

    init(_ config: SharpAIMimoV2Configuration, retainMTP: Bool) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIMimoV2DecoderLayer(config, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        self.hybridLayerPattern = config.hybridLayerPattern
        self.slidingWindowSize = config.slidingWindowSize

        let mtpLayers: SharpAIMimoV2MTPLayers? =
            (retainMTP && config.numNextnPredictLayers > 0)
            ? SharpAIMimoV2MTPLayers(config) : nil
        _mtp.wrappedValue = mtpLayers
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        // Masks come from the first layer of each kind. With no global layer the
        // full-attention index falls back to 0 and its mask is never used.
        let fullIndex = hybridLayerPattern.firstIndex(of: 0) ?? 0
        let swaIndex = hybridLayerPattern.firstIndex(of: 1) ?? 0
        let fullMask = createAttentionMask(h: h, cache: cache?[fullIndex])
        let swaMask = createAttentionMask(
            h: h, cache: cache?[swaIndex], windowSize: slidingWindowSize)

        for (i, layer) in layers.enumerated() {
            let mask = hybridLayerPattern[i] == 1 ? swaMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

class SharpAIMimoV2Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]

    let model: SharpAIMimoV2ModelInner
    let configuration: SharpAIMimoV2Configuration
    let retainMTP: Bool

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIMimoV2Configuration, retainMTP: Bool = MTPConfig.retainMTPWeights) {
        self.configuration = config
        self.retainMTP = retainMTP
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIMimoV2ModelInner(config, retainMTP: retainMTP)

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

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIMimoV2Sanitizer.sanitize(
            weights, configuration: configuration, retainMTP: retainMTP)
    }

    /// Hidden state from the MTP layer for `tokens` and the previous hidden state.
    /// Returns nil when MTP weights are not retained. Not wired into generation.
    /// Verified against vLLM mimo_v2_mtp.py: num_mtp_layers is hard-coded to 1.
    /// UNVERIFIED: no checkpoint run.
    func mtpHiddenStates(
        tokens: MLXArray, previousHidden: MLXArray, cache: KVCache?
    ) -> MLXArray? {
        guard let layer = model.mtp?.layers.first else { return nil }
        let embeddings = model.embedTokens(tokens)
        let mask = createAttentionMask(
            h: embeddings, cache: cache, windowSize: configuration.slidingWindowSize)
        return layer(embeddings, hidden: previousHidden, mask: mask, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { layer in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindowSize,
                usesSlidingWindow: layer.isSlidingWindow)
        }
    }
}

extension SharpAIMimoV2Model: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
