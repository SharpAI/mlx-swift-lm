//
//  SharpAIHyV4Model.swift
//  MLXLLM
//
//  Text model for hy_v4 (HYV4ForCausalLM). Port of HYV4Model and HYV4ForCausalLM in
//  transformers modular_hy_v4.py. Each decoder layer runs MLA attention (with the
//  DSA indexer on full layers), then a dense or sparse MoE block. Both are wrapped in
//  iHC stream mixing.
//
//  Not implemented: the multi-token-prediction layers (num_nextn_predict_layers).
//  The reference ignores `model.mtp_layers.*` on load, and sanitize drops the layers
//  at index >= num_hidden_layers, so MTP weights are not loaded.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIHyV4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIHyV4Attention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: SharpAIHyV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: SharpAIHyV4HyperConnection

    init(_ config: SharpAIHyV4Configuration, layerIndex: Int) {
        let isSparse = config.resolvedMlpLayerTypes[layerIndex] == "sparse"
        let hasIndexer = config.resolvedIndexerTypes[layerIndex] == "full"

        _attention.wrappedValue = SharpAIHyV4Attention(config, hasIndexer: hasIndexer)
        if isSparse {
            self.mlp = SharpAIHyV4SparseMoE(config)
        } else {
            self.mlp = SharpAIHyV4MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attnHC.wrappedValue = SharpAIHyV4HyperConnection(config)
        _ffnHC.wrappedValue = SharpAIHyV4HyperConnection(config)
    }

    /// `streams` is [B, L, N, D]. Returns the updated streams and the DSA selection.
    func callAsFunction(
        _ streams: MLXArray, causal: MLXArray, sparse: MLXArray?, cache: KVCache?
    ) -> (streams: MLXArray, selection: MLXArray?) {
        let dtype = streams.dtype

        let (attnPost, attnInput) = attnHC(streams)
        let attn = attention(
            inputLayerNorm(attnInput), causal: causal, sparse: sparse, cache: cache)
        let afterAttn = (attnPost.expandedDimensions(axis: -1)
            * attn.output.asType(.float32).expandedDimensions(axis: -2)
            + streams.asType(.float32)).asType(dtype)

        let (ffnPost, ffnInput) = ffnHC(afterAttn)
        let mlpOut = mlp(postAttentionLayerNorm(ffnInput))
        let out = (ffnPost.expandedDimensions(axis: -1)
            * mlpOut.asType(.float32).expandedDimensions(axis: -2)
            + afterAttn.asType(.float32)).asType(dtype)

        return (out, attn.selection)
    }
}

class SharpAIHyV4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIHyV4DecoderLayer]
    let norm: RMSNorm
    @ModuleInfo(key: "hc_head") var hcHead: SharpAIHyV4HyperHead
    let hcMult: Int

    init(_ config: SharpAIHyV4Configuration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIHyV4DecoderLayer(config, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _hcHead.wrappedValue = SharpAIHyV4HyperHead(config)
        self.hcMult = config.resolvedHcMult
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let embedded = embedTokens(inputs)  // [B, L, D]
        let L = embedded.dim(1)

        // Causal mask for the queries at positions offset..<offset+L against all T keys.
        let offset = cache?.first?.offset ?? 0
        let queryPositions = MLXArray((offset ..< offset + L).map { Int32($0) })
            .expandedDimensions(axis: 1)
        let keyPositions = MLXArray((0 ..< offset + L).map { Int32($0) })
            .expandedDimensions(axis: 0)
        let causal = queryPositions .>= keyPositions  // [L, T]

        var h = repeated(embedded.expandedDimensions(axis: 2), count: hcMult, axis: 2)
        var selection: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            (h, selection) = layer(h, causal: causal, sparse: selection, cache: cache?[i])
        }

        return norm(hcHead(h))
    }
}

class SharpAIHyV4Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    /// MLA caches one latent head per layer.
    let kvHeads: [Int]

    let model: SharpAIHyV4ModelInner
    let configuration: SharpAIHyV4Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIHyV4Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in 1 }
        self.model = SharpAIHyV4ModelInner(config)

        if !(config.tieWordEmbeddings ?? false) {
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

    /// Drops rotary buffers and MTP layers, then maps the routed experts to `switch_mlp`.
    /// Packed experts (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`) are split and
    /// renamed. Per-expert keys (`mlp.experts.<e>.gate_proj.weight`) are stacked.
    /// UNVERIFIED against a real checkpoint index.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = SharpAIHyV4Sanitizer.sanitize(
            filterLMHeadWeights(
                from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings ?? false),
            configuration: configuration)

        for layer in 0 ..< configuration.hiddenLayers
        where configuration.resolvedMlpLayerTypes[layer] == "sparse" {
            let prefix = "model.layers.\(layer).mlp"

            if let gateUp = sanitized.removeValue(forKey: "\(prefix).experts.gate_up_proj") {
                let halves = split(
                    gateUp, indices: [configuration.moeIntermediateSize], axis: 1)
                sanitized["\(prefix).switch_mlp.gate_proj.weight"] = halves[0]
                sanitized["\(prefix).switch_mlp.up_proj.weight"] = halves[1]
            }
            if let down = sanitized.removeValue(forKey: "\(prefix).experts.down_proj") {
                sanitized["\(prefix).switch_mlp.down_proj.weight"] = down
            }

            for name in ["gate_proj", "up_proj", "down_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).experts.0.\(name).\(kind)"] != nil else { continue }
                    let toJoin = (0 ..< configuration.numRoutedExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(name).\(kind)")
                    }
                    sanitized["\(prefix).switch_mlp.\(name).\(kind)"] = MLX.stacked(toJoin)
                }
            }
        }

        return sanitized
    }
}

extension SharpAIHyV4Model: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
