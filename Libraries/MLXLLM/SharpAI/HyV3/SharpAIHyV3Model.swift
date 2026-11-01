//
//  SharpAIHyV3Model.swift
//  MLXLLM
//
//  Text model for hy_v3 (HYV3ForCausalLM). Layers below first_k_dense_replace
//  use a dense MLP; the rest use the sparse MoE block.
//  The MTP (nextn) layer sits at index num_hidden_layers. It is built and its
//  weights are loaded. mtpLogits(nextTokens:hiddenStates:) runs one stateless MTP
//  step. Generation does not call it (no speculative decoding here).
//  Verified against: transformers modeling_hy_v3.py (decoder layer, attention and
//  MoE key names; no MTP there, its loader ignores layer 80) and vLLM
//  hy_v3_mtp.py (MTP modules, forward order, shared embedding and lm_head).
//  Not confirmed: the layer-80 key names in the checkpoint safetensors index.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIHyV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIHyV3Attention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    // MTP modules. Present only on the nextn layer at index num_hidden_layers.
    @ModuleInfo(key: "enorm") var enorm: RMSNorm?
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm?
    @ModuleInfo(key: "eh_proj") var ehProj: Linear?
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm?

    init(_ config: SharpAIHyV3Configuration, layerIndex: Int) {
        _attention.wrappedValue = SharpAIHyV3Attention(config)

        if layerIndex >= config.firstKDenseReplace {
            self.mlp = SharpAIHyV3SparseMoE(config)
        } else {
            self.mlp = SharpAIHyV3MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if layerIndex >= config.hiddenLayers {
            _enorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _hnorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _ehProj.wrappedValue = Linear(2 * config.hiddenSize, config.hiddenSize, bias: false)
            _finalLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }

    /// One MTP step through this layer. `embeddings` is the next-token embedding and
    /// `hidden` the main model's hidden state, both [B, L, H]. Returns the hidden state
    /// after final_layernorm. Order follows vLLM hy_v3_mtp.py: enorm and hnorm, concat
    /// (embedding first), eh_proj, decoder block, final_layernorm.
    func multiTokenPredict(
        embeddings: MLXArray, hidden: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let enorm, let hnorm, let ehProj, let finalLayerNorm else {
            preconditionFailure("multiTokenPredict called on a layer without MTP modules")
        }
        let fused = ehProj(MLX.concatenated([enorm(embeddings), hnorm(hidden)], axis: -1))
        let y = callAsFunction(fused, mask: mask, cache: nil)
        return finalLayerNorm(y)
    }
}

class SharpAIHyV3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    /// Main decoder layers first, then the MTP layers (index >= num_hidden_layers).
    let layers: [SharpAIHyV3DecoderLayer]
    let mainLayerCount: Int
    let norm: RMSNorm

    init(_ config: SharpAIHyV3Configuration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.mainLayerCount = config.hiddenLayers
        self.layers = (0 ..< config.hiddenLayers + config.numNextnPredictLayers).map {
            SharpAIHyV3DecoderLayer(config, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for i in 0 ..< mainLayerCount {
            h = layers[i](h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

class SharpAIHyV3Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]

    let model: SharpAIHyV3ModelInner
    let configuration: SharpAIHyV3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIHyV3Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIHyV3ModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return projectToVocabulary(out)
    }

    private func projectToVocabulary(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    /// One stateless MTP step: logits for the token after the next one.
    /// `nextTokens` [B, L] holds the token at position i+1, aligned with hidden state i.
    /// `hiddenStates` [B, L, H] is the main model's final hidden state. No KV cache is used.
    /// The MTP step reuses the main embedding and lm_head, as in vLLM hy_v3_mtp.py.
    func mtpLogits(nextTokens: MLXArray, hiddenStates: MLXArray) -> MLXArray {
        precondition(configuration.numNextnPredictLayers > 0, "hy_v3 config has no MTP layer")
        let layer = model.layers[configuration.hiddenLayers]
        let mask = createAttentionMask(h: hiddenStates, cache: nil as KVCache?)
        let hidden = layer.multiTokenPredict(
            embeddings: model.embedTokens(nextTokens), hidden: hiddenStates, mask: mask)
        return projectToVocabulary(hidden)
    }

    /// Drops rotary caches, and for the MTP layer the embed_tokens and shared_head copies
    /// (the MTP step reuses the main ones, as vLLM does). Keeps the MTP layer's own weights.
    /// Then stacks routed experts into `switch_mlp` for every layer, MTP included.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        let mainLayerCount = configuration.hiddenLayers
        let layerCount = mainLayerCount + configuration.numNextnPredictLayers
        sanitized = sanitized.filter {
            if $0.key.contains("rotary_emb.inv_freq") { return false }
            guard let index = SharpAIHyV3Model.layerIndex(of: $0.key) else { return true }
            guard index < layerCount else { return false }
            let isMTPLayer = index >= mainLayerCount
            let isSharedCopy =
                $0.key.hasPrefix("model.layers.\(index).embed_tokens.")
                || $0.key.hasPrefix("model.layers.\(index).shared_head.")
            return !(isMTPLayer && isSharedCopy)
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

extension SharpAIHyV3Model: LoRAModel {
    var loraLayers: [Module] {
        // Main layers only. The MTP layer is not part of the LoRA target set.
        Array(model.layers.prefix(configuration.hiddenLayers))
    }
}
