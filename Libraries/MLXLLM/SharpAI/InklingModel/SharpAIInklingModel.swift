// SharpAI fork overlay: inkling_model text model. Not part of upstream mlx-swift-lm.
//
// Text backbone of Thinking Machines Inkling, from transformers `InklingForCausalLM`:
// normed embedding, decoder layers, final RMSNorm, and an LM head that divides the hidden states
// by the muP width multiplier and drops the padded vocabulary rows.
// Incremental decoding recomputes the full sequence from a token-history cache (exact, O(n^2));
// per-layer KV-cache and short-conv state is not implemented.
// Vision, audio and MTP weights are dropped by the sanitizer.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIInklingTextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: SharpAIInklingRMSNorm
    let layers: [SharpAIInklingDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIInklingRMSNorm

    init(_ config: SharpAIInklingTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = SharpAIInklingRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIInklingDecoderLayer(config, layer: $0)
        }
        _norm.wrappedValue = SharpAIInklingRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        var h = embedNorm(embedTokens(inputs))
        for layer in layers {
            h = layer(h)
        }
        return norm(h)
    }
}

class SharpAIInklingModel: Module, LLMModel {
    let vocabularySize: Int
    let kvHeads: [Int]
    let configuration: SharpAIInklingTextConfiguration
    let model: SharpAIInklingTextModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAIInklingTextConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = config.slidingLayers.map { sliding in
            sliding ? config.swaKVHeads : config.kvHeads
        }
        self.model = SharpAIInklingTextModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Generation recomputes the full sequence from the recorded token ids, so the output is
        // exactly the full-sequence forward. Only the logits of the new tokens are returned.
        let newCount = inputs.dim(1)
        let record = cache?.compactMap { $0 as? SharpAIInklingTokenHistoryCache }.first
        var sequence = inputs
        if let record, let past = record.history {
            sequence = MLX.concatenated([past.asType(inputs.dtype), inputs], axis: 1)
        }
        var hidden = model(sequence)
        if sequence.dim(1) > newCount {
            hidden = hidden[0..., (sequence.dim(1) - newCount)..., 0...]
        }
        record?.append(inputs)
        hidden = hidden / configuration.logitsMupWidthMultiplier
        var logits = lmHead(hidden)
        if let unpadded = configuration.unpaddedVocabularySize,
            unpadded < logits.dim(-1)
        {
            logits = logits[.ellipsis, ..<unpadded]
        }
        return logits
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIInklingModelSanitizer.sanitize(weights)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        // One token-history entry. Per-layer KV and short-conv state is not implemented, so each
        // step recomputes the full sequence (see SharpAIInklingModelCache.swift).
        [SharpAIInklingTokenHistoryCache()]
    }

    /// Decoder layers that LoRA adapters attach to. Same as the other SharpAI overlays.
    var loraLayers: [Module] {
        model.layers
    }
}
