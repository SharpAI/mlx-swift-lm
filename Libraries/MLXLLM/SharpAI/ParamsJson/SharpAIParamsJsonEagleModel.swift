// SharpAI fork overlay: EAGLE drafter for EagleMistralLarge3ForCausalLM. Not part of upstream.
//
// A drafter does not run alone. Each step takes the target model's last hidden state, fuses it
// with the token embedding, runs the decoder layers, and returns the drafter's own logits.
// The fusion is a single linear over [embedding, hidden] (eagle_proj). The order matches vLLM's
// EagleMistralLarge3Model.forward: fc(cat([inputs_embeds, hidden_states], -1)), with no norm on
// the concatenated input. Tested in SharpAIParamsJsonModelTests. Real-checkpoint parity is still
// a known gap.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIParamsJsonEagleModel: Module {
    public let config: SharpAIParamsJsonConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "eagle_proj") var eagleProj: Linear
    let layers: [SharpAIParamsJsonDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: SharpAIParamsJsonConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dim)
        _eagleProj.wrappedValue = Linear(2 * config.dim, config.dim, bias: false)
        layers = (0..<config.nLayers).map { _ in SharpAIParamsJsonDecoderLayer(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        if !config.tiedEmbeddings {
            _lmHead.wrappedValue = Linear(config.dim, config.vocabSize, bias: false)
        }
        super.init()
    }

    /// One draft step.
    /// - Parameters:
    ///   - inputIDs: token ids, shape [B, L].
    ///   - hiddenStates: target hidden states for the same positions, shape [B, L, dim].
    ///   - cache: optional per-layer KV caches, one per decoder layer.
    /// - Returns: the normed drafter hidden state and the logits, both over the same [B, L] grid.
    public func draft(
        inputIDs: MLXArray, hiddenStates: MLXArray, cache: [KVCache]?
    ) -> (hidden: MLXArray, logits: MLXArray) {
        let embeds = embedTokens(inputIDs)
        var h = eagleProj(
            concatenated([embeds, hiddenStates.asType(embeds.dtype)], axis: -1))

        let mask = createAttentionMask(h: h, cache: cache?.first)
        let offset = cache?.first?.offset ?? 0
        let attnScale = SharpAIParamsJsonLlama4Scaling.queryScale(
            config: config, offset: offset, length: h.dim(1), dtype: h.dtype)

        for (i, layer) in layers.enumerated() {
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
        }

        let hidden = norm(h)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = embedTokens.asLinear(hidden)
        }
        return (hidden, logits)
    }

    /// Maps Mistral native and HF-style weight names to this model's keys. See SharpAIParamsJsonSanitize.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIParamsJsonSanitize.sanitize(weights, config: config)
    }
}
