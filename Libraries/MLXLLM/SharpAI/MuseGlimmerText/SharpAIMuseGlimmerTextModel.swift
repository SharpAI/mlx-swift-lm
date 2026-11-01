// SharpAI fork overlay: muse_glimmer_text (Muse-Glimmer text-only) model. Not part of upstream
// mlx-swift-lm.
//
// Text-only path of Libraries/MLXVLM/Models/MuseGlimmer.swift. Tool calls use the ATEM format and
// reasoning uses the same delimiters as the VLM.
//
// UNVERIFIED: no text-only checkpoint is on disk, so the `model.*` / `lm_head.*` key layout is
// inferred from the VLM sanitize and the HF layout. `tie_word_embeddings` is not handled: the
// lm_head is always loaded. The 30B text_config has it false.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIMuseGlimmerTextModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]

    public let config: SharpAIMuseGlimmerTextConfiguration

    @ModuleInfo var model: SharpAIMuseGlimmerTextInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public var toolCallFormat: ToolCallFormat? { .atem }

    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
            promptStrategy: .none, isSpecialToken: true)
    }

    public init(_ configuration: SharpAIMuseGlimmerTextConfiguration) {
        self.config = configuration
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self._model.wrappedValue = SharpAIMuseGlimmerTextInner(configuration)
        self._lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }

    /// Sliding layers get a rotating cache bounded by the window. Full layers are unbounded.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try config.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: config.slidingWindow,
                usesSlidingWindow: layerType == "sliding_attention")
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let optionalCache = cache?.map { $0 as KVCache? }
        var logits = lmHead(model(inputs, cache: optionalCache)) * config.outputMultiplier
        let cap = config.finalLogitSoftcapping
        if cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    /// Key renaming and vision-key removal. See `SharpAIMuseGlimmerTextSanitize`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIMuseGlimmerTextSanitize.sanitize(weights)
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
