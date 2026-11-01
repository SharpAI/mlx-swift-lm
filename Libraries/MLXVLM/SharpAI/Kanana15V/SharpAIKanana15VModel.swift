// SharpAI fork overlay: kanana-1.5-v model. Not part of upstream mlx-swift-lm.
//
// Text path: the Llama tower (`LlamaForCausalLM`) from MLXLLM, used unchanged.
// Vision path: the Qwen2-VL style encoder in SharpAIKanana15VVision.swift. The C-Abstractor
// projector (dynamic-c-abs, RegStage) is not built, so image inputs throw
// `SharpAIKanana15VError.projectorNotImplemented`.
// UNVERIFIED: the `vision_model` and `language_model` checkpoint prefixes.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Kanana-1.5-V model. Created by ``SharpAIKanana15VOverlay``.
public class SharpAIKanana15VModel: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_model") private var visionModel: SharpAIKanana15VVisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LlamaModel

    public let configuration: SharpAIKanana15VConfiguration

    public var vocabularySize: Int { languageModel.vocabularySize }

    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ configuration: SharpAIKanana15VConfiguration) {
        self.configuration = configuration
        _visionModel.wrappedValue = SharpAIKanana15VVisionModel(configuration.visionConfiguration)
        _languageModel.wrappedValue = LlamaModel(configuration.textConfiguration)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil else {
            throw SharpAIKanana15VError.projectorNotImplemented
        }

        let tokens = input.text.tokens
        let logits = languageModel(tokens, cache: cache)
        let total = tokens.dim(-1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIKanana15VSanitizer.sanitize(weights, configuration: configuration)
    }
}
