// SharpAI fork overlay: nemotron_h_omni model. Not part of upstream mlx-swift-lm.
//
// Composition over NemotronHModel, which already runs the latent MoE path (fc1/fc2_latent_proj).
// The text stack is stored under `language_model`. The vision tower is not implemented, so
// image and video input throw. Sound input has no tower either, so audio throws too.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public final class SharpAINemotronHOmniModel: Module, VLMModel {

    public let configuration: SharpAINemotronHOmniConfiguration

    @ModuleInfo(key: "language_model") public var languageModel: NemotronHModel

    public init(_ configuration: SharpAINemotronHOmniConfiguration) {
        self.configuration = configuration
        self._languageModel.wrappedValue = NemotronHModel(configuration.llmConfig)
        super.init()
    }

    public var loraLayers: [Module] { languageModel.loraLayers }

    public var toolCallFormat: ToolCallFormat? { languageModel.toolCallFormat }

    public var reasoningConfig: ReasoningConfig? { languageModel.reasoningConfig }

    public var excludedCheckpointNamespaces: [String] {
        // UNVERIFIED: tower namespaces, skipped by the loader.
        ["mtp", "vision_model", "vision_tower", "mlp1", "sound_encoder", "sound_projection"]
    }

    public var castPredicate: ((String) -> Bool)? { languageModel.castPredicate }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw SharpAINemotronHOmniError.visionTowerNotImplemented
        }
        return .tokens(input.text)
    }

    /// Keeps text weights under `language_model.` and drops tower weights.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let text = try SharpAINemotronHOmniSanitizer.textWeights(weights)

        var sanitized = [String: MLXArray]()
        for (key, value) in languageModel.sanitize(weights: text) {
            sanitized[SharpAINemotronHOmniSanitizer.languagePrefix + key] = value
        }
        return sanitized
    }
}
