// SharpAI fork overlay: NemotronH_Nano_Omni_Reasoning_V3 model. Not part of upstream mlx-swift-lm.
//
// Composition over NemotronHModel: the text stack is stored under `language_model`, so weight
// keys keep the `language_model.` prefix. Checkpoint prefixes checked against
// https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/model.safetensors.index.json
// (language_model.backbone.*, language_model.lm_head.*, vision_model.radio_model.*, mlp1.{0,1,3}.*,
// sound_encoder.encoder.*, sound_projection.*).
//
// Built here: the RADIO ViT tower (reused from the NemotronH Nano VL v2 port, same checkpoint key
// layout), the mlp1 pixel-shuffle projector, and the sound projection.
// Not built: the Parakeet sound encoder (`sound_encoder.*`), the video embedder (`video_embedder`),
// and the image and audio processors. Image features come from `imageFeatures(_:)`; the text
// `callAsFunction` path does not yet merge them into the prompt embeddings.
//
// UNVERIFIED: the RADIO tower reuses the C-RADIOv2-H block and patch-generator code from the
// NemotronH Nano VL v2 port. The checkpoint key names seen in the index match that layout, but the
// C-RADIOv4-H block code was not fetched, and the cls_token and register key names fall past the
// 100k characters read from model.safetensors.index.json. The tower is built with
// vit_huge_patch16_224 depth 32, 16 heads, and cls token count from cls_token_per_teacher.

import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public final class SharpAINemotronhNanoOmniReasoningV3Model: Module, LanguageModel {

    private static let languagePrefix = "language_model."
    private static let textKeyPrefixes = ["backbone.", "lm_head."]
    /// Video-only patch embedder. Video input is not ported, so these weights are dropped.
    private static let droppedSubstrings = ["video_embedder"]
    /// RADIO buffers with no module in the built tower (the processor normalizes pixels, and the
    /// summary token layout comes from the configuration).
    private static let droppedSuffixes = [
        "input_conditioner.norm_mean", "input_conditioner.norm_std", "radio_model.summary_idxs",
    ]

    public let configuration: SharpAINemotronhNanoOmniReasoningV3Configuration

    @ModuleInfo(key: "language_model") public var languageModel: NemotronHModel
    @ModuleInfo(key: "vision_model") public var visionModel: SharpAINemotronhNanoVlV2VisionModel?
    @ModuleInfo(key: "mlp1") public var mlp1: SharpAINemotronhNanoVlV2Projector?
    @ModuleInfo(key: "sound_projection")
    public var soundProjection: SharpAINemotronhNanoOmniReasoningV3SoundProjection?

    public init(_ configuration: SharpAINemotronhNanoOmniReasoningV3Configuration) {
        self.configuration = configuration
        let llmHidden = configuration.llmConfig.hiddenSize
        self._languageModel.wrappedValue = NemotronHModel(configuration.llmConfig)
        self._visionModel.wrappedValue = configuration.visionTower.map {
            SharpAINemotronhNanoVlV2VisionModel($0)
        }
        // mlp1 input is vit_hidden_size * (1 / downsample_ratio)^2, which is 4x at ratio 0.5.
        self._mlp1.wrappedValue = configuration.visionTower.map { _ in
            SharpAINemotronhNanoVlV2Projector(
                inputSize: configuration.vitHiddenSize * 4,
                hiddenSize: configuration.projectorHiddenSize,
                outputSize: llmHidden)
        }
        self._soundProjection.wrappedValue = configuration.soundProjection.map { shape in
            SharpAINemotronhNanoOmniReasoningV3SoundProjection(
                inputSize: shape.inputSize,
                hiddenSize: shape.hiddenSize,
                outputSize: llmHidden,
                bias: shape.bias)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.multimodalInputNotSupported
        }
        return try languageModel.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public var toolCallFormat: ToolCallFormat? { languageModel.toolCallFormat }

    public var reasoningConfig: ReasoningConfig? { languageModel.reasoningConfig }

    public var excludedCheckpointNamespaces: [String] {
        // Towers the loader must skip: the Parakeet encoder is not built, and mtp and
        // vision_tower are not in the released index (defensive excludes).
        ["mtp", "vision_tower", "sound_encoder"]
    }

    public var castPredicate: ((String) -> Bool)? { languageModel.castPredicate }

    /// NCHW pixels, already mean/std normalized, to projected image features (N, tokens, llm_hidden).
    /// Tokens are (H / patch / 2) * (W / patch / 2), following the reference extract_feature.
    public func imageFeatures(_ pixels: MLXArray) throws -> MLXArray {
        guard let tower = configuration.visionTower, let visionModel, let mlp1 else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.visionTowerUnavailable
        }
        let gridH = pixels.dim(2) / tower.patchSize
        let gridW = pixels.dim(3) / tower.patchSize
        guard gridH % 2 == 0, gridW % 2 == 0 else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.oddPatchGrid(gridH, gridW)
        }
        let features = visionModel(pixels)
        let grid = features.reshaped(features.dim(0), gridH, gridW, features.dim(2))
        return mlp1(sharpAINemotronhNanoVlV2PixelShuffle(grid))
    }

    /// `mlp1.0/1/3` (nn.Sequential indices in the checkpoint) map to the projector's `norm`, `fc1`, `fc2`.
    static func projectorName(_ key: String) -> String {
        let renames = [
            "mlp1.0.": "mlp1.norm.",
            "mlp1.1.": "mlp1.fc1.",
            "mlp1.3.": "mlp1.fc2.",
        ]
        for (from, to) in renames where key.hasPrefix(from) {
            return to + String(key.dropFirst(from.count))
        }
        return key
    }

    /// Keeps text weights under `language_model.`, keeps the built tower and projector weights as
    /// they are, and drops the rest (unbuilt towers, MTP, video embedder).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var keptPrefixes = [String]()
        if visionModel != nil {
            keptPrefixes += ["vision_model.", "mlp1."]
        }
        if soundProjection != nil {
            keptPrefixes.append("sound_projection.")
        }

        var text = [String: MLXArray]()
        var kept = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix(Self.languagePrefix) {
                text[String(key.dropFirst(Self.languagePrefix.count))] = value
            } else if Self.textKeyPrefixes.contains(where: { key.hasPrefix($0) }) {
                text[key] = value
            } else if keptPrefixes.contains(where: { key.hasPrefix($0) }) {
                if Self.droppedSubstrings.contains(where: { key.contains($0) })
                    || Self.droppedSuffixes.contains(where: { key.hasSuffix($0) })
                {
                    continue
                }
                kept[Self.projectorName(key)] = value
            }
        }

        var sanitized = kept
        for (key, value) in languageModel.sanitize(weights: text) {
            sanitized[Self.languagePrefix + key] = value
        }
        return sanitized
    }
}
