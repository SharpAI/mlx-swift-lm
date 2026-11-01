// SharpAI fork overlay: opencua config. Not part of upstream mlx-swift-lm.
//
// OpenCUA nests its text fields under `text_config`. Qwen25VLConfiguration reads
// them from the top level, so this file rewrites config.json into that flat layout.

import Foundation
import MLXLMCommon

public enum SharpAIOpencuaError: Error, Equatable {
    case invalidConfiguration(String)
}

public struct SharpAIOpencuaConfiguration: Decodable, Sendable {

    public let modelType: String
    public let textConfiguration: Qwen25VLConfiguration.TextConfiguration
    public let visionConfiguration: Qwen25VLConfiguration.VisionConfiguration
    public let mediaPlaceholderTokenId: Int

    /// Verified against xlangai/OpenCUA-7B config.json, which sets media_placeholder_token_id to 151664.
    static let defaultMediaPlaceholderTokenId = 151664

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case mediaPlaceholderTokenId = "media_placeholder_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.textConfiguration = try container.decode(
            Qwen25VLConfiguration.TextConfiguration.self, forKey: .textConfiguration)
        self.visionConfiguration = try container.decode(
            Qwen25VLConfiguration.VisionConfiguration.self, forKey: .visionConfiguration)
        self.mediaPlaceholderTokenId =
            try container.decodeIfPresent(Int.self, forKey: .mediaPlaceholderTokenId)
            ?? Self.defaultMediaPlaceholderTokenId
    }

    /// Rewrites the raw config.json bytes into the flat layout and decodes `Qwen25VLConfiguration`.
    public func qwen25VLConfiguration(from data: Data) throws -> Qwen25VLConfiguration {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SharpAIOpencuaError.invalidConfiguration("config.json is not a JSON object")
        }
        let text = (root["text_config"] as? [String: Any]) ?? [:]

        // Top-level keys first, then the text fields overlay them, as in the flat layout.
        var flat = root.filter { $0.key != "text_config" }
        for (key, value) in text {
            flat[key] = value
        }
        flat["model_type"] = text["model_type"] ?? "qwen2"

        // Token ids Qwen25VLConfiguration requires but OpenCUA may not declare.
        // UNVERIFIED: the vision_start/vision_end/vision_token defaults below are Qwen2.5-VL ids.
        // The OpenCUA tokenizer (tokenization_opencua.py) defines no <|vision_start|>,
        // <|vision_end|> or <|image_pad|> token, so these values are likely wrong for OpenCUA.
        // They are harmless today: Qwen25VL.getRopeIndex takes visionStartTokenId but never reads
        // it, and the merge path uses only image/video token ids. Checked against:
        // https://huggingface.co/xlangai/OpenCUA-7B/raw/main/tokenization_opencua.py
        // video_token_id -1 never matches: OpenCUA has no video path.
        Self.fillMissing(&flat, "image_token_id", mediaPlaceholderTokenId)
        Self.fillMissing(&flat, "video_token_id", -1)
        Self.fillMissing(&flat, "vision_start_token_id", 151652)
        Self.fillMissing(&flat, "vision_end_token_id", 151653)
        Self.fillMissing(&flat, "vision_token_id", 151654)
        Self.fillMissing(&flat, "sliding_window", 32768)
        Self.fillMissing(&flat, "use_sliding_window", false)
        Self.fillMissing(&flat, "max_window_layers", textConfiguration.hiddenLayers)

        // The real OpenCUA text config has no rope_scaling (plain Qwen2 RoPE in
        // modeling_opencua.py). Verified against:
        // https://huggingface.co/xlangai/OpenCUA-7B/raw/main/config.json
        // Qwen25VL's attention requires an mrope_section, so supply one.
        // With the same position on all three axes, M-RoPE reduces to plain RoPE, so text-only
        // output is unchanged. Image positions still follow Qwen25VL's 2D grid (see the KNOWN GAP
        // in SharpAIOpencuaForwardTests). The split is Qwen2.5-VL's [16, 24, 24] for head_dim 128,
        // scaled to head_dim.
        if flat["rope_scaling"] == nil || flat["rope_scaling"] is NSNull {
            let headDim = textConfiguration.hiddenSize / max(textConfiguration.attentionHeads, 1)
            let half = headDim / 2
            let first = half / 4
            let second = (half - first) / 2
            let rope: [String: Any] = [
                "type": "mrope",
                "mrope_section": [first, second, half - first - second],
            ]
            flat["rope_scaling"] = rope
        }

        let flatData = try JSONSerialization.data(withJSONObject: flat)
        return try JSONDecoder.json5().decode(Qwen25VLConfiguration.self, from: flatData)
    }

    private static func fillMissing(_ dictionary: inout [String: Any], _ key: String, _ value: Any)
    {
        if dictionary[key] == nil {
            dictionary[key] = value
        }
    }
}

extension SharpAIOpencuaConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard modelType == "opencua" else {
            throw SharpAIOpencuaError.invalidConfiguration(
                "model_type must be opencua, got \(modelType)")
        }
        // A missing rope_scaling is valid: the real OpenCUA checkpoint uses plain RoPE and
        // qwen25VLConfiguration(from:) supplies the section. A present one must be well formed.
        if let ropeScaling = textConfiguration.ropeScaling {
            try validateMROPESection(
                ropeScaling,
                context: "SharpAIOpencuaConfiguration.text_config.rope_scaling")
        }
    }
}
