// SharpAI fork overlay: cosmos3_omni configuration. Not part of upstream mlx-swift-lm.
//
// The understanding path (text and vision) uses the existing Qwen3VLConfiguration.
// The generation experts (MoT, diffusion, action, sound) and the tokenizer blocks are not decoded.
// Verified against: https://huggingface.co/hf-tiny-v2/tiny-random-Cosmos3OmniForConditionalGeneration/raw/main/config.json
//   (architectures `Cosmos3OmniForConditionalGeneration`, model_type `cosmos3_omni`).
// The name `Cosmos3ForConditionalGeneration` in the decoding fixtures is not a real architecture name.
// Routing uses model_type only, so the fixture name is never checked.

import Foundation
import MLXLMCommon

public enum SharpAICosmos3OmniError: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)
    case generationWeightNotSupported(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(Cosmos3 generation experts and weight layout are unverified)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after cosmos3_omni sanitize: \(key)"
        case .generationWeightNotSupported(let key):
            return "cosmos3_omni generation weight is not supported yet: \(key)"
        }
    }
}

public struct SharpAICosmos3OmniConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let architectures: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "cosmos3_omni"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    }

    public func validateModelConfiguration() throws {
        guard modelType == "cosmos3_omni" else {
            throw ModelFactoryError.invalidConfiguration(
                "cosmos3_omni: unexpected model_type \(modelType)")
        }
    }

    /// Decodes the understanding path as a Qwen3-VL configuration.
    ///
    /// The top-level `text_config` and `vision_config` blocks are decoded with the Qwen3-VL layout.
    /// `text_config.model_type` is not checked. Verified against
    /// https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cosmos3_omni/configuration_cosmos3_omni.py
    /// (default `text_config` model_type `qwen3_vl_text`, `tie_word_embeddings` default False).
    public static func qwen3VLConfiguration(from data: Data) throws -> Qwen3VLConfiguration {
        try JSONDecoder.json5().decode(
            Qwen3VLConfiguration.self, from: try normalizedUnderstandingJSON(data))
    }

    /// Copies the Cosmos3 RoPE and tie settings into the keys that `Qwen3VLConfiguration` reads.
    ///
    /// The tiny reference config (URL in the file header) writes RoPE as
    /// `text_config.rope_parameters` (`rope_theta`, `mrope_interleaved`, `mrope_section`) and sets
    /// `tie_word_embeddings` at the top level. Existing `rope_theta`, `rope_scaling` and
    /// `tie_word_embeddings` keys in `text_config` are kept. Input that is not strict JSON (for
    /// example JSON5 comments) is returned unchanged and decoded as written.
    static func normalizedUnderstandingJSON(_ data: Data) throws -> Data {
        guard var root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            var text = root["text_config"] as? [String: Any]
        else {
            return data
        }

        if let parameters = text["rope_parameters"] as? [String: Any] {
            if text["rope_scaling"] == nil {
                text["rope_scaling"] = parameters
            }
            if text["rope_theta"] == nil, let theta = parameters["rope_theta"] {
                text["rope_theta"] = theta
            }
        }
        if text["tie_word_embeddings"] == nil {
            if let tie = root["tie_word_embeddings"] {
                text["tie_word_embeddings"] = tie
            } else {
                text["tie_word_embeddings"] = false
            }
        }

        root["text_config"] = text
        return try JSONSerialization.data(withJSONObject: root)
    }
}
