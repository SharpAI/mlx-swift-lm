// SharpAI fork overlay: axk1 (A.X K1) configuration. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the config keys are assumed to match the DeepSeek-V3 names (MoE gate with
// e_score_correction_bias, MLA attention). No axk1 config.json was available to check.

import Foundation
import MLXLMCommon

public struct SharpAIAxk1Configuration: Decodable, Sendable {
    public let modelType: String
    /// The DeepSeek-V3 field set, decoded from the same top-level JSON object.
    public let base: DeepseekV3Configuration

    private enum ModelTypeKey: String, CodingKey {
        case modelType = "model_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: ModelTypeKey.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "axk1"
        base = try DeepseekV3Configuration(from: decoder)
    }
}
