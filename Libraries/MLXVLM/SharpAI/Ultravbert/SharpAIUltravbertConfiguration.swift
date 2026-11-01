// SharpAI fork overlay: UltraVBERT configuration. Not part of upstream mlx-swift-lm.

import Foundation

/// Config for `model_type` "ultravbert" (`ColUltraVBertForRetrieval`).
///
/// Only the keys needed to identify the checkpoint are decoded. The encoder layout is not in
/// the repo or the spec, so no architecture fields are read yet.
/// UNVERIFIED: `embedding_dim` is the key name used by the sibling ModernVBERT overlay. It is
/// not checked against a real UltraVBERT config. The dimension has no default here.
public struct SharpAIUltravbertConfiguration: Decodable, Sendable {

    public let modelType: String
    public let architectures: [String]
    /// Output width per token vector. Nil when the config does not state it.
    public let embeddingDimensions: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case embeddingDimensions = "embedding_dim"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "ultravbert"
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        embeddingDimensions = try container.decodeIfPresent(Int.self, forKey: .embeddingDimensions)
    }
}
