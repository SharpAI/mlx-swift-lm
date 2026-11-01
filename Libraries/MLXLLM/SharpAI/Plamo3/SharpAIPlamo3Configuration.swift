// SharpAI fork overlay: plamo3 configuration. Not part of upstream mlx-swift-lm.
//
// Key names for the layer layout and RoPE come from the reference sources: vLLM
// vllm/model_executor/models/plamo3.py (Plamo3Config: layer_types, interleaved_sliding_window,
// rope_local_theta, sliding_window_pattern) and mlx-lm mlx_lm/models/plamo3.py (rope_theta default
// 1_000_000, rope_local_theta default 10_000, sliding_window_pattern default 8).
// The HF repo is gated, so no config.json is on this machine: the real checkpoint config is UNVERIFIED.
// The architecture string comes from the model spec. Check against the real config.json.

import Foundation
import MLXLMCommon

public enum SharpAIPlamo3Error: LocalizedError, Equatable {
    case invalidConfiguration(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Plamo3 invalid configuration: \(detail)"
        case .unsupported(let detail):
            return "Plamo3 unsupported: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: plamo3`.
public struct SharpAIPlamo3Configuration: Decodable, Sendable {
    public static let expectedModelType = "plamo3"
    public static let expectedArchitecture = "Plamo3ForCausalLM"
    public static let fullAttention = "full_attention"
    public static let slidingAttention = "sliding_attention"

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let intermediateSize: Int?
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int?
    public let vocabularySize: Int
    public let rmsNormEps: Float
    /// Global RoPE base, used by full-attention layers.
    public let ropeTheta: Float
    /// RoPE base for sliding-window layers (vLLM `rope_local_theta`).
    public let ropeLocalTheta: Float
    public let maxPositionEmbeddings: Int?
    public let slidingWindow: Int?
    /// Explicit per-layer attention kinds (`layer_types`), when the checkpoint config carries them.
    public let layerTypes: [String]?
    /// Every `pattern`-th layer (1-indexed) is full attention when `layerTypes` is absent.
    public let slidingWindowPattern: Int
    public let tieWordEmbeddings: Bool

    /// Attention head width. Falls back to hidden size divided by head count.
    public var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / max(attentionHeads, 1))
    }

    /// Per-layer attention kind. Explicit `layer_types` wins; otherwise layer `i` (0-indexed) is
    /// full attention when `(i + 1) % slidingWindowPattern == 0`, matching mlx-lm.
    public var resolvedLayerTypes: [String] {
        if let layerTypes {
            return layerTypes
        }
        return (0 ..< max(hiddenLayers, 0)).map { index in
            (index + 1) % max(slidingWindowPattern, 1) == 0
                ? Self.fullAttention : Self.slidingAttention
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeLocalTheta = "rope_local_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case slidingWindowPattern = "sliding_window_pattern"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads =
            try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeLocalTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalTheta) ?? 10_000
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 8
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    /// Checks the model type, the architecture name when present, and basic shape sanity.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIPlamo3Error.invalidConfiguration(
                "model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIPlamo3Error.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard hiddenSize > 0, hiddenLayers > 0, attentionHeads > 0, kvHeads > 0 else {
            throw SharpAIPlamo3Error.invalidConfiguration(
                "hidden_size, num_hidden_layers, num_attention_heads and num_key_value_heads must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw SharpAIPlamo3Error.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard slidingWindowPattern > 0 else {
            throw SharpAIPlamo3Error.invalidConfiguration(
                "sliding_window_pattern must be positive")
        }
        if let intermediateSize, intermediateSize <= 0 {
            throw SharpAIPlamo3Error.invalidConfiguration("intermediate_size must be positive")
        }
        if let layerTypes {
            guard layerTypes.count == hiddenLayers else {
                throw SharpAIPlamo3Error.invalidConfiguration(
                    "layer_types must have num_hidden_layers entries")
            }
            guard
                layerTypes.allSatisfy({
                    $0 == Self.fullAttention || $0 == Self.slidingAttention
                })
            else {
                throw SharpAIPlamo3Error.invalidConfiguration(
                    "layer_types entries must be full_attention or sliding_attention")
            }
        }
    }
}
