// SharpAI fork overlay: telechat3 configuration. Not part of upstream mlx-swift-lm.
//
// telechat3 reuses the Llama layout (see the alias note in SharpAITelechat3Registration.swift).
// LlamaConfiguration rejects rope types other than linear, dynamic and llama3, so this type
// decodes the Llama keys itself and keeps rope_scaling as-is (type "telechat3-yarn").
// UNVERIFIED: the key names and defaults follow the Llama layout. The checkpoint config.json is
// not in the repo. Check tie_word_embeddings, rms_norm_eps and the yarn fields against it.

import Foundation
import MLXLMCommon

/// Top-level `config.json` for `model_type: telechat3`.
public struct SharpAITelechat3Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "telechat3"
    public static let expectedArchitecture = "TeleChat3ForCausalLM"

    /// rope_scaling type of the telechat3 checkpoints (Tele-AI/TeleChat3-36B-Thinking config.json).
    static let yarnRopeType = "telechat3-yarn"
    /// Reference mscale slope 0.07 divided by the shared YaRN slope 0.1.
    static let mscaleSlopeRatio: Float = 0.7

    public let modelType: String?
    public let architectures: [String]
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int?
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Float
    public let ropeTraditional: Bool
    public let ropeScaling: [String: StringOrNumber]?
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
    }

    /// Attention head width. Falls back to hidden size divided by head count.
    public var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / max(attentionHeads, 1))
    }

    /// rope_scaling as the shared YaRN RoPE (`YarnRoPE`, RoPEUtils.swift) reads it.
    ///
    /// Reference (Tele-AI/TeleChat3-36B-Thinking, modeling_telechat3.py, `get_mscale`) uses the
    /// slope 0.07: `0.07 * mscale * ln(factor) + 1`. The shared `yarnGetMscale` hardcodes 0.1.
    /// Scaling `mscale` and `mscale_all_dim` by 0.7 makes the shared coefficient equal the
    /// reference one, because 0.1 * 0.7 = 0.07. The reference applies `mscale` and
    /// `mscale_all_dim` only when both are present and non-zero. Otherwise its factor is
    /// `get_mscale(factor)`, which is `mscale = 1`, so this maps to `mscale = 0.7` with
    /// `mscale_all_dim = 0`.
    var llamaRopeScaling: [String: StringOrNumber]? {
        guard var scaling = ropeScaling else { return nil }
        let ropeType: String? = {
            if let value = scaling["type"] ?? scaling["rope_type"], case .string(let s) = value {
                return s
            }
            return nil
        }()
        guard ropeType == Self.yarnRopeType else { return scaling }

        let mscale = scaling["mscale"]?.asFloat()
        let mscaleAllDim = scaling["mscale_all_dim"]?.asFloat()
        if let mscale, let mscaleAllDim, mscale != 0, mscaleAllDim != 0 {
            scaling["mscale"] = .float(mscale * Self.mscaleSlopeRatio)
            scaling["mscale_all_dim"] = .float(mscaleAllDim * Self.mscaleSlopeRatio)
        } else {
            scaling["mscale"] = .float(Self.mscaleSlopeRatio)
            scaling["mscale_all_dim"] = .float(0)
        }
        return scaling
    }

    /// The Llama configuration the model is built from.
    public var llamaConfiguration: LlamaConfiguration {
        LlamaConfiguration(
            hiddenSize: hiddenSize,
            hiddenLayers: hiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: attentionHeads,
            headDimensions: headDim,
            rmsNormEps: rmsNormEps,
            vocabularySize: vocabularySize,
            kvHeads: kvHeads,
            maxPositionEmbeddings: maxPositionEmbeddings,
            ropeTheta: ropeTheta,
            ropeTraditional: ropeTraditional,
            ropeScaling: llamaRopeScaling,
            tieWordEmbeddings: tieWordEmbeddings,
            attentionBias: attentionBias,
            mlpBias: mlpBias,
            modelType: modelType
        )
    }

    /// Checks the model type, the architecture name when present, shape sanity and the rope block.
    public func validateModelConfiguration() throws {
        if let modelType, modelType != Self.expectedModelType {
            throw ModelFactoryError.invalidConfiguration(
                "model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw ModelFactoryError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard hiddenSize > 0, hiddenLayers > 0, attentionHeads > 0, kvHeads > 0,
            vocabularySize > 0, intermediateSize > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, "
                    + "intermediate_size and vocab_size must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        try validateRoPEConfiguration(ropeScaling)
    }
}
