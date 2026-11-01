import Foundation
import MLXLMCommon

/// Beam model config. Keys follow the Hugging Face `config.json` names.
public struct SharpAIBeamConfiguration: Codable, Sendable {
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var tieWordEmbeddings: Bool
    public var slidingWindow: Int
    public var layerTypes: [String]
    public var mlpLayerTypes: [String]
    public var moeIntermediateSize: Int
    public var numExpertsPerTok: Int
    public var numExperts: Int
    public var normTopkProb: Bool
    public var nGroup: Int?
    public var topkGroup: Int?
    public var routedScalingFactor: Float
    public var finalLogitSoftcapping: Float?
    public var ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExpertsPerTok = "num_experts_per_tok"
        case numExperts = "num_experts"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        let heads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        let layers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "beam"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151936
        hiddenSize = hidden
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        hiddenLayers = layers
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / heads)
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4096
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "sliding_attention", count: layers)
        mlpLayerTypes =
            try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
            ?? Array(repeating: "sparse", count: layers)
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 768
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup)
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup)
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)

        // `rope_theta` may be null when `rope_parameters` carries the theta, as in the release config.
        // `rope_parameters` is decoded from its own key type: it has no stored property, so it
        // must stay out of `CodingKeys`, which the synthesized `Encodable` conformance uses.
        var theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        if theta == nil {
            let rope = try decoder.container(keyedBy: RopeParametersKey.self)
            if let entries = try? rope.decodeIfPresent(
                [String: SharpAIBeamRopeEntry?].self, forKey: .ropeParameters)
            {
                theta = entries.values.compactMap { $0?.theta }.first
            } else if let flat = try? rope.decodeIfPresent(
                SharpAIBeamRopeEntry.self, forKey: .ropeParameters)
            {
                theta = flat.theta
            }
        }
        ropeTheta = theta ?? 10000.0
    }
}

/// Key for the decode-only `rope_parameters` field, kept apart from `SharpAIBeamConfiguration.CodingKeys`.
private enum RopeParametersKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

/// One RoPE entry. Layer-keyed `rope_parameters` hold one of these per layer type, or null.
private struct SharpAIBeamRopeEntry: Decodable {
    let theta: Float?

    enum CodingKeys: String, CodingKey {
        case theta = "rope_theta"
    }
}

public struct SharpAIBeamError: Error, CustomStringConvertible {
    public let message: String

    public init(_ message: String) {
        self.message = message
    }

    public var description: String { message }
}

extension SharpAIBeamConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" })
        else {
            throw SharpAIBeamError(
                "layer_types needs one full_attention or sliding_attention entry per layer.")
        }
        guard mlpLayerTypes.count == hiddenLayers,
            mlpLayerTypes.allSatisfy({ $0 == "dense" || $0 == "sparse" })
        else {
            throw SharpAIBeamError("mlp_layer_types needs one dense or sparse entry per layer.")
        }
        guard hiddenAct == "silu" else {
            throw SharpAIBeamError("Unsupported hidden_act \(hiddenAct); Beam uses silu.")
        }
        guard !tieWordEmbeddings else {
            throw SharpAIBeamError("Beam requires tie_word_embeddings = false.")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIBeamError("num_attention_heads must be a multiple of num_key_value_heads.")
        }
        if numExperts > 0 {
            guard numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
                throw SharpAIBeamError("num_experts_per_tok must be between 1 and num_experts.")
            }
        }
        if let nGroup, let topkGroup {
            guard nGroup > 0, numExperts % nGroup == 0, topkGroup >= 1, topkGroup <= nGroup
            else {
                throw SharpAIBeamError("n_group must divide num_experts; topk_group must fit n_group.")
            }
        }
    }
}
