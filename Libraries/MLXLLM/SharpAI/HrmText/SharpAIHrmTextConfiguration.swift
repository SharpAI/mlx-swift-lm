// SharpAI fork overlay: hrm_text configuration. Not part of upstream mlx-swift-lm.
//
// Key names and defaults follow the transformers hrm_text implementation (HrmTextConfig,
// transformers v5.22.1, configuration_hrm_text.py) and the HRM-Text-1B fixture
// (scripts/conformance/offline/fixtures/hrm_text). H_cycles and L_cycles stay required, as the
// tests in this overlay expect.

import Foundation
import MLXLMCommon

public enum SharpAIHrmTextError: LocalizedError {
    /// Kept for API compatibility. The creator no longer throws this: the forward pass exists.
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return "Model type \(type) is registered, but its forward pass is not implemented."
        }
    }
}

public struct SharpAIHrmTextConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    /// Real number of transformer blocks inside each of the H and L stacks (HF `num_layers_per_stack`).
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let initializerRange: Float
    public let embeddingScale: Float
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let hiddenAct: String
    public let prefixLM: Bool
    public let hCycles: Int
    public let lCycles: Int
    public let lBpCycles: [Int]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case initializerRange = "initializer_range"
        case embeddingScale = "embedding_scale"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case hiddenAct = "hidden_act"
        case prefixLM = "prefix_lm"
        case hCycles = "H_cycles"
        case lCycles = "L_cycles"
        case lBpCycles = "L_bp_cycles"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hrm_text"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        let heads = try c.decode(Int.self, forKey: .attentionHeads)
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        let initRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        initializerRange = initRange
        embeddingScale =
            try c.decodeIfPresent(Float.self, forKey: .embeddingScale) ?? (1 / initRange)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        prefixLM = try c.decodeIfPresent(Bool.self, forKey: .prefixLM) ?? true
        hCycles = try c.decode(Int.self, forKey: .hCycles)
        lCycles = try c.decode(Int.self, forKey: .lCycles)
        lBpCycles = try c.decodeIfPresent([Int].self, forKey: .lBpCycles) ?? [2]
    }

    public func validateModelConfiguration() throws {
        let dims = [
            hiddenSize, hiddenLayers, attentionHeads, kvHeads, intermediateSize, vocabularySize,
            hCycles, lCycles, headDim,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "hrm_text: sizes and cycle counts must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "hrm_text: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard hiddenSize % attentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "hrm_text: hidden_size must be divisible by num_attention_heads")
        }
        guard hiddenAct == "silu" else {
            throw ModelFactoryError.invalidConfiguration(
                "hrm_text: only hidden_act \"silu\" is implemented, got \(hiddenAct)")
        }
    }

    /// Number of attention invocations per token: each of the H_cycles high cycles runs
    /// L_cycles low stack calls and one high stack call.
    public var stackCallsPerForward: Int { hCycles * (lCycles + 1) }

    /// Total KV cache slots. Each stack call owns `hiddenLayers` slots (HF `DynamicCache` layout).
    public var cacheSlotCount: Int { hiddenLayers * stackCallsPerForward }
}
