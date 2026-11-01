// SharpAI fork overlay: domynedge (Domyn-Edge-Base) configuration. Not part of upstream mlx-swift-lm.
//
// Field names come from scripts/conformance/offline/fixtures/domynedge/domyn__Domyn-Edge-Base.json.
// UNVERIFIED: the Hub modeling code (auto_map -> modeling_domynedge.py) is not on this machine.
// Defaults for keys the fixture lacks are guesses and are marked where they matter.

import Foundation
import MLXLMCommon

public enum SharpAIDomynedgeError: Error, Equatable {
    case invalidConfiguration(String)
}

/// Per-layer attention settings resolved from the layer arrays in the config.
public struct SharpAIDomynedgeLayerSpec: Equatable, Sendable {
    public let headDim: Int
    /// `false` for `"nope"` layers, which skip RoPE.
    public let usesRope: Bool
    /// Window in tokens, or `nil` for full attention.
    public let slidingWindow: Int?

    public init(headDim: Int, usesRope: Bool, slidingWindow: Int?) {
        self.headDim = headDim
        self.usesRope = usesRope
        self.slidingWindow = slidingWindow
    }
}

public struct SharpAIDomynedgeConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    /// Head size for layers when `head_sizes` is absent.
    public let headDim: Int
    /// Per-layer head size (`head_sizes`).
    public let headSizes: [Int]?
    /// Per-layer `"rope"` or `"nope"` (`position_layers`).
    public let positionLayers: [String]?
    /// Per-layer `[left, right]` window (`window_sizes`). A left value of `-1` means full attention.
    /// UNVERIFIED: the meaning of the right value (always `0` in the fixture) is not confirmed.
    public let windowSizes: [[Int]]?
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let hiddenAct: String
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case headSizes = "head_sizes"
        case positionLayers = "position_layers"
        case windowSizes = "window_sizes"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try container.decode(Int.self, forKey: .hiddenSize)
        let heads = try container.decode(Int.self, forKey: .attentionHeads)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "domynedge"
        hiddenSize = hidden
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = heads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (heads > 0 ? hidden / heads : 0)
        headSizes = try container.decodeIfPresent([Int].self, forKey: .headSizes)
        positionLayers = try container.decodeIfPresent([String].self, forKey: .positionLayers)
        windowSizes = try container.decodeIfPresent([[Int]].self, forKey: .windowSizes)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
    }

    public func validateModelConfiguration() throws {
        guard hiddenSize > 0, hiddenLayers > 0, intermediateSize > 0, vocabularySize > 0,
            attentionHeads > 0, kvHeads > 0, headDim > 0
        else {
            throw SharpAIDomynedgeError.invalidConfiguration(
                "sizes, head counts and vocabulary must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw SharpAIDomynedgeError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard hiddenAct == "silu" else {
            throw SharpAIDomynedgeError.invalidConfiguration(
                "unsupported hidden_act \(hiddenAct)")
        }
        if let headSizes {
            guard headSizes.count == hiddenLayers, headSizes.allSatisfy({ $0 > 0 }) else {
                throw SharpAIDomynedgeError.invalidConfiguration(
                    "head_sizes must have one positive entry per layer")
            }
        }
        if let positionLayers {
            guard positionLayers.count == hiddenLayers,
                positionLayers.allSatisfy({ $0 == "rope" || $0 == "nope" })
            else {
                throw SharpAIDomynedgeError.invalidConfiguration(
                    "position_layers must hold \"rope\" or \"nope\" for each layer")
            }
        }
        if let windowSizes {
            guard windowSizes.count == hiddenLayers,
                windowSizes.allSatisfy({ $0.count == 2 && $0[0] >= -1 && $0[0] != 0 })
            else {
                throw SharpAIDomynedgeError.invalidConfiguration(
                    "window_sizes must hold a [left, right] pair per layer")
            }
        }
    }

    /// Resolves the per-layer settings. Missing arrays fall back to the global values.
    /// Call `validateModelConfiguration()` first; out-of-range entries use the fallback.
    public func layerSpecs() -> [SharpAIDomynedgeLayerSpec] {
        (0 ..< hiddenLayers).map { index in
            let dim = Self.entry(headSizes, index) ?? headDim
            let usesRope = Self.entry(positionLayers, index).map { $0 == "rope" } ?? true
            var window: Int? = nil
            if let pair = Self.entry(windowSizes, index), let left = pair.first, left > 0 {
                window = left
            }
            return SharpAIDomynedgeLayerSpec(
                headDim: dim, usesRope: usesRope, slidingWindow: window)
        }
    }

    private static func entry<T>(_ values: [T]?, _ index: Int) -> T? {
        guard let values, index < values.count else { return nil }
        return values[index]
    }
}
