// SharpAI fork overlay: StepVL configuration for `model_type` "step_robotics". Not part of upstream mlx-swift-lm.

import Foundation
import MLXLMCommon

/// Config for StepVL (`StepVLForConditionalGeneration`).
///
/// Top level: `vision_config` (Perception Encoder), `text_config` (`step3_text`), and the projector
/// and placeholder fields. Text defaults copy `Step3TextConfig` in vLLM's
/// `transformers_utils/configs/step3_vl.py`. Vision defaults are UNVERIFIED: no vision config class
/// exists in the sources on this machine, so only the fields the forward pass needs are required.
public struct SharpAIStepRoboticsConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    /// `step3_text` fields. Single KV head, as in `Step3TextAttention`.
    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let intermediateSize: Int
        public let vocabularySize: Int
        public let headDim: Int
        /// Width of the `qkv_proj` query slice. 0 falls back to `headDim`.
        public let shareQDim: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeType: String
        public let moeIntermediateSize: Int
        public let moeExperts: Int
        public let moeTopK: Int
        public let normExpertWeight: Bool
        public let shareExpertDim: Int
        /// Layer indices that use the MoE block. The others use a dense MLP.
        public let moeLayers: [Int]

        public var resolvedQDim: Int { shareQDim > 0 ? shareQDim : headDim }

        /// Step3TextConfig class defaults.
        public init(
            hiddenSize: Int = 7168,
            hiddenLayers: Int = 61,
            attentionHeads: Int = 64,
            intermediateSize: Int = 18432,
            vocabularySize: Int = 128815,
            headDim: Int = 256,
            shareQDim: Int = 2048,
            rmsNormEps: Float = 1e-5,
            ropeTheta: Float = 500_000,
            ropeType: String = "default",
            moeIntermediateSize: Int = 5120,
            moeExperts: Int = 48,
            moeTopK: Int = 3,
            normExpertWeight: Bool = false,
            shareExpertDim: Int = 5120,
            moeLayers: [Int] = Array(4 ..< 60)
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.attentionHeads = attentionHeads
            self.intermediateSize = intermediateSize
            self.vocabularySize = vocabularySize
            self.headDim = headDim
            self.shareQDim = shareQDim
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.ropeType = ropeType
            self.moeIntermediateSize = moeIntermediateSize
            self.moeExperts = moeExperts
            self.moeTopK = moeTopK
            self.normExpertWeight = normExpertWeight
            self.shareExpertDim = shareExpertDim
            self.moeLayers = moeLayers
        }

        /// `rope_scaling` or `rope_parameters`, read for `rope_type` and `rope_theta`.
        struct RopeParameters: Decodable {
            let ropeType: String?
            let ropeTheta: Float?

            enum CodingKeys: String, CodingKey {
                case ropeType = "rope_type"
                case ropeTheta = "rope_theta"
            }
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case vocabularySize = "vocab_size"
            case headDim = "head_dim"
            case shareQDim = "share_q_dim"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeParameters = "rope_parameters"
            case ropeScaling = "rope_scaling"
            case moeIntermediateSize = "moe_intermediate_size"
            case moeExperts = "moe_num_experts"
            case moeTopK = "moe_top_k"
            case normExpertWeight = "norm_expert_weight"
            case shareExpertDim = "share_expert_dim"
            case moeLayers = "moe_layers_enum"
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let d = TextConfiguration()

            // `rope_scaling` wins over `rope_parameters`, as in step3_vl.py (UNVERIFIED: source not re-checked here).
            let rope =
                try c.decodeIfPresent(RopeParameters.self, forKey: .ropeScaling)
                ?? c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
            let topTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)

            // A string or list the Int decoder cannot read is treated as absent. The default applies.
            let moe: [Int]
            if let list = try? c.decode([Int].self, forKey: .moeLayers) {
                moe = list
            } else if let text = try? c.decode(String.self, forKey: .moeLayers) {
                moe = text.split(separator: ",").compactMap {
                    Int($0.trimmingCharacters(in: .whitespaces))
                }
            } else {
                moe = d.moeLayers
            }

            self.init(
                hiddenSize: try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? d.hiddenSize,
                hiddenLayers: try c.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                    ?? d.hiddenLayers,
                attentionHeads: try c.decodeIfPresent(Int.self, forKey: .attentionHeads)
                    ?? d.attentionHeads,
                intermediateSize: try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
                    ?? d.intermediateSize,
                vocabularySize: try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
                    ?? d.vocabularySize,
                headDim: try c.decodeIfPresent(Int.self, forKey: .headDim) ?? d.headDim,
                shareQDim: try c.decodeIfPresent(Int.self, forKey: .shareQDim) ?? d.shareQDim,
                rmsNormEps: try c.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                    ?? d.rmsNormEps,
                ropeTheta: rope?.ropeTheta ?? topTheta ?? d.ropeTheta,
                ropeType: rope?.ropeType ?? d.ropeType,
                moeIntermediateSize: try c.decodeIfPresent(
                    Int.self, forKey: .moeIntermediateSize) ?? d.moeIntermediateSize,
                moeExperts: try c.decodeIfPresent(Int.self, forKey: .moeExperts) ?? d.moeExperts,
                moeTopK: try c.decodeIfPresent(Int.self, forKey: .moeTopK) ?? d.moeTopK,
                normExpertWeight: try c.decodeIfPresent(Bool.self, forKey: .normExpertWeight)
                    ?? d.normExpertWeight,
                shareExpertDim: try c.decodeIfPresent(Int.self, forKey: .shareExpertDim)
                    ?? d.shareExpertDim,
                moeLayers: moe
            )
        }
    }

    /// `vision_config`: the Perception Encoder (`PerceptionEncoder` in vLLM `step_vl.py`).
    public struct VisionConfiguration: Decodable, Sendable {
        public let width: Int
        public let layers: Int
        public let heads: Int
        public let patchSize: Int
        public let imageSize: Int
        /// UNVERIFIED default: 4.0, the Perception Encoder default.
        public let mlpRatio: Float
        /// nil means no LayerScale.
        public let layerScaleInit: Float?
        /// UNVERIFIED default: false.
        public let useClsToken: Bool
        /// UNVERIFIED default: true.
        public let useAbsPosEmb: Bool
        /// UNVERIFIED default: true.
        public let useLnPre: Bool
        /// UNVERIFIED default: true.
        public let useLnPost: Bool
        /// Must be true. vLLM rejects the absolute-only variant.
        public let useRope2d: Bool
        /// UNVERIFIED default: "gelu".
        public let hiddenAct: String

        public init(
            width: Int, layers: Int, heads: Int, patchSize: Int, imageSize: Int,
            mlpRatio: Float = 4.0, layerScaleInit: Float? = nil, useClsToken: Bool = false,
            useAbsPosEmb: Bool = true, useLnPre: Bool = true, useLnPost: Bool = true,
            useRope2d: Bool = true, hiddenAct: String = "gelu"
        ) {
            self.width = width
            self.layers = layers
            self.heads = heads
            self.patchSize = patchSize
            self.imageSize = imageSize
            self.mlpRatio = mlpRatio
            self.layerScaleInit = layerScaleInit
            self.useClsToken = useClsToken
            self.useAbsPosEmb = useAbsPosEmb
            self.useLnPre = useLnPre
            self.useLnPost = useLnPost
            self.useRope2d = useRope2d
            self.hiddenAct = hiddenAct
        }

        enum CodingKeys: String, CodingKey {
            case width, layers, heads
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case mlpRatio = "mlp_ratio"
            case layerScaleInit = "ls_init_value"
            case useClsToken = "use_cls_token"
            case useAbsPosEmb = "use_abs_posemb"
            case useLnPre = "use_ln_pre"
            case useLnPost = "use_ln_post"
            case useRope2d = "use_rope2d"
            case hiddenAct = "hidden_act"
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                width: try c.decode(Int.self, forKey: .width),
                layers: try c.decode(Int.self, forKey: .layers),
                heads: try c.decode(Int.self, forKey: .heads),
                patchSize: try c.decode(Int.self, forKey: .patchSize),
                imageSize: try c.decode(Int.self, forKey: .imageSize),
                mlpRatio: try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4.0,
                layerScaleInit: try c.decodeIfPresent(Float.self, forKey: .layerScaleInit),
                useClsToken: try c.decodeIfPresent(Bool.self, forKey: .useClsToken) ?? false,
                useAbsPosEmb: try c.decodeIfPresent(Bool.self, forKey: .useAbsPosEmb) ?? true,
                useLnPre: try c.decodeIfPresent(Bool.self, forKey: .useLnPre) ?? true,
                useLnPost: try c.decodeIfPresent(Bool.self, forKey: .useLnPost) ?? true,
                useRope2d: try c.decodeIfPresent(Bool.self, forKey: .useRope2d) ?? true,
                hiddenAct: try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
            )
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let projectorBias: Bool
    /// Placeholder token that marks image slots. Default from `Step3VLConfig`, UNVERIFIED for StepVL.
    public let imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case projectorBias = "projector_bias"
        case imageTokenId = "image_token_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "step_robotics"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration()
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 128001
    }

    public func validateModelConfiguration() throws {
        let t = textConfiguration
        let v = visionConfiguration

        guard t.hiddenSize > 0, t.hiddenLayers > 0, t.attentionHeads > 0,
            t.intermediateSize > 0, t.vocabularySize > 0, t.resolvedQDim > 0
        else {
            throw SharpAIStepRoboticsError.invalidConfiguration(
                "text_config sizes must be positive")
        }
        guard t.headDim > 0, t.headDim % 2 == 0 else {
            throw SharpAIStepRoboticsError.invalidConfiguration("head_dim must be a positive even number")
        }
        guard t.ropeType == "default" else {
            throw SharpAIStepRoboticsError.invalidConfiguration(
                "unsupported rope_type '\(t.ropeType)'")
        }
        guard t.moeExperts > 0, t.moeTopK > 0, t.moeTopK <= t.moeExperts,
            t.moeIntermediateSize > 0, t.shareExpertDim > 0
        else {
            throw SharpAIStepRoboticsError.invalidConfiguration("MoE sizes are invalid")
        }
        guard v.width > 0, v.layers > 0, v.heads > 0, v.width % v.heads == 0 else {
            throw SharpAIStepRoboticsError.invalidConfiguration(
                "vision width must be a positive multiple of heads")
        }
        guard (v.width / v.heads) % 4 == 0 else {
            throw SharpAIStepRoboticsError.invalidConfiguration(
                "vision head_dim must be divisible by 4 for 2D RoPE")
        }
        guard v.patchSize > 0, v.imageSize > 0, v.imageSize % v.patchSize == 0 else {
            throw SharpAIStepRoboticsError.invalidConfiguration(
                "image_size must be a positive multiple of patch_size")
        }
        guard v.useRope2d else {
            throw SharpAIStepRoboticsError.invalidConfiguration("use_rope2d must be true")
        }
        guard v.mlpRatio > 0 else {
            throw SharpAIStepRoboticsError.invalidConfiguration("mlp_ratio must be positive")
        }
        guard SharpAIStepRoboticsActivation.function(for: v.hiddenAct) != nil else {
            throw SharpAIStepRoboticsError.invalidConfiguration(
                "unsupported vision hidden_act '\(v.hiddenAct)'")
        }
    }
}

public enum SharpAIStepRoboticsError: Error, Equatable {
    case invalidConfiguration(String)
    case imageChannelMismatch(Int)
    case imageSizeMismatch(expected: Int, height: Int, width: Int)
    /// Only the fixed training grid is supported. Position embeddings are not resampled.
    case placeholderMismatch(placeholders: Int, features: Int)
}
