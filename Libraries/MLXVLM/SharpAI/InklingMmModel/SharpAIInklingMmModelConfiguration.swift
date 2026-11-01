// SharpAI fork overlay: inkling_mm_model configuration. Not part of upstream mlx-swift-lm.
//
// Key names and values come from the published config.json of thinkingmachines/Inkling
// (Verified against: https://huggingface.co/thinkingmachines/Inkling/resolve/main/config.json)
// and transformers `configuration_inkling.py`
// (Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/inkling/configuration_inkling.py).
// Architecture values have no defaults: a missing key throws instead of guessing a value.
// Two derivations follow transformers: `layer_types` comes from `local_layer_ids` (or the
// `(i + 1) % 6` rule when neither is present), and `mlp_layer_types` comes from `dense_mlp_idx`
// (default 0). `intermediate_size` is the expert width when `dense_intermediate_size` is present,
// which is the published layout (the tensor shapes in the safetensors headers confirm it).
// The MTP block (`mtp_config`) is not decoded: MTP weights are dropped at load.

import Foundation
import MLXLMCommon

public enum SharpAIInklingMmModelLayerType: String, Sendable, Equatable {
    case hybrid
    case hybridSliding = "hybrid_sliding"
}

public enum SharpAIInklingMmModelMLPType: String, Sendable, Equatable {
    case dense
    case sparse
}

public struct SharpAIInklingMmModelTextConfiguration: Decodable, Sendable, Equatable {
    public let vocabularySize: Int
    /// Rows of the unembedding that carry real tokens. Logits past this count are dropped.
    public let unpaddedVocabularySize: Int?
    public let hiddenSize: Int
    public let hiddenLayers: Int

    // Full-attention ("hybrid") layers.
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int

    // Sliding-window ("hybrid_sliding") layers.
    public let swaAttentionHeads: Int
    public let swaKVHeads: Int
    public let swaHeadDim: Int
    public let slidingWindowSize: Int

    /// Per-head width of the relative states that mix into the position bias.
    public let dRel: Int
    /// Backward distance covered by the bias in full-attention layers. Sliding layers use the window.
    public let relExtent: Int
    /// Position from which full-attention logits are log-scaled. `nil` disables the scaling.
    public let logScalingNFloor: Int?
    public let logScalingAlpha: Float

    public let layerTypes: [SharpAIInklingMmModelLayerType]
    public let mlpLayerTypes: [SharpAIInklingMmModelMLPType]

    public let rmsNormEps: Float
    public let convKernelSize: Int

    /// Width of the dense MLP.
    public let intermediateSize: Int
    /// Width of each routed and shared expert.
    public let moeIntermediateSize: Int
    public let hiddenAct: String

    public let routedExperts: Int
    public let numExpertsPerToken: Int
    public let sharedExperts: Int
    public let sharedExpertSink: Bool
    public let routeScale: Float

    /// The final hidden states are divided by this before the LM head.
    public let logitsMupWidthMultiplier: Float

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case unpaddedVocabSize = "unpadded_vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case swaNumAttentionHeads = "swa_num_attention_heads"
        case swaNumKeyValueHeads = "swa_num_key_value_heads"
        case swaHeadDim = "swa_head_dim"
        case slidingWindowSize = "sliding_window_size"
        case dRel = "d_rel"
        case relExtent = "rel_extent"
        case logScalingNFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case localLayerIds = "local_layer_ids"
        case layerTypes = "layer_types"
        case denseMlpIdx = "dense_mlp_idx"
        case mlpLayerTypes = "mlp_layer_types"
        case rmsNormEps = "rms_norm_eps"
        case sconvKernelSize = "sconv_kernel_size"
        case convKernelSize = "conv_kernel_size"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "dense_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenAct = "hidden_act"
        case routedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case sharedExperts = "n_shared_experts"
        case sharedExpertSink = "shared_expert_sink"
        case routeScale = "route_scale"
        case logitsMupWidthMultiplier = "logits_mup_width_multiplier"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        vocabularySize = try c.decode(Int.self, forKey: .vocabSize)
        unpaddedVocabularySize = try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)

        attentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        swaAttentionHeads = try c.decode(Int.self, forKey: .swaNumAttentionHeads)
        swaKVHeads = try c.decode(Int.self, forKey: .swaNumKeyValueHeads)
        swaHeadDim = try c.decode(Int.self, forKey: .swaHeadDim)
        slidingWindowSize = try c.decode(Int.self, forKey: .slidingWindowSize)

        dRel = try c.decode(Int.self, forKey: .dRel)
        relExtent = try c.decode(Int.self, forKey: .relExtent)
        logScalingNFloor = try c.decodeIfPresent(Int.self, forKey: .logScalingNFloor)
        logScalingAlpha = try c.decode(Float.self, forKey: .logScalingAlpha)

        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        if let sconv = try c.decodeIfPresent(Int.self, forKey: .sconvKernelSize) {
            convKernelSize = sconv
        } else {
            convKernelSize = try c.decode(Int.self, forKey: .convKernelSize)
        }

        // transformers: with `dense_intermediate_size`, `intermediate_size` is the expert width.
        let rawIntermediate = try c.decode(Int.self, forKey: .intermediateSize)
        if let dense = try c.decodeIfPresent(Int.self, forKey: .denseIntermediateSize) {
            intermediateSize = dense
            moeIntermediateSize = rawIntermediate
        } else {
            intermediateSize = rawIntermediate
            moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        }
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"

        routedExperts = try c.decode(Int.self, forKey: .routedExperts)
        numExpertsPerToken = try c.decode(Int.self, forKey: .numExpertsPerTok)
        sharedExperts = try c.decode(Int.self, forKey: .sharedExperts)
        sharedExpertSink = try c.decodeIfPresent(Bool.self, forKey: .sharedExpertSink) ?? true
        routeScale = try c.decode(Float.self, forKey: .routeScale)
        logitsMupWidthMultiplier = try c.decode(Float.self, forKey: .logitsMupWidthMultiplier)

        if let raw = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = try raw.map { (name: String) throws -> SharpAIInklingMmModelLayerType in
                guard let type = SharpAIInklingMmModelLayerType(rawValue: name) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: CodingKeys.layerTypes, in: c,
                        debugDescription: "unknown layer type \(name)")
                }
                return type
            }
        } else {
            let local = try c.decodeIfPresent([Int].self, forKey: .localLayerIds)
            let localSet = Set(local ?? (0 ..< hiddenLayers).filter { ($0 + 1) % 6 != 0 })
            layerTypes = (0 ..< hiddenLayers).map {
                (i: Int) -> SharpAIInklingMmModelLayerType in
                localSet.contains(i) ? .hybridSliding : .hybrid
            }
        }

        if let raw = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            mlpLayerTypes = try raw.map { (name: String) throws -> SharpAIInklingMmModelMLPType in
                guard let type = SharpAIInklingMmModelMLPType(rawValue: name) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: CodingKeys.mlpLayerTypes, in: c,
                        debugDescription: "unknown MLP type \(name)")
                }
                return type
            }
        } else {
            let denseIndex = try c.decodeIfPresent(Int.self, forKey: .denseMlpIdx) ?? 0
            mlpLayerTypes = (0 ..< hiddenLayers).map {
                (i: Int) -> SharpAIInklingMmModelMLPType in
                i < denseIndex ? .dense : .sparse
            }
        }
    }
}

public struct SharpAIInklingMmModelAudioConfiguration: Decodable, Sendable, Equatable {
    public let nMelBins: Int
    public let melVocabularySize: Int
    /// Width of the audio embeddings, equal to the text hidden size in the published config.
    public let decoderDModel: Int

    enum CodingKeys: String, CodingKey {
        case nMelBins = "n_mel_bins"
        case melVocabularySize = "mel_vocab_size"
        case decoderDModel = "decoder_dmodel"
    }
}

public struct SharpAIInklingMmModelVisionConfiguration: Decodable, Sendable, Equatable {
    /// Only `hmlp` exists in the published checkpoint.
    public let encoderType: String
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let numChannels: Int
    /// Encoder depth. The last encoder layer projects to the text hidden size.
    public let encoderLayers: Int
    public let decoderDModel: Int

    enum CodingKeys: String, CodingKey {
        case encoderType = "vision_encoder_type"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case numChannels = "n_channels"
        case encoderLayers = "n_layers"
        case decoderDModel = "decoder_dmodel"
    }
}

public struct SharpAIInklingMmModelConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let text: SharpAIInklingMmModelTextConfiguration
    public let audio: SharpAIInklingMmModelAudioConfiguration
    public let vision: SharpAIInklingMmModelVisionConfiguration
    /// Placeholder token ids are not in the published config. Only multimodal input needs them.
    public let imageTokenId: Int?
    public let audioTokenId: Int?
    public let imageBosTokenId: Int?
    public let audioBosTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case audio = "audio_config"
        case vision = "vision_config"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case imageBosTokenId = "image_bos_token_id"
        case audioBosTokenId = "audio_bos_token_id"
    }

    public func validateModelConfiguration() throws {
        let text = self.text
        let dims = [
            text.vocabularySize, text.hiddenSize, text.hiddenLayers, text.attentionHeads,
            text.kvHeads, text.headDim, text.swaAttentionHeads, text.swaKVHeads, text.swaHeadDim,
            text.slidingWindowSize, text.convKernelSize, text.intermediateSize,
            text.moeIntermediateSize, text.routedExperts, text.numExpertsPerToken,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "inkling_mm_model: vocab, hidden, layer, head, window, conv and expert sizes must be positive")
        }
        guard text.layerTypes.count == text.hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "inkling_mm_model: layer_types has \(text.layerTypes.count) entries, "
                    + "but num_hidden_layers is \(text.hiddenLayers)")
        }
        guard text.mlpLayerTypes.count == text.hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "inkling_mm_model: mlp_layer_types has \(text.mlpLayerTypes.count) entries, "
                    + "but num_hidden_layers is \(text.hiddenLayers)")
        }
        guard text.attentionHeads % text.kvHeads == 0, text.swaAttentionHeads % text.swaKVHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "inkling_mm_model: attention heads must be divisible by kv heads in both branches")
        }
        guard text.numExpertsPerToken <= text.routedExperts else {
            throw ModelFactoryError.invalidConfiguration(
                "inkling_mm_model: num_experts_per_tok must not exceed n_routed_experts")
        }
        guard vision.encoderType == "hmlp" else {
            throw ModelFactoryError.invalidConfiguration(
                "inkling_mm_model: vision_encoder_type \(vision.encoderType) is not supported (only hmlp)")
        }
        guard vision.encoderLayers >= 1, vision.patchSize > 0, vision.temporalPatchSize > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "inkling_mm_model: vision encoder needs at least one layer and positive patch sizes")
        }
    }
}
