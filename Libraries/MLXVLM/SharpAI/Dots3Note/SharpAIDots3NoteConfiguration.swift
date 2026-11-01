// SharpAI fork overlay: dots3_note configuration. Not part of upstream mlx-swift-lm.
//
// Decodes the dots-studio/dots3-note-prev config, including the per-layer full and
// sliding-window attention geometry. Tower fields are decoded, but no tower is built yet.
// Reference: vLLM dots3_note modules (Dots3NoteFullAttention, Dots3NoteSlidingAttention,
// Dots3NoteMoE) and the published config.json. config.json has no MTP key.
// Verified against: https://huggingface.co/dots-studio/dots3-note-prev/raw/main/config.json
//   Every vision_config and audio_config key decoded below is present in the published config.json.
//   Decoding is not yet checked against a running model.

import Foundation
import MLXLMCommon

public enum SharpAIDots3NoteError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(MLA/DSA attention, headwise gates, MoE routing and towers are missing; "
                + "architecture UNVERIFIED)."
        }
    }
}

public enum SharpAIDots3NoteLayerType: String, Sendable, Equatable {
    case fullAttention = "full_attention"
    case slidingAttention = "sliding_attention"
}

/// Attention shape for one decoder layer, picked from the full or SWA branch of the config.
public struct SharpAIDots3NoteAttentionGeometry: Sendable, Equatable {
    public let layerType: SharpAIDots3NoteLayerType
    public let attentionHeads: Int
    public let kvHeads: Int
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let ropeTheta: Float
    public let gateType: String
    /// Nil for full-attention layers.
    public let slidingWindow: Int?

    public var qHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }
}

public struct SharpAIDots3NoteWhisperConfiguration: Decodable, Sendable, Equatable {
    public let dModel: Int
    public let encoderAttentionHeads: Int
    public let encoderFfnDim: Int
    public let encoderLayers: Int
    public let numMelBins: Int
    public let maxSourcePositions: Int
    public let activationFunction: String

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case encoderLayers = "encoder_layers"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case activationFunction = "activation_function"
    }
}

public struct SharpAIDots3NoteAudioConfiguration: Decodable, Sendable, Equatable {
    public let encoderType: String
    public let whisper: SharpAIDots3NoteWhisperConfiguration
    public let useConv2dStem: Bool
    public let downsampleHiddenSize: Int
    public let samplingRate: Int
    public let chunkSeconds: Int
    public let whisperAdapterInDim: Int
    public let whisperAdapterOutDim: Int

    enum CodingKeys: String, CodingKey {
        case encoderType = "encoder_type"
        case whisper = "whisper_config"
        case useConv2dStem = "use_conv2d_stem"
        case downsampleHiddenSize = "downsample_hidden_size"
        case samplingRate = "sampling_rate"
        case chunkSeconds = "chunk_seconds"
        case whisperAdapterInDim = "whisper_adapter_in_dim"
        case whisperAdapterOutDim = "whisper_adapter_out_dim"
    }
}

public struct SharpAIDots3NoteVisionConfiguration: Decodable, Sendable, Equatable {
    public let embedDim: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let numChannels: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let rmsNormEps: Float
    public let pyramidNumRouted: [Int]
    public let capacityFactor: Float
    public let adapterType: String
    public let adapterInDim: Int
    public let adapterOutDim: Int
    public let adapterMergeSize: Int

    enum CodingKeys: String, CodingKey {
        case embedDim = "embed_dim"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case rmsNormEps = "rms_norm_eps"
        case pyramidNumRouted = "pyramid_num_routed"
        case capacityFactor = "capacity_factor"
        case adapterType = "adapter_type"
        case adapterInDim = "adapter_in_dim"
        case adapterOutDim = "adapter_out_dim"
        case adapterMergeSize = "adapter_merge_size"
    }
}

public struct SharpAIDots3NoteConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let layerTypes: [SharpAIDots3NoteLayerType]

    // Full-attention branch (MLA, plus DSA indexer fields on the top level).
    public let attentionHeads: Int
    public let kvHeads: Int
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let ropeTheta: Float
    public let attentionGateType: String

    // Sliding-window branch (SWA MLA). Verified against: model.safetensors.index.json, where
    // `self_attn.indexer.*` keys exist on exactly the 13 full_attention layers and on no
    // sliding_attention layer.
    public let slidingWindowSize: Int
    public let swaAttentionHeads: Int
    public let swaKvHeads: Int
    public let swaQLoraRank: Int
    public let swaKvLoraRank: Int
    public let swaQkNopeHeadDim: Int
    public let swaQkRopeHeadDim: Int
    public let swaVHeadDim: Int
    public let swaRopeTheta: Float
    public let swaAttentionGateType: String

    public let indexHeadDim: Int?
    public let indexHeads: Int?
    public let indexTopK: Int?

    // MoE. Layer selection follows the DeepSeek rule used in DeepseekV3.swift.
    public let moeIntermediateSize: Int
    public let moeLayerFreq: Int
    public let firstKDenseReplace: Int
    public let routedExperts: Int
    public let sharedExperts: Int
    public let numExpertsPerTok: Int
    public let scoringFunc: String
    public let routedScalingFactor: Float
    public let normTopkProb: Bool

    public let vision: SharpAIDots3NoteVisionConfiguration?
    public let audio: SharpAIDots3NoteAudioConfiguration?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeTheta = "rope_theta"
        case attentionGateType = "attention_gate_type"
        case slidingWindowSize = "sliding_window_size"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKvHeads = "swa_num_key_value_heads"
        case swaQLoraRank = "swa_q_lora_rank"
        case swaKvLoraRank = "swa_kv_lora_rank"
        case swaQkNopeHeadDim = "swa_qk_nope_head_dim"
        case swaQkRopeHeadDim = "swa_qk_rope_head_dim"
        case swaVHeadDim = "swa_v_head_dim"
        case swaRopeTheta = "swa_rope_theta"
        case swaAttentionGateType = "swa_attention_gate_type"
        case indexHeadDim = "index_head_dim"
        case indexHeads = "index_n_heads"
        case indexTopK = "index_topk"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case routedExperts = "n_routed_experts"
        case sharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case scoringFunc = "scoring_func"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case vision = "vision_config"
        case audio = "audio_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "dots3_note"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        let rawLayerTypes = try c.decode([String].self, forKey: .layerTypes)
        layerTypes = try rawLayerTypes.map { raw in
            guard let kind = SharpAIDots3NoteLayerType(rawValue: raw) else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath + [CodingKeys.layerTypes as CodingKey],
                        debugDescription: "unsupported dots3_note layer type '\(raw)'"))
            }
            return kind
        }

        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        attentionGateType = try c.decode(String.self, forKey: .attentionGateType)

        slidingWindowSize = try c.decode(Int.self, forKey: .slidingWindowSize)
        swaAttentionHeads = try c.decode(Int.self, forKey: .swaAttentionHeads)
        swaKvHeads = try c.decode(Int.self, forKey: .swaKvHeads)
        swaQLoraRank = try c.decode(Int.self, forKey: .swaQLoraRank)
        swaKvLoraRank = try c.decode(Int.self, forKey: .swaKvLoraRank)
        swaQkNopeHeadDim = try c.decode(Int.self, forKey: .swaQkNopeHeadDim)
        swaQkRopeHeadDim = try c.decode(Int.self, forKey: .swaQkRopeHeadDim)
        swaVHeadDim = try c.decode(Int.self, forKey: .swaVHeadDim)
        swaRopeTheta = try c.decode(Float.self, forKey: .swaRopeTheta)
        swaAttentionGateType = try c.decode(String.self, forKey: .swaAttentionGateType)

        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim)
        indexHeads = try c.decodeIfPresent(Int.self, forKey: .indexHeads)
        indexTopK = try c.decodeIfPresent(Int.self, forKey: .indexTopK)

        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        routedExperts = try c.decode(Int.self, forKey: .routedExperts)
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        vision = try c.decodeIfPresent(SharpAIDots3NoteVisionConfiguration.self, forKey: .vision)
        audio = try c.decodeIfPresent(SharpAIDots3NoteAudioConfiguration.self, forKey: .audio)
    }

    /// Attention shape for `layer`, or nil when the index is out of range.
    public func attentionGeometry(at layer: Int) -> SharpAIDots3NoteAttentionGeometry? {
        guard layer >= 0, layer < layerTypes.count else { return nil }
        switch layerTypes[layer] {
        case .fullAttention:
            return SharpAIDots3NoteAttentionGeometry(
                layerType: .fullAttention,
                attentionHeads: attentionHeads,
                kvHeads: kvHeads,
                qLoraRank: qLoraRank,
                kvLoraRank: kvLoraRank,
                qkNopeHeadDim: qkNopeHeadDim,
                qkRopeHeadDim: qkRopeHeadDim,
                vHeadDim: vHeadDim,
                ropeTheta: ropeTheta,
                gateType: attentionGateType,
                slidingWindow: nil)
        case .slidingAttention:
            return SharpAIDots3NoteAttentionGeometry(
                layerType: .slidingAttention,
                attentionHeads: swaAttentionHeads,
                kvHeads: swaKvHeads,
                qLoraRank: swaQLoraRank,
                kvLoraRank: swaKvLoraRank,
                qkNopeHeadDim: swaQkNopeHeadDim,
                qkRopeHeadDim: swaQkRopeHeadDim,
                vHeadDim: swaVHeadDim,
                ropeTheta: swaRopeTheta,
                gateType: swaAttentionGateType,
                slidingWindow: slidingWindowSize)
        }
    }

    /// True when `layer` uses the routed MoE block instead of a dense MLP.
    public func isMoELayer(_ layer: Int) -> Bool {
        layer >= firstKDenseReplace && layer % moeLayerFreq == 0
    }

    public func validateModelConfiguration() throws {
        let dims = [hiddenSize, hiddenLayers, vocabularySize, attentionHeads, kvHeads,
            swaAttentionHeads, swaKvHeads, moeIntermediateSize, routedExperts, numExpertsPerTok]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "dots3_note: hidden, layer, head, expert and vocab sizes must be positive")
        }
        guard layerTypes.count == hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "dots3_note: layer_types has \(layerTypes.count) entries, "
                    + "but num_hidden_layers is \(hiddenLayers)")
        }
        guard attentionHeads % kvHeads == 0, swaAttentionHeads % swaKvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "dots3_note: attention heads must be divisible by kv heads in both branches")
        }
        guard moeLayerFreq > 0, numExpertsPerTok <= routedExperts else {
            throw ModelFactoryError.invalidConfiguration(
                "dots3_note: moe_layer_freq must be positive and experts_per_tok <= n_routed_experts")
        }
    }
}
