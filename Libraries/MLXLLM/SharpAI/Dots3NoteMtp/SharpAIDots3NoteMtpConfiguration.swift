// SharpAI fork overlay: dots3_note_mtp configuration. Not part of upstream.
//
// The dots3-note checkpoint config has no MTP-specific field. The MTP head is one dense
// MLA decoder layer at index `num_hidden_layers`, with the sliding-attention (`swa_*`)
// geometry. Sources: sglang python/sglang/srt/models/dots3_common/nextn.py and modeling.py
// (`num_nextn_predict_layers == 1`, `layer_types == ["sliding_attention"]`, dense MLP), and
// transformers modeling_dots3_note.py (`Dots3NoteTextAttention`, `Dots3NoteTextDecoderLayer`).

import Foundation

public struct SharpAIDots3NoteMtpConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var vocabSize: Int
    public var intermediateSize: Int
    /// Index of the MTP layer in the checkpoint (`model.layers.{numHiddenLayers}.*`).
    public var numHiddenLayers: Int
    /// The released checkpoint carries exactly one MTP layer. Absent from config.json, so it defaults to 1.
    public var numNextnPredictLayers: Int
    public var rmsNormEps: Float
    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    /// HF `sliding_window` (the `sliding_window_size` alias in config.json). A query at position i
    /// attends to keys j with 0 <= i - j < slidingWindowSize.
    public var slidingWindowSize: Int
    public var applyMlaQkvLoraRescale: Bool
    public var swaNumAttentionHeads: Int
    public var swaNumKeyValueHeads: Int
    public var swaQLoraRank: Int
    public var swaKvLoraRank: Int
    public var swaQkNopeHeadDim: Int
    public var swaQkRopeHeadDim: Int
    public var swaVHeadDim: Int
    public var swaRopeTheta: Float
    public var swaAttentionGateType: String

    public var swaQkHeadDim: Int { swaQkNopeHeadDim + swaQkRopeHeadDim }

    public init(
        hiddenSize: Int,
        vocabSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int = 46,
        numNextnPredictLayers: Int = 1,
        rmsNormEps: Float = 1e-5,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 524_288,
        slidingWindowSize: Int = 513,
        applyMlaQkvLoraRescale: Bool = true,
        swaNumAttentionHeads: Int = 64,
        swaNumKeyValueHeads: Int = 64,
        swaQLoraRank: Int = 1024,
        swaKvLoraRank: Int = 1024,
        swaQkNopeHeadDim: Int = 192,
        swaQkRopeHeadDim: Int = 64,
        swaVHeadDim: Int = 128,
        swaRopeTheta: Float = 50_000,
        swaAttentionGateType: String = "headwise"
    ) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numNextnPredictLayers = numNextnPredictLayers
        self.rmsNormEps = rmsNormEps
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.slidingWindowSize = slidingWindowSize
        self.applyMlaQkvLoraRescale = applyMlaQkvLoraRescale
        self.swaNumAttentionHeads = swaNumAttentionHeads
        self.swaNumKeyValueHeads = swaNumKeyValueHeads
        self.swaQLoraRank = swaQLoraRank
        self.swaKvLoraRank = swaKvLoraRank
        self.swaQkNopeHeadDim = swaQkNopeHeadDim
        self.swaQkRopeHeadDim = swaQkRopeHeadDim
        self.swaVHeadDim = swaVHeadDim
        self.swaRopeTheta = swaRopeTheta
        self.swaAttentionGateType = swaAttentionGateType
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case rmsNormEps = "rms_norm_eps"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindowSize = "sliding_window_size"
        case applyMlaQkvLoraRescale = "apply_mla_qkv_lora_rescale"
        case swaNumAttentionHeads = "swa_num_attention_heads"
        case swaNumKeyValueHeads = "swa_num_key_value_heads"
        case swaQLoraRank = "swa_q_lora_rank"
        case swaKvLoraRank = "swa_kv_lora_rank"
        case swaQkNopeHeadDim = "swa_qk_nope_head_dim"
        case swaQkRopeHeadDim = "swa_qk_rope_head_dim"
        case swaVHeadDim = "swa_v_head_dim"
        case swaRopeTheta = "swa_rope_theta"
        case swaAttentionGateType = "swa_attention_gate_type"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let config = try Self.decodeValidated(c)
        self = config
    }

    private static func decodeValidated(
        _ c: KeyedDecodingContainer<CodingKeys>
    ) throws -> SharpAIDots3NoteMtpConfiguration {
        let config = SharpAIDots3NoteMtpConfiguration(
            hiddenSize: try c.decode(Int.self, forKey: .hiddenSize),
            vocabSize: try c.decode(Int.self, forKey: .vocabSize),
            intermediateSize: try c.decode(Int.self, forKey: .intermediateSize),
            numHiddenLayers: try c.decode(Int.self, forKey: .numHiddenLayers),
            numNextnPredictLayers: try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1,
            rmsNormEps: try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            hiddenAct: try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu",
            maxPositionEmbeddings: try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? 524_288,
            slidingWindowSize: try c.decodeIfPresent(Int.self, forKey: .slidingWindowSize) ?? 513,
            applyMlaQkvLoraRescale: try c.decodeIfPresent(Bool.self, forKey: .applyMlaQkvLoraRescale)
                ?? true,
            swaNumAttentionHeads: try c.decodeIfPresent(Int.self, forKey: .swaNumAttentionHeads) ?? 64,
            swaNumKeyValueHeads: try c.decodeIfPresent(Int.self, forKey: .swaNumKeyValueHeads) ?? 64,
            swaQLoraRank: try c.decodeIfPresent(Int.self, forKey: .swaQLoraRank) ?? 1024,
            swaKvLoraRank: try c.decodeIfPresent(Int.self, forKey: .swaKvLoraRank) ?? 1024,
            swaQkNopeHeadDim: try c.decodeIfPresent(Int.self, forKey: .swaQkNopeHeadDim) ?? 192,
            swaQkRopeHeadDim: try c.decodeIfPresent(Int.self, forKey: .swaQkRopeHeadDim) ?? 64,
            swaVHeadDim: try c.decodeIfPresent(Int.self, forKey: .swaVHeadDim) ?? 128,
            swaRopeTheta: try c.decodeIfPresent(Float.self, forKey: .swaRopeTheta) ?? 50_000,
            swaAttentionGateType: try c.decodeIfPresent(String.self, forKey: .swaAttentionGateType)
                ?? "headwise"
        )

        // Fail loudly on geometry this head does not implement, rather than loading wrong weights.
        if config.numNextnPredictLayers != 1 {
            throw DecodingError.dataCorruptedError(
                forKey: .numNextnPredictLayers, in: c,
                debugDescription: "dots3_note_mtp supports exactly one MTP layer.")
        }
        if config.hiddenAct != "silu" {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenAct, in: c,
                debugDescription: "dots3_note_mtp only supports silu, got \(config.hiddenAct).")
        }
        if config.swaAttentionGateType != "headwise" {
            throw DecodingError.dataCorruptedError(
                forKey: .swaAttentionGateType, in: c,
                debugDescription:
                    "dots3_note_mtp only supports headwise gates, got \(config.swaAttentionGateType).")
        }
        if config.swaNumKeyValueHeads != config.swaNumAttentionHeads {
            throw DecodingError.dataCorruptedError(
                forKey: .swaNumKeyValueHeads, in: c,
                debugDescription: "dots3 MLA requires equal query and key/value head counts.")
        }
        return config
    }
}
