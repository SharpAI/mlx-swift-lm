// SharpAI fork overlay: molmo2 text decoder (language stack only). Not part of upstream mlx-swift-lm.
//
// Port of the text half of modeling_molmo2.py in allenai/Molmo2-O-7B: Molmo2TextModel,
// Molmo2PostNormDecoderLayer / Molmo2DecoderLayer, Molmo2Attention, LanguageModelMLP,
// Molmo2Embedding, Molmo2RotaryEmbedding (YaRN for the layers in rope_scaling_layers) and the
// lm_head of Molmo2ForConditionalGeneration. Checked against the transformers modeling file and
// the Molmo2-O-7B config.json, both fetched from Hugging Face on 2026-10-30.
//
// NOT covered here: the vision backbone (ViT, image pooling, projector) and the image/video token
// merge. The creator in SharpAIMolmo2Registration.swift still throws, so this decoder is not
// reachable from model loading yet.
//
// Mask: causal only. The reference attention_mask (padding) is not read.
// Weight keys (checkpoint layout): model.transformer.wte.{embedding,new_embedding},
// model.transformer.blocks.N.{att_proj,attn_out,q_norm,k_norm,ff_proj,ff_out,attn_norm,ff_norm},
// model.transformer.ln_f.weight, lm_head.weight. The rotary inv_freq is a buffer, not a weight.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Arguments

/// Text-side arguments, decoded from the `text_config` object of a Molmo2 config.json.
/// Missing fields take the defaults of `Molmo2TextConfig.__init__` in configuration_molmo2.py.
struct SharpAIMolmo2TextArguments: Decodable {

    /// YaRN parameters from `rope_scaling`. `mscale` and `mscale_all_dim` are not read: Molmo2
    /// sets `attention_factor` directly, so the reference's mscale branch is never taken.
    struct RopeScaling: Decodable {
        let factor: Float
        let attentionFactor: Float?
        let betaFast: Float
        let betaSlow: Float
        let originalMaxPositionEmbeddings: Int?
        let truncate: Bool

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case factor
            case attentionFactor = "attention_factor"
            case betaFast = "beta_fast"
            case betaSlow = "beta_slow"
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case truncate
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType)
            guard ropeType == "yarn" else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeType, in: container,
                    debugDescription:
                        "molmo2: only rope_type yarn is supported, got \(ropeType ?? "nil")")
            }
            factor = try container.decode(Float.self, forKey: .factor)
            attentionFactor = try container.decodeIfPresent(Float.self, forKey: .attentionFactor)
            betaFast = try container.decodeIfPresent(Float.self, forKey: .betaFast) ?? 32
            betaSlow = try container.decodeIfPresent(Float.self, forKey: .betaSlow) ?? 1
            originalMaxPositionEmbeddings = try container.decodeIfPresent(
                Int.self, forKey: .originalMaxPositionEmbeddings)
            truncate = try container.decodeIfPresent(Bool.self, forKey: .truncate) ?? true
        }
    }

    let hiddenSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let additionalVocabularySize: Int
    let qkvBias: Bool
    let layers: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let ropeScaling: RopeScaling?
    let ropeScalingLayers: [Int]?
    let useQKNorm: Bool
    let qkNormType: String
    let layerNormEps: Float
    let normAfter: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case additionalVocabularySize = "additional_vocab_size"
        case qkvBias = "qkv_bias"
        case layers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeScalingLayers = "rope_scaling_layers"
        case useQKNorm = "use_qk_norm"
        case qkNormType = "qk_norm_type"
        case layerNormEps = "layer_norm_eps"
        case normAfter = "norm_after"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3584
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 28
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 152064
        additionalVocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .additionalVocabularySize) ?? 128
        qkvBias = try container.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? true
        layers = try container.decodeIfPresent(Int.self, forKey: .layers) ?? 48
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 18944
        let hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        guard hiddenAct == "silu" else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenAct, in: container,
                debugDescription: "molmo2: only hidden_act silu is supported, got \(hiddenAct)")
        }
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeScaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        ropeScalingLayers = try container.decodeIfPresent([Int].self, forKey: .ropeScalingLayers)
        useQKNorm = try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false
        qkNormType = try container.decodeIfPresent(String.self, forKey: .qkNormType) ?? "olmo"
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        normAfter = try container.decodeIfPresent(Bool.self, forKey: .normAfter) ?? false

        let positive = [
            hiddenSize, attentionHeads, kvHeads, headDim, vocabularySize, layers, intermediateSize,
        ]
        guard positive.allSatisfy({ $0 > 0 }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenSize, in: container,
                debugDescription: "molmo2: text sizes must be positive")
        }
        guard headDim % 2 == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .headDim, in: container,
                debugDescription: "molmo2: head_dim must be even for rotary embeddings")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .kvHeads, in: container,
                debugDescription: "molmo2: num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}

// MARK: - Rotary

/// YaRN inverse frequencies, as `_compute_yarn_parameters` in transformers `modeling_rope_utils.py`.
/// Returns the inverse frequencies and the factor applied to both cos and sin.
func sharpAIMolmo2YarnInverseFrequencies(
    headDim: Int,
    base: Float,
    scaling: SharpAIMolmo2TextArguments.RopeScaling,
    maxPositionEmbeddings: Int
) -> (inverseFrequencies: [Float], attentionFactor: Float) {
    let dim = Double(headDim)
    let theta = Double(base)
    let factor = Double(scaling.factor)
    let originalMax = Double(scaling.originalMaxPositionEmbeddings ?? maxPositionEmbeddings)
    let attentionFactor: Double =
        scaling.attentionFactor.map { Double($0) } ?? (factor <= 1 ? 1 : 0.1 * log(factor) + 1)

    func correctionDim(_ rotations: Double) -> Double {
        dim * log(originalMax / (rotations * 2 * Double.pi)) / (2 * log(theta))
    }

    var low = correctionDim(Double(scaling.betaFast))
    var high = correctionDim(Double(scaling.betaSlow))
    if scaling.truncate {
        low = floor(low)
        high = ceil(high)
    }
    low = max(low, 0)
    high = min(high, dim - 1)
    if low == high {
        high += 0.001  // same singularity guard as linear_ramp_factor
    }

    var inverseFrequencies = [Float]()
    inverseFrequencies.reserveCapacity(headDim / 2)
    for index in 0 ..< headDim / 2 {
        let ramp = min(max((Double(index) - low) / (high - low), 0), 1)
        let extrapolationFactor = 1 - ramp
        let posFreq = pow(theta, Double(2 * index) / dim)
        let extrapolation = 1 / posFreq
        let interpolation = 1 / (factor * posFreq)
        let value =
            interpolation * (1 - extrapolationFactor) + extrapolation * extrapolationFactor
        inverseFrequencies.append(Float(value))
    }
    return (inverseFrequencies, Float(attentionFactor))
}

/// Rotary cos/sin for one frequency schedule. Half-split rotate_half, as in the reference.
final class SharpAIMolmo2Rotary {
    let inverseFrequencies: MLXArray
    let attentionScaling: Float

    init(
        headDim: Int,
        ropeTheta: Float,
        scaling: SharpAIMolmo2TextArguments.RopeScaling?,
        maxPositionEmbeddings: Int
    ) {
        if let scaling {
            let yarn = sharpAIMolmo2YarnInverseFrequencies(
                headDim: headDim, base: ropeTheta, scaling: scaling,
                maxPositionEmbeddings: maxPositionEmbeddings)
            inverseFrequencies = MLXArray(yarn.inverseFrequencies)
            attentionScaling = yarn.attentionFactor
        } else {
            let values: [Float] = (0 ..< headDim / 2).map { index in
                Float(1 / pow(Double(ropeTheta), Double(2 * index) / Double(headDim)))
            }
            inverseFrequencies = MLXArray(values)
            attentionScaling = 1
        }
    }

    /// cos and sin of shape [B or 1, L, headDim], for positions offset..<offset+L.
    func cosSin(length: Int, offset: RoPEOffset?) -> (cos: MLXArray, sin: MLXArray) {
        let steps = MLXArray((0 ..< length).map { Float($0) })
        let positions: MLXArray
        switch offset {
        case nil:
            positions = steps.reshaped(1, length)
        case .scalar(let start):
            positions = (steps + Float(start)).reshaped(1, length)
        case .batch(let starts):
            positions = starts.asType(.float32).reshaped(-1, 1) + steps.reshaped(1, length)
        }
        let freqs = positions.expandedDimensions(axis: -1) * inverseFrequencies
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb) * attentionScaling, MLX.sin(emb) * attentionScaling)
    }

    /// x: [B, H, L, D]. cos and sin: [B or 1, L, D].
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let c = cos.expandedDimensions(axis: 1).asType(x.dtype)
        let s = sin.expandedDimensions(axis: 1).asType(x.dtype)
        let halves = x.split(parts: 2, axis: -1)
        let rotated = concatenated([-halves[1], halves[0]], axis: -1)
        return x * c + rotated * s
    }
}

// MARK: - Embedding

/// Molmo2Embedding: one table for the base vocabulary, one for the additional tokens.
/// Ids at or above the base vocabulary size index `newEmbedding`.
final class SharpAIMolmo2Embedding: Module {
    @ParameterInfo(key: "embedding") var embedding: MLXArray
    @ParameterInfo(key: "new_embedding") var newEmbedding: MLXArray

    init(vocabularySize: Int, additionalVocabularySize: Int, hiddenSize: Int) {
        _embedding.wrappedValue = MLXRandom.normal([vocabularySize, hiddenSize]) * 0.02
        _newEmbedding.wrappedValue = MLXRandom.normal([additionalVocabularySize, hiddenSize]) * 0.02
        super.init()
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let base = embedding.dim(0)
        if newEmbedding.dim(0) == 0 {
            return embedding.take(ids, axis: 0)
        }
        let isBase = ids .< MLXArray(Int32(base))
        let baseRows = embedding.take(
            MLX.where(isBase, ids, MLXArray(Int32(0))), axis: 0)
        let newRows = newEmbedding.take(
            MLX.where(isBase, MLXArray(Int32(0)), ids - MLXArray(Int32(base))), axis: 0)
        return MLX.where(isBase.expandedDimensions(axis: -1), baseRows, newRows)
    }
}

// MARK: - Attention and MLP

final class SharpAIMolmo2Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let qWidth: Int
    let kvWidth: Int
    let scale: Float
    /// true for qk_norm_type "qwen3": the norm is per head (size head_dim).
    /// false for "olmo": the norm is over the full projected width.
    let qkNormPerHead: Bool

    @ModuleInfo(key: "att_proj") var attProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    @ModuleInfo(key: "attn_out") var attnOut: Linear

    init(_ args: SharpAIMolmo2TextArguments) {
        let heads = args.attentionHeads
        let kv = args.kvHeads
        let dim = args.headDim
        let qWidthValue = heads * dim
        let kvWidthValue = kv * dim
        let perHead = args.qkNormType == "qwen3"

        attentionHeads = heads
        kvHeads = kv
        headDim = dim
        qWidth = qWidthValue
        kvWidth = kvWidthValue
        scale = pow(Float(dim), -0.5)
        qkNormPerHead = perHead

        _attProj.wrappedValue = Linear(
            args.hiddenSize, qWidthValue + 2 * kvWidthValue, bias: args.qkvBias)
        _qNorm.wrappedValue =
            args.useQKNorm
            ? RMSNorm(dimensions: perHead ? dim : qWidthValue, eps: args.layerNormEps)
            : nil
        _kNorm.wrappedValue =
            args.useQKNorm
            ? RMSNorm(dimensions: perHead ? dim : kvWidthValue, eps: args.layerNormEps)
            : nil
        _attnOut.wrappedValue = Linear(qWidthValue, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        let fused = attProj(x)
        var q = fused[0..., 0..., ..<qWidth]
        var k = fused[0..., 0..., qWidth ..< qWidth + kvWidth]
        let v = fused[0..., 0..., (qWidth + kvWidth) ..< (qWidth + 2 * kvWidth)]

        if let qn = qNorm, let kn = kNorm, !qkNormPerHead {
            q = qn(q)
            k = kn(k)
        }

        var queries = q.reshaped(batch, length, attentionHeads, headDim)
        var keys = k.reshaped(batch, length, kvHeads, headDim)
        var values = v.reshaped(batch, length, kvHeads, headDim)

        if let qn = qNorm, let kn = kNorm, qkNormPerHead {
            queries = qn(queries)
            keys = kn(keys)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        queries = SharpAIMolmo2Rotary.apply(queries, cos: cos, sin: sin)
        keys = SharpAIMolmo2Rotary.apply(keys, cos: cos, sin: sin)

        if kvHeads != attentionHeads {
            // repeat_kv: each kv head is repeated for its group of query heads.
            let repeats = attentionHeads / kvHeads
            keys = MLX.repeated(keys, count: repeats, axis: 1)
            values = MLX.repeated(values, count: repeats, axis: 1)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return attnOut(output)
    }
}

/// LanguageModelMLP: ff_proj yields [x, gate]; output is ff_out(silu(gate) * x).
final class SharpAIMolmo2MLP: Module {
    @ModuleInfo(key: "ff_proj") var ffProj: Linear
    @ModuleInfo(key: "ff_out") var ffOut: Linear

    init(_ args: SharpAIMolmo2TextArguments) {
        _ffProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize * 2, bias: false)
        _ffOut.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = ffProj(x).split(parts: 2, axis: -1)
        return ffOut(silu(parts[1]) * parts[0])
    }
}

// MARK: - Decoder

final class SharpAIMolmo2Block: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIMolmo2Attention
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIMolmo2MLP
    @ModuleInfo(key: "ff_norm") var ffNorm: RMSNorm
    let normAfter: Bool

    init(_ args: SharpAIMolmo2TextArguments) {
        normAfter = args.normAfter
        _attention.wrappedValue = SharpAIMolmo2Attention(args)
        _attnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        _mlp.wrappedValue = SharpAIMolmo2MLP(args)
        _ffNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        if normAfter {
            // Molmo2PostNormDecoderLayer: norms wrap the sub-layer outputs.
            let h = x + attnNorm(attention(x, cos: cos, sin: sin, mask: mask, cache: cache))
            return h + ffNorm(mlp(h))
        }
        // Molmo2DecoderLayer: pre-norm.
        let h = x + attention(attnNorm(x), cos: cos, sin: sin, mask: mask, cache: cache)
        return h + mlp(ffNorm(h))
    }
}

/// Molmo2TextModel: embedding, decoder blocks, final norm.
final class SharpAIMolmo2TextStack: Module {
    @ModuleInfo(key: "wte") var wte: SharpAIMolmo2Embedding
    let blocks: [SharpAIMolmo2Block]
    @ModuleInfo(key: "ln_f") var lnF: RMSNorm

    let defaultRotary: SharpAIMolmo2Rotary
    let scalingRotary: SharpAIMolmo2Rotary
    /// Per layer: true when the layer uses the YaRN rotary. Follows rope_scaling_layers when
    /// present; otherwise every layer uses the scaling rotary exactly when rope_scaling is set.
    let usesScalingRotary: [Bool]

    init(_ args: SharpAIMolmo2TextArguments) {
        _wte.wrappedValue = SharpAIMolmo2Embedding(
            vocabularySize: args.vocabularySize,
            additionalVocabularySize: args.additionalVocabularySize,
            hiddenSize: args.hiddenSize)
        blocks = (0 ..< args.layers).map { _ in SharpAIMolmo2Block(args) }
        _lnF.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        defaultRotary = SharpAIMolmo2Rotary(
            headDim: args.headDim, ropeTheta: args.ropeTheta, scaling: nil,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
        scalingRotary = SharpAIMolmo2Rotary(
            headDim: args.headDim, ropeTheta: args.ropeTheta, scaling: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
        usesScalingRotary = (0 ..< args.layers).map { (index: Int) -> Bool in
            if let scalingLayers = args.ropeScalingLayers {
                return scalingLayers.contains(index)
            }
            return args.ropeScaling != nil
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Molmo2 uses -1 as the image placeholder id; the reference maps it to 0 before lookup.
        let ids = MLX.where(inputs .== MLXArray(Int32(-1)), MLXArray(Int32(0)), inputs)
        var h = wte(ids)
        let length = h.dim(1)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        let offset = cache?.first?.ropeOffset

        let defaultCosSin = defaultRotary.cosSin(length: length, offset: offset)
        let scalingCosSin = scalingRotary.cosSin(length: length, offset: offset)
        for (index, block) in blocks.enumerated() {
            let rotary = usesScalingRotary[index] ? scalingCosSin : defaultCosSin
            h = block(h, cos: rotary.cos, sin: rotary.sin, mask: mask, cache: cache?[index])
        }
        return lnF(h)
    }
}

/// Molmo2Model, text side only: checkpoint prefix `model.transformer.`.
final class SharpAIMolmo2Backbone: Module {
    @ModuleInfo(key: "transformer") var transformer: SharpAIMolmo2TextStack

    init(_ args: SharpAIMolmo2TextArguments) {
        _transformer.wrappedValue = SharpAIMolmo2TextStack(args)
        super.init()
    }
}

/// Text-only Molmo2 causal LM: `model.transformer.*` plus `lm_head`. Vision is not included.
final class SharpAIMolmo2ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: SharpAIMolmo2Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: SharpAIMolmo2TextArguments) {
        _model.wrappedValue = SharpAIMolmo2Backbone(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        super.init()
    }

    /// Logits of shape [B, L, vocabularySize].
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model.transformer(inputs, cache: cache))
    }
}
