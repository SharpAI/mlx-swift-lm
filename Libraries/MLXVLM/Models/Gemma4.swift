import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

private enum Gemma4Error: LocalizedError {
    case multimodalTokenCountMismatch(kind: String, featureTokens: Int, promptTokens: Int)
    case imagePlaceholderMismatch(images: Int, placeholders: Int)

    var errorDescription: String? {
        switch self {
        case .multimodalTokenCountMismatch(let kind, let featureTokens, let promptTokens):
            return
                "Gemma4 \(kind) token count mismatch: encoder produced \(featureTokens) soft tokens, but the prompt contains \(promptTokens) \(kind) tokens."
        case .imagePlaceholderMismatch(let images, let placeholders):
            return
                "Gemma4 image placeholder mismatch: the request has \(images) images but the prompt contains at least \(placeholders) image placeholders."
        }
    }
}

private func gemma4BuildLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
        + ["full_attention"]
    guard !pattern.isEmpty else { return Array(repeating: "full_attention", count: hiddenLayers) }
    var result: [String] = []
    result.reserveCapacity(hiddenLayers)
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4DefaultTextRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        "full_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional"),
        ],
        "sliding_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default"),
        ],
    ]
}

private func gemma4DefaultVisionRopeParameters() -> [String: StringOrNumber] {
    [
        "rope_theta": .float(100.0),
        "rope_type": .string("default"),
    ]
}

private func gemma4MaskedScatter(
    inputTensor: MLXArray, mask: MLXArray, source: MLXArray
) -> MLXArray {
    let flattenedInput = inputTensor.flattened()
    let flattenedMask = mask.flattened().asArray(Bool.self)
    let flattenedSource = source.flattened()

    let targetIndices = flattenedMask.enumerated().compactMap { idx, value in
        value ? Int32(idx) : nil
    }
    guard !targetIndices.isEmpty else {
        return inputTensor
    }

    guard flattenedSource.shape[0] == targetIndices.count else {
        fatalError(
            "Masked scatter shape mismatch. source=\(flattenedSource.shape[0]) mask=\(targetIndices.count)"
        )
    }

    let result = flattenedInput
    result[MLXArray(targetIndices, [targetIndices.count])] = flattenedSource
    return result.reshaped(inputTensor.shape)
}

private func gemma4OneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< numClasses)
}

private func gemma4RotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape[x.shape.count - 1] / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

private func gemma4ApplyMultiDimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float
) -> MLXArray {
    let headDim = inputs.shape[inputs.ndim - 1]
    if positions.ndim == 2 {
        let half = headDim / 2
        let freqExponents =
            (2.0 / Float(headDim)) * MLXArray(0 ..< half).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let sinusoid = positions.asType(.float32).expandedDimensions(axis: -1) / timescale
        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)
        return inputs * cosValue + gemma4RotateHalf(inputs) * sinValue
    }

    let numDimensions = positions.shape[positions.ndim - 1]
    let channelsPerDimension = 2 * (headDim / (2 * numDimensions))
    let halfPerDimension = channelsPerDimension / 2

    var parts: [MLXArray] = []
    parts.reserveCapacity(numDimensions)

    for d in 0 ..< numDimensions {
        let start = d * channelsPerDimension
        let end = start + channelsPerDimension
        let part = inputs[.ellipsis, start ..< end]

        let freqExponents =
            (2.0 / Float(channelsPerDimension)) * MLXArray(0 ..< halfPerDimension).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let dimPositions = positions[.ellipsis, d ..< d + 1].asType(.float32)
        let sinusoid = dimPositions / timescale

        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)

        parts.append(part * cosValue + gemma4RotateHalf(part) * sinValue)
    }

    return concatenated(parts, axis: -1)
}

private func gemma4EnsureFusedSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let fusedDims = [64, 80, 128]
    let d = queries.dim(queries.ndim - 1)
    let target = fusedDims.first(where: { d <= $0 }) ?? d

    if target == d {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    let paddedQueries = MLX.padded(
        queries, widths: [0, 0, 0, .init((0, target - d))])
    let paddedKeys = MLX.padded(
        keys, widths: [0, 0, 0, .init((0, target - d))])
    let paddedValues = MLX.padded(
        values, widths: [0, 0, 0, .init((0, target - d))])

    return MLXFast.scaledDotProductAttention(
        queries: paddedQueries, keys: paddedKeys, values: paddedValues, scale: scale, mask: mask
    )[.ellipsis, ..<d]
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            return keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            return keys.0.dim(-2)
        }
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4AdjustAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mask {
    case .array(let maskArray):
        guard let maskLength = maskArray.shape.last, maskLength > keyLength else {
            return mask
        }
        let start = maskLength - keyLength
        return .array(maskArray[.ellipsis, start...])
    case .arrays, .causal, .none:
        return mask
    }
}

private func gemma4TokenTypeIds(
    inputIds: MLXArray,
    imageTokenId: Int?,
    videoTokenId: Int?,
    audioTokenId: Int?
) -> MLXArray {
    var tokenTypeIds = MLXArray.zeros(like: inputIds).asType(.int32)
    if let imageTokenId {
        tokenTypeIds = MLX.where(inputIds .== imageTokenId, MLXArray(1), tokenTypeIds)
    }
    if let videoTokenId {
        tokenTypeIds = MLX.where(inputIds .== videoTokenId, MLXArray(2), tokenTypeIds)
    }
    if let audioTokenId {
        tokenTypeIds = MLX.where(inputIds .== audioTokenId, MLXArray(3), tokenTypeIds)
    }
    return tokenTypeIds
}

private func gemma4TextOnlyPromptTokens(_ input: LMInput) -> MLXArray {
    let tokens = input.text.tokens
    if tokens.ndim == 2, tokens.dim(0) == 1 {
        return tokens[0]
    }
    if tokens.ndim == 1 {
        return tokens
    }
    return tokens.flattened()
}

private func gemma4PrepareTextOnly(
    _ input: LMInput,
    cache: [any KVCache],
    prefill: PrefillParameters,
    languageModel: Gemma4TextLanguageModel
) throws -> PrepareResult {
    let y = gemma4TextOnlyPromptTokens(input).expandedDimensions(axis: 0)
    let convertedCache = cache.map { $0 }
    let totalPositions = y.dim(1)

    let processed = try prefill.forEachChunk(total: totalPositions) { range in
        _ = languageModel(y[0..., range], cache: convertedCache)
        asyncEval(cache)
    }
    if processed > 0 { eval(cache) }
    let result = languageModel(y[0..., processed...], cache: convertedCache)
    prefill.progress?(totalPositions, totalPositions)
    return .logits(result)
}

private func gemma4BlockSequenceIdsForMask(_ tokenTypeIds: MLXArray) -> MLXArray {
    let isVision = (tokenTypeIds .== 1) | (tokenTypeIds .== 2)
    let sequenceLength = isVision.dim(1)
    guard sequenceLength > 0 else {
        return MLXArray.zeros(like: tokenTypeIds).asType(.int32) - 1
    }

    let previous =
        if sequenceLength == 1 {
            MLXArray.zeros(like: isVision)
        } else {
            concatenated(
                [
                    MLXArray.zeros(like: isVision[0..., ..<1]),
                    isVision[0..., ..<(sequenceLength - 1)],
                ],
                axis: 1
            )
        }
    let starts = isVision & logicalNot(previous)
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray.zeros(like: groupIds) - 1)
}

private func gemma4ApplyBlockwiseBidirectionalOverlay(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    tokenTypeIds: MLXArray,
    sequenceLength: Int,
    windowSize: Int?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let baseMask: MLXArray
    switch mask {
    case .array(let array):
        baseMask = array
    case .arrays(let arrays):
        guard let first = arrays.first else {
            return mask
        }
        baseMask = first
    case .causal:
        baseMask = createCausalMask(n: sequenceLength, offset: 0, windowSize: windowSize)
    case .none:
        baseMask = MLXArray.ones([sequenceLength, sequenceLength], dtype: .bool)
    }

    guard tokenTypeIds.dim(1) == baseMask.dim(-1) else {
        return mask
    }

    let blockSequenceIds = gemma4BlockSequenceIdsForMask(tokenTypeIds)
    let queryBlocks = expandedDimensions(blockSequenceIds, axis: -1)
    let keyBlocks = expandedDimensions(blockSequenceIds, axis: -2)
    let sameBlock = (queryBlocks .!= -1) & (queryBlocks .== keyBlocks)
    return .array(baseMask | expandedDimensions(sameBlock, axis: 1))
}

private func gemma4CompactPrefixRows(features: MLXArray, validMask: MLXArray) -> MLXArray {
    let maskRows = validMask.asArray(Bool.self)
    let batch = validMask.dim(0)
    let length = validMask.dim(1)
    var rows: [MLXArray] = []
    rows.reserveCapacity(batch)

    for batchIdx in 0 ..< batch {
        let start = batchIdx * length
        let count = maskRows[start ..< start + length].reduce(0) { $0 + ($1 ? 1 : 0) }
        if count > 0 {
            rows.append(features[batchIdx, ..<count, 0...])
        }
    }

    guard !rows.isEmpty else {
        return features.reshaped(-1, features.dim(-1))[..<0, 0...]
    }
    return concatenated(rows, axis: 0)
}
// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let globalKVHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabularySize: Int
    public let vocabularySizePerLayerInput: Int
    public let numKVSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTraditional: Bool
    public let finalLogitSoftcapping: Float?
    public let useDoubleWideMLP: Bool
    public let enableMoEBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let attentionKEqV: Bool
    public let useBidirectionalAttention: String?
    public let layerTypes: [String]
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_text"
        let isUnified = modelType == "gemma4_unified_text" || modelType == "gemma4_unified"
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize)
            ?? (isUnified ? 3840 : 1536)
        hiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers)
            ?? (isUnified ? 48 : 35)
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize)
            ?? (isUnified ? 15_360 : 6144)
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads)
            ?? (isUnified ? 16 : 8)
        kvHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.kvHeads) ?? (isUnified ? 8 : 1)
        globalKVHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalKVHeads)
            ?? (isUnified ? 1 : nil)
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalHeadDim) ?? 512
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySize) ?? 262_144
        vocabularySizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySizePerLayerInput)
            ?? vocabularySize
        // An absent key must mean "no KV sharing": this value now decides whether K/V
        // projections are built at all, so defaulting to 20 would silently drop K/V
        // weights for the last 20 layers of any config that omits the key — and drive
        // the boundary negative for models shorter than that.
        numKVSharedLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numKVSharedLayers) ?? 0
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSizePerLayerInput)
            ?? (isUnified ? 0 : 256)
        slidingWindow =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindow)
            ?? (isUnified ? 1024 : 512)
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindowPattern)
            ?? (isUnified ? 6 : 5)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: CodingKeys.finalLogitSoftcapping) ?? 30.0
        useDoubleWideMLP =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useDoubleWideMLP) ?? true
        enableMoEBlock =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.enableMoEBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: CodingKeys.moeIntermediateSize)
        attentionKEqV =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.attentionKEqV) ?? isUnified
        useBidirectionalAttention =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.useBidirectionalAttention)
            ?? (isUnified ? "vision" : nil)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: CodingKeys.ropeParameters)
            ?? (isUnified
                ? [
                    "full_attention": [
                        "partial_rotary_factor": .float(0.25),
                        "rope_theta": .float(1_000_000.0),
                        "rope_type": .string("proportional"),
                    ],
                    "sliding_attention": [
                        "rope_theta": .float(10_000.0),
                        "rope_type": .string("default"),
                    ],
                ] : gemma4DefaultTextRopeParameters())
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: CodingKeys.layerTypes)
            ?? gemma4BuildLayerTypes(
                hiddenLayers: hiddenLayers, slidingWindowPattern: slidingWindowPattern)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
    }
}

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let rmsNormEps: Float
    public let defaultOutputLength: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let useClippedLinears: Bool
    public let standardize: Bool
    public let ropeParameters: [String: StringOrNumber]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_vision"
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers) ?? 16
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 768
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize) ?? 3072
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads) ?? 12
        keyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.keyValueHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize) ?? 16
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        defaultOutputLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.defaultOutputLength) ?? 280
        positionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.positionEmbeddingSize) ?? 10_240
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize) ?? 3
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: CodingKeys.standardize) ?? false
        ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: CodingKeys.ropeParameters)
            ?? gemma4DefaultVisionRopeParameters()
    }
}

public struct Gemma4Configuration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4VisionConfiguration
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let visionSoftTokensPerVideoFrame: Int
    public let tieWordEmbeddings: Bool
    public let audioConfig: Gemma4AudioConfiguration?

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case visionSoftTokensPerVideoFrame = "vision_soft_tokens_per_video_frame"
        case tieWordEmbeddings = "tie_word_embeddings"
        case audioConfig = "audio_config"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(
            Gemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
        visionConfiguration = try c.decode(
            Gemma4VisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId)
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId)
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage)
            ?? visionConfiguration.defaultOutputLength
        visionSoftTokensPerVideoFrame =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerVideoFrame) ?? 70
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
        audioConfig = try c.decodeIfPresent(Gemma4AudioConfiguration.self, forKey: CodingKeys.audioConfig)
    }
}

// MARK: - Text

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4RMSNormZeroShift: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

final class Gemma4TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        let isKVSharedLayer = layerIdx >= firstKVSharedLayer
        let useDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let hiddenDimensions = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4TextRouter: Module {
    let topKExperts: Int
    let config: Gemma4TextConfiguration
    private let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma4 MoE router requires numExperts and topKExperts")
        }

        self.topKExperts = topKExperts
        self.config = config
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let normed = MLXFast.rmsNorm(
            x, weight: (scale * rootSize).asType(x.dtype), eps: config.rmsNormEps)

        let scores = proj(normed)

        let topKIndices = MLX.argPartition(scores, kth: -topKExperts, axis: -1)[
            .ellipsis, (-topKExperts)...,
        ]
        var topKWeights = MLX.takeAlong(scores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

final class Gemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma4 MoE experts require numExperts and moeIntermediateSize")
        }

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK).asType(expertOutput.dtype)
        return weightedExpertSum(expertOutput, weights).reshaped(batch, length, hidden)
    }
}

private class Gemma4ScaledLinear: Module, UnaryLayer, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        super.init()
    }

    /// Adopt an already-materialised weight — used by the quantized subclass, which
    /// cannot assign the wrapped property directly.
    fileprivate init(weight: MLXArray, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = weight
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x.matmul(weight.transposed())) * scalar
    }

    // This module holds a bare MLXArray rather than wrapping a Linear, so quantize()
    // could not convert it: it stayed in float form while the checkpoint shipped it
    // quantized, and model.update failed with a packed-vs-unpacked shape mismatch
    // (`Actual [10752, 320], expected [10752, 2560]` — 4-bit packing, SwiftLM #120).
    // The quantized form subclasses this one because Module.update can only replace a
    // child with an instance of the declared property type, which is how Linear and
    // QuantizedLinear are related.
    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedGemma4ScaledLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

private final class QuantizedGemma4ScaledLinear: Gemma4ScaledLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    init(_ other: Gemma4ScaledLinear, groupSize: Int, bits: Int, mode: QuantizationMode) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        super.init(weight: quantizedWeight, scalar: other.scalar)
        // Packed integer weights must not surface as trainable parameters, matching
        // QuantizedLinear and QuantizedSwitchLinear.
        self.freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = quantizedMatmul(
            x, weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        return y * scalar
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
/// Use `kvSharedOnly: true` in the constructor to skip building local K/V
/// projections (the drafter consumes the target's K/V via `sharedKV` instead).
final class Gemma4TextAttention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let isKVSharedLayer: Bool
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale
    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim =
            layerType == "full_attention" && config.globalHeadDim > 0
            ? config.globalHeadDim : config.headDim
        self.numHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.numKVHeads =
            useKEqV ? (config.globalKVHeads ?? config.kvHeads) : config.kvHeads
        self.scale = 1.0

        // No `firstKVSharedLayer > 0` clause: with numKVSharedLayers == 0 the boundary
        // equals hiddenLayers and no layer is shared anyway, while for an assistant model
        // (hiddenLayers == numKVSharedLayers) the boundary is 0 and *no* layer owns K/V —
        // which is what the checkpoint ships. The clause made this file build K/V for
        // every layer in that case, disagreeing with MLXLLM's Gemma4Text (review
        // follow-up on #44).
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self.isKVSharedLayer = layerIdx >= firstKVSharedLayer

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        // KV-shared layers read K/V from an earlier layer's cache and ship no k_proj,
        // v_proj or k_norm weights. Creating them anyway made model.update(verify: .all)
        // demand tensors the checkpoint does not contain, so gemma-4-e4b-it-4bit
        // (num_kv_shared_layers: 18 of 42) failed to load at layer 24 (SwiftLM #120).
        if !isKVSharedLayer {
            self._kProj.wrappedValue = Linear(
                config.hiddenSize, numKVHeads * headDim, bias: false)
            if !useKEqV {
                self._vProj.wrappedValue = Linear(
                    config.hiddenSize, numKVHeads * headDim, bias: false)
            }
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)
        if !isKVSharedLayer {
            self._kNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: headDim, eps: config.rmsNormEps)
        }
        self._vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)

        let currentOffset: Int
        let kvState: Gemma4SharedKVState?

        if let sharedKV {
            currentOffset = offset ?? 0
            kvState = sharedKV
        } else {
            // KV-owning path: K/V projections must be present. If they are nil
            // here the layer is KV-shared (drafter `kvSharedOnly`, or the target's
            // shared tail `isKVSharedLayer`) and the caller forgot to pass
            // `sharedKV` — a configuration bug.
            guard let kProj, let kNorm, let vNorm else {
                fatalError(
                    "Gemma4 attention called without sharedKV on a KV-shared layer")
            }
            currentOffset = cache?.offset ?? 0
            guard let kProj, let kNorm else {
                fatalError(
                    "Gemma4 layer \(layerIdx) is KV-shared but received no shared KV state")
            }
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values =
                if useKEqV {
                    keys
                } else {
                    vProj!(x).reshaped(batch, length, numKVHeads, headDim)
                }
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            values = vNorm(values).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: currentOffset)
            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
                    keys: keys, values: values)
                kvState = .quantized(
                    keys: quantizedKeys,
                    values: quantizedValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else {
                if let cache {
                    (keys, values) = cache.update(keys: keys, values: values)
                }
                kvState = .regular(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: currentOffset)

        guard let kvState else {
            fatalError("Gemma4 attention expected a KV state")
        }
        let localMask = gemma4AdjustAttentionMask(mask, keyLength: kvState.sequenceLength)

        let output: MLXArray =
            switch kvState {
            case .regular(let keys, let values):
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: localMask
                )
            case .quantized(let keys, let values, let groupSize, let bits, let mode):
                quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keys,
                    quantizedValues: values,
                    scale: scale,
                    mask: localMask,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }

        return (
            oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)),
            kvState,
            currentOffset
        )
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4TextDecoderLayer: Module {
    let layerType: String
    let enableMoE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4TextAttention
    @ModuleInfo var mlp: Gemma4TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "router") var router: Gemma4TextRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoE = config.enableMoEBlock
        self._selfAttention.wrappedValue = Gemma4TextAttention(
            config: config, layerIdx: layerIdx, kvSharedOnly: kvSharedOnly)
        self._mlp.wrappedValue = Gemma4TextMLP(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4TextRouter(config: config)
            self._experts.wrappedValue = Gemma4TextExperts(config: config)
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attentionOutput, kvState, attentionOffset) = selfAttention(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = attentionOutput
        h = postAttentionLayerNorm(h)
        h = residual + h

        residual = h
        if enableMoE,
            let router,
            let experts,
            let postFeedforwardLayerNorm1,
            let postFeedforwardLayerNorm2,
            let preFeedforwardLayerNorm2
        {
            var dense = preFeedforwardLayerNorm(h)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let (topKIndices, topKWeights) = router(h)
            var sparse = preFeedforwardLayerNorm2(h)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = postFeedforwardLayerNorm2(sparse)

            h = dense + sparse
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let perLayerInputGate, let perLayerProjection, let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gated = perLayerInputGate(h)
            gated = geluApproximate(gated)
            gated = gated * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            h = residual + gated
        }

        return (h * layerScalar, kvState, attentionOffset)
    }
}

private final class Gemma4TextBackbone: Module, LayerPartitionable, StreamableMoE {
    let config: Gemma4TextConfiguration
    let firstKVSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    let perLayerProjectionScale: Float
    private let _perLayerInputScale: MLXArray

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4RMSNormZeroShift?

    // MARK: - LayerPartitionable
    public var gpuLayerCount: Int?
    public var totalLayerCount: Int { layers.count }

    // MARK: - StreamableMoE
    public var streamExperts: Bool = false

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIdx = config.hiddenLayers - config.numKVSharedLayers
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(max(config.hiddenSizePerLayerInput, 1)), 0.5)
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

        let concreteLayers = Array(config.layerTypes.prefix(firstKVSharedLayerIdx))
        let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention") ?? 0

        var cacheMap: [Int] = []
        cacheMap.reserveCapacity(config.hiddenLayers)
        for (idx, layerType) in config.layerTypes.enumerated() {
            if idx < firstKVSharedLayerIdx {
                cacheMap.append(idx)
            } else {
                cacheMap.append(layerType == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
            }
        }
        layerIdxToCacheIdx = cacheMap
        firstFullCacheIdx = concreteLayers.firstIndex(of: "full_attention") ?? 0
        firstSlidingCacheIdx = concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        // KV-shared tail layers (`layer_idx >= hidden_layers - num_kv_shared_layers`)
        // reuse an earlier same-type layer's K/V and own no local K/V projection or
        // K norm. Build them with `kvSharedOnly: true` so the module tree omits those
        // tensors — matching what `sanitize` drops from the checkpoint, and mirroring
        // both `Gemma4AssistantDraftInner` and upstream mlx-lm (`Attention.has_kv`).
        // Without this, shared layers declare a `v_proj`/`k_proj`/`k_norm` that no
        // weight fills, so loading a Gemma 4 checkpoint fails under the strict loader
        // verify with e.g. `keyNotFound(… layers.24.self_attn.v_proj.weight …)`.
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { idx in
            Gemma4TextDecoderLayer(
                config: config, layerIdx: idx,
                kvSharedOnly: firstKVSharedLayer > 0 && idx >= firstKVSharedLayer)
        }
        self._norm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.hiddenSizePerLayerInput > 0 {
            self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            self.perLayerProjectionScale = 1.0
        }

        super.init()
    }

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else {
            fatalError("Per-layer inputs requested for a model without embed_tokens_per_layer")
        }
        let validMask =
            logicalAnd(
                inputIds .>= 0, inputIds .< config.vocabularySizePerLayerInput)
        let tokens = MLX.where(validMask, inputIds, MLXArray.zeros(like: inputIds))
        var result = embedTokensPerLayer(tokens)
        result = (result * MLXArray(embedTokensPerLayerScale, dtype: .float32)).asType(result.dtype)
        return result.reshaped(
            Array(inputIds.shape) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        )
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            return nil
        }

        var perLayerProjection = perLayerModelProjection(inputsEmbeds) * perLayerProjectionScale
        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.hiddenLayers, config.hiddenSizePerLayerInput,
            ]
        )
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs)
            * _perLayerInputScale.asType(inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        emitDrafterState: Bool = false
    ) -> (
        hidden: MLXArray, sharedKV: [String: (MLXArray, MLXArray)]?,
        sharedKVSources: [String: Int]
    ) {
        // Tolerate callers that hand us a 1D `(L,)` token array instead
        // of the canonical 2D `(B, L)` produced by `Gemma4Processor.prepare`.
        // The downstream `perLayerInputs` indexing path (`finalPerLayerInputs[
        // 0..., 0..., idx, 0...]`) requires 4D shapes; with 1D inputs the
        // model otherwise crashes inside `MLXArray.subscript.getter`
        // → `mlx_array_dim` → `_mlx_error`. This expansion is zero-copy
        // and behaves identically when the caller already passed 2D.
        let inputs = inputs.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        let inputsEmbeds = inputsEmbeds.map {
            $0.ndim == 2 ? $0.expandedDimensions(axis: 0) : $0
        }

        let h0: MLXArray
        if let inputsEmbeds {
            h0 = inputsEmbeds
        } else if let inputs {
            let embeddings = embedTokens(inputs)
            h0 = (embeddings * MLXArray(embedScale, dtype: .float32)).asType(embeddings.dtype)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        let processedPerLayerInputs: MLXArray?
        if config.hiddenSizePerLayerInput > 0 {
            if let perLayerInputs {
                processedPerLayerInputs = perLayerInputs
            } else if let inputs {
                processedPerLayerInputs = getPerLayerInputs(inputs)
            } else {
                processedPerLayerInputs = nil
            }
        } else {
            processedPerLayerInputs = nil
        }
        let finalPerLayerInputs = projectPerLayerInputs(h0, perLayerInputs: processedPerLayerInputs)

        let localCache =
            cache ?? Array(repeating: nil as KVCache?, count: max(firstKVSharedLayerIdx, 1))
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            fullMask = mask
            slidingMask = mask
        } else {
            let tokenTypeIds = tokenTypeIds.map {
                $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0
            }
            let hasAudioTokens =
                if let tokenTypeIds {
                    (tokenTypeIds .== 3).asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let hasVisualTokens =
                if let tokenTypeIds {
                    ((tokenTypeIds .== 1) | (tokenTypeIds .== 2))
                        .asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let useBidirectionalVision =
                config.useBidirectionalAttention == "vision"
                && tokenTypeIds != nil
                && hasVisualTokens
                && !hasAudioTokens
                && h0.dim(1) > 1

            fullMask = createAttentionMask(
                h: h0,
                cache: firstFullCacheIdx < localCache.count ? localCache[firstFullCacheIdx] : nil,
                returnArray: useBidirectionalVision)
            slidingMask = createAttentionMask(
                h: h0,
                cache: firstSlidingCacheIdx < localCache.count
                    ? localCache[firstSlidingCacheIdx] : nil,
                windowSize: config.slidingWindow,
                returnArray: useBidirectionalVision
            )
            if useBidirectionalVision, let tokenTypeIds {
                fullMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    fullMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: nil
                )
                slidingMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    slidingMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: config.slidingWindow
                )
            }
        }

        var h = h0
        var intermediates = [(kv: Gemma4SharedKVState?, offset: Int?)](
            repeating: (nil, nil), count: config.hiddenLayers)
        for (idx, layer) in layers.enumerated() {
            let sourceIdx = layerIdxToCacheIdx[idx]
            let layerCache: KVCache? =
                if idx < firstKVSharedLayerIdx, sourceIdx < localCache.count {
                    localCache[sourceIdx]
                } else {
                    nil
                }
            let layerMask =
                if layer.layerType == "full_attention" {
                    fullMask
                } else {
                    slidingMask
                }
            let layerInput: MLXArray? =
                if let finalPerLayerInputs {
                    finalPerLayerInputs[0..., 0..., idx, 0...]
                } else {
                    nil
                }
            // Shared state must reach KV-shared layers on every forward, cache or not:
            // those layers no longer build their own k_proj/v_proj (they have no
            // weights for them), so a nil here is not "compute locally" but a trap.
            // intermediates[] is populated by every non-shared layer regardless of
            // whether an explicit cache was passed.
            let sharedKVForLayer: Gemma4SharedKVState? =
                idx >= firstKVSharedLayerIdx ? intermediates[sourceIdx].kv : nil
            let sharedOffsetForLayer: Int? =
                idx >= firstKVSharedLayerIdx ? intermediates[sourceIdx].offset : nil
            let (output, kvState, attentionOffset) = partitionedLayerCall(
                index: idx, gpuLayerCount: gpuLayerCount, stream: streamExperts
            ) {
                layer(
                    h,
                    mask: layerMask,
                    cache: layerCache,
                    perLayerInput: layerInput,
                    sharedKV: sharedKVForLayer,
                    offset: sharedOffsetForLayer
                )
            }
            h = output
            intermediates[idx] = (kvState, attentionOffset)
        }
        let finalHidden = norm(h)

        guard emitDrafterState else {
            return (finalHidden, nil, [:])
        }

        // Walk intermediates from the last layer backward; for each unique
        // `layer_type`, take the first `.regular` K/V encountered. Quantized
        // cases are skipped — the iterator treats absent `sharedKV` as a
        // signal to fall back to single-token generation (R8/R13 limitation,
        // documented).
        var sharedKV: [String: (MLXArray, MLXArray)] = [:]
        // Which cache entry each emitted tuple came from. The consumer reconciles the emitted
        // snapshot against the cache after a speculative commit, and it can only do that exactly
        // if it knows the entry -- a sliding layer's snapshot is bounded by its ring, a global
        // layer's is not, and the two are indistinguishable by length at the crossing.
        var sharedKVSources: [String: Int] = [:]
        var seenTypes = Set<String>()
        let targetTypes: Set<String> = ["full_attention", "sliding_attention"]
        for idx in stride(from: layers.count - 1, through: 0, by: -1) {
            let layerType = layers[idx].layerType
            guard targetTypes.contains(layerType), !seenTypes.contains(layerType) else {
                continue
            }
            if case .regular(let keys, let values) = intermediates[idx].kv {
                sharedKV[layerType] = (keys, values)
                // Recorded here rather than derived from `config.layerTypes`: the walk keeps
                // descending past a quantized entry, so which layer supplies a type is a runtime
                // fact.
                sharedKVSources[layerType] = layerIdxToCacheIdx[idx]
                seenTypes.insert(layerType)
            }
            if seenTypes == targetTypes { break }
        }
        // Treat partial coverage (e.g. only one layer_type populated, or
        // quantized cache for the other) as no-emit — iterator falls back.
        let complete = seenTypes == targetTypes
        return (finalHidden, complete ? sharedKV : nil, complete ? sharedKVSources : [:])
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift` (the MTP drafter
/// reaches `embed_tokens` / `embed_scale` / `config.layer_types` through this).
final class Gemma4TextLanguageModel: Module, KVCacheDimensionProvider {
    let config: Gemma4TextConfiguration
    let finalLogitSoftcapping: Float?

    @ModuleInfo(key: "model") var model: Gemma4TextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map { idx in
            let layerType = config.layerTypes[idx]
            if config.attentionKEqV && layerType == "full_attention" {
                return config.globalKVHeads ?? config.kvHeads
            } else {
                return config.kvHeads
            }
        }
    }

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._model.wrappedValue = Gemma4TextBackbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        return try config.layerTypes.prefix(config.hiddenLayers - config.numKVSharedLayers).map {
            layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: slidingWindow,
                usesSlidingWindow: layerType != "full_attention")
        }
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        emitDrafterState: Bool = false
    ) -> LMOutput {
        let (hidden, sharedKV, sharedKVSources) = model(
            inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache?.map { $0 as KVCache? },
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds,
            emitDrafterState: emitDrafterState
        )
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = model.embedTokens.asLinear(hidden)
        }
        let softcappedLogits: MLXArray
        if let finalLogitSoftcapping, finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            softcappedLogits = tanh(logits / scale) * scale
        } else {
            softcappedLogits = logits
        }

        guard emitDrafterState, let sharedKV else {
            return LMOutput(logits: softcappedLogits)
        }
        var state = LMOutput.State()
        state[mtpLastHiddenStatesKey] = hidden
        state[mtpSharedKVStatesKey] = sharedKV
        state[mtpSharedKVSourceIndicesKey] = sharedKVSources
        return LMOutput(logits: softcappedLogits, state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (key, value) in weights {
            if key.contains("rotary_emb") {
                continue
            }
            // Drop redundant k_proj/v_proj/k_norm for KV-shared layers: they reuse an
            // earlier layer's K/V and own no K projection or K norm, so the module tree
            // has none. QAT checkpoints already omit these; some (PTQ) checkpoints still
            // ship them, and keeping them would be an unexpected weight.
            // Scope: text backbone only — the vision/audio towers share the
            // `layers.N.self_attn.{k,v}_proj` naming, so without these guards the drop
            // would amputate tower layers >= firstKVSharedLayer.
            if firstKVSharedLayer > 0,
                !key.contains("vision_tower"),
                !key.contains("audio_tower"),
                key.contains("self_attn.k_proj")
                    || key.contains("self_attn.v_proj")
                    || key.contains("self_attn.k_norm"),
                let layerIdx = Self.decoderLayerIndex(in: key),
                layerIdx >= firstKVSharedLayer
            {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("language_model."),
                !newKey.hasPrefix("language_model.model."),
                !newKey.hasPrefix("language_model.lm_head.")
            {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model.\(rest)"
            }

            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(
                    of: ".experts.down_proj",
                    with: ".experts.switch_glu.down_proj.weight"
                )
            }

            if newKey.hasSuffix(".experts.gate_up_proj") {
                let mid = value.dim(-2) / 2
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = value[.ellipsis, ..<mid, 0...]
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = value[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[newKey] = value
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("language_model.lm_head.")
            }
        } else if sanitized["language_model.lm_head.weight"] == nil,
            let embedWeight = sanitized["language_model.model.embed_tokens.weight"]
        {
            sanitized["language_model.lm_head.weight"] = embedWeight
        }

        return sanitized
    }

    /// Extract `N` from a weight key shaped like `…layers.N.…`, else nil.
    private static func decoderLayerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }
}

// MARK: - Vision

private final class Gemma4ClippableLinear: Module, UnaryLayer {
    let useClipping: Bool

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)
        if useClipping {
            self._inputMin.wrappedValue = MLXArray(-Float.infinity)
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(-Float.infinity)
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let clippedInput =
            if let inputMin, let inputMax {
                clip(x, min: inputMin, max: inputMax)
            } else {
                x
            }
        let projected = linear(clippedInput)
        if let outputMin, let outputMax {
            return clip(projected, min: outputMin, max: outputMax)
        }
        return projected
    }
}

private final class Gemma4VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        let normalized = xFloat * rsqrt(variance + eps)
        return (normalized * weight.asType(.float32)).asType(x.dtype)
    }
}

private final class Gemma4VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        return (xFloat * rsqrt(variance + eps)).asType(x.dtype)
    }
}

private final class Gemma4VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: Gemma4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: Gemma4ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: Gemma4VisionRMSNormNoScale

    init(config: Gemma4VisionConfiguration) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.keyValueHeads
        self.headDim = config.headDim
        self.hiddenSize = config.hiddenSize
        self.ropeBaseFrequency = config.ropeParameters["rope_theta"]?.asFloat() ?? 100.0

        self._qProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._kProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._vProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._oProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: numHeads * headDim,
            outFeatures: hiddenSize,
            useClipping: config.useClippedLinears
        )
        self._qNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma4VisionRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
        var values = vProj(x).reshaped(batch, length, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)
        values = vNorm(values)

        queries = gemma4ApplyMultiDimensionalRoPE(
            queries, positions: positions, baseFrequency: ropeBaseFrequency)
        keys = gemma4ApplyMultiDimensionalRoPE(
            keys, positions: positions, baseFrequency: ropeBaseFrequency)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let mask {
                .array(mask)
            } else {
                .none
            }
        let output = gemma4EnsureFusedSDPA(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

private final class Gemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4ClippableLinear

    init(config: Gemma4VisionConfiguration) {
        self._gateProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._upProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._downProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.intermediateSize,
            outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private final class Gemma4VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo var mlp: Gemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift

    init(config: Gemma4VisionConfiguration) {
        self._selfAttention.wrappedValue = Gemma4VisionAttention(config: config)
        self._mlp.wrappedValue = Gemma4VisionMLP(config: config)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attentionOutput = selfAttention(normed, positions: positions, mask: mask)
        let h = x + postAttentionLayerNorm(attentionOutput)
        let ff = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(ff)
    }
}

private final class Gemma4VisionPatchEmbedder: Module {
    let patchSize: Int
    let hiddenSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProjection: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(config: Gemma4VisionConfiguration) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize
        self.positionEmbeddingSize = config.positionEmbeddingSize
        self._inputProjection.wrappedValue = Linear(
            3 * patchSize * patchSize, hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = MLXArray.ones([
            2, positionEmbeddingSize, hiddenSize,
        ])
        super.init()
    }

    private func patchify(_ pixelValues: MLXArray) -> MLXArray {
        let (batch, channels, height, width) = (
            pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2), pixelValues.dim(3)
        )
        let patchesH = height / patchSize
        let patchesW = width / patchSize

        var patches = pixelValues.reshaped(
            batch, channels, patchesH, patchSize, patchesW, patchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(batch, patchesH * patchesW, channels * patchSize * patchSize)
        patches = 2 * (patches - 0.5)
        return inputProjection(patches.asType(inputProjection.weight.dtype))
    }

    func callAsFunction(
        _ pixelValues: MLXArray, patchPositions: MLXArray
    ) -> MLXArray {
        let hiddenStates = patchify(pixelValues)
        let batch = patchPositions.dim(0)
        let seqLen = patchPositions.dim(1)

        let xIndices = patchPositions[0..., 0..., 0].flattened().asType(.int32)
        let yIndices = patchPositions[0..., 0..., 1].flattened().asType(.int32)
        let xEmbeddings = take(positionEmbeddingTable[0], xIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        let yEmbeddings = take(positionEmbeddingTable[1], yIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        return hiddenStates + xEmbeddings + yEmbeddings
    }
}

private final class Gemma4VisionPooler: Module {
    let hiddenSize: Int
    let rootHiddenSize: Float

    init(config: Gemma4VisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.rootHiddenSize = pow(Float(config.hiddenSize), 0.5)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        patchPositions: MLXArray,
        patchesW: Int,
        outputLength: Int
    ) -> MLXArray {
        let scale = MLXArray(rootHiddenSize, dtype: hiddenStates.dtype)
        let numPatches = hiddenStates.dim(1)
        let length = max(outputLength, 1)
        if numPatches <= length {
            return hiddenStates * scale
        }

        // All batch rows share one position grid, so a single [l, L] weight
        // matrix pools every row. The processor's resize keeps both sides
        // divisible by kernel * patchSize, so the pooled grid covers the
        // image exactly.
        let positions = patchPositions[0]
        let kernel = max(Int(sqrt(Double(numPatches / length))), 1)
        let divisor = kernel * kernel

        let kernelIndices = floor(positions.asType(.float32) / Float(kernel)).asType(.int32)
        let flatKernel =
            kernelIndices[0..., 0] + MLXArray(Int32(max(patchesW / kernel, 1)))
            * kernelIndices[0..., 1]
        let weights =
            gemma4OneHot(flatKernel, numClasses: length).asType(.float32)
            / Float(divisor)
        let output = einsum("lL,bld->bLd", weights, hiddenStates)
            .asType(hiddenStates.dtype)
        return output * scale
    }
}

private final class Gemma4VisionTransformerModel: Module {
    @ModuleInfo(key: "layers") var layers: [Gemma4VisionTransformerBlock]

    init(config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4VisionTransformerBlock(config: config)
        }
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray
    {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

private final class Gemma4VisionModel: Module {
    let config: Gemma4VisionConfiguration
    let patchSize: Int
    let poolingKernelSize: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: Gemma4VisionTransformerModel
    @ModuleInfo(key: "pooler") var pooler: Gemma4VisionPooler
    @ModuleInfo(key: "std_bias") var standardizationBias: MLXArray?
    @ModuleInfo(key: "std_scale") var standardizationScale: MLXArray?

    init(config: Gemma4VisionConfiguration) {
        self.config = config
        self.patchSize = config.patchSize
        self.poolingKernelSize = config.poolingKernelSize
        self._patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config: config)
        self._encoder.wrappedValue = Gemma4VisionTransformerModel(config: config)
        self._pooler.wrappedValue = Gemma4VisionPooler(config: config)
        if config.standardize {
            self._standardizationBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._standardizationScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
        super.init()
    }

    private func patchPositions(batch: Int, patchesH: Int, patchesW: Int) -> MLXArray {
        // .xy indexing makes x vary fastest, matching the row-major patch
        // order the embedder and pooler expect.
        let grids = meshGrid([
            MLXArray.arange(patchesW, dtype: .int32),
            MLXArray.arange(patchesH, dtype: .int32),
        ])
        let positions = stacked([grids[0].flattened(), grids[1].flattened()], axis: 1)
            .reshaped(1, patchesH * patchesW, 2)
        return batch == 1
            ? positions
            : broadcast(positions, to: [batch, patchesH * patchesW, 2])
    }

    /// Encodes a batch of same-sized images. Every patch is real (callers
    /// slice padded canvases down to each image's true size first), so
    /// attention is dense and the pooled output length falls out of the
    /// patch grid: numPatches / poolingKernelSize².
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let pixels =
            if pixelValues.ndim == 3 {
                expandedDimensions(pixelValues, axis: 0)
            } else {
                pixelValues
            }
        let batch = pixels.dim(0)
        let patchesH = pixels.dim(2) / patchSize
        let patchesW = pixels.dim(3) / patchSize
        let numPatches = patchesH * patchesW
        let outputLength = max(numPatches / (poolingKernelSize * poolingKernelSize), 1)

        let patchPositions = patchPositions(batch: batch, patchesH: patchesH, patchesW: patchesW)
        var hiddenStates = patchEmbedder(pixels, patchPositions: patchPositions)
        hiddenStates = encoder(hiddenStates, positions: patchPositions, mask: nil)
        hiddenStates = pooler(
            hiddenStates, patchPositions: patchPositions, patchesW: patchesW,
            outputLength: outputLength)

        if let standardizationBias, let standardizationScale {
            hiddenStates = (hiddenStates - standardizationBias) * standardizationScale
        }
        return hiddenStates
    }
}

private final class Gemma4MultimodalEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm:
        Gemma4RMSNormNoScale

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = Gemma4RMSNormNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(embeddingPreProjectionNorm(x))
    }
}

// MARK: - Model

public final class Gemma4: Module, VLMModel, KVCacheDimensionProvider, StreamableMoE {
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    /// Module-internal — also reached by `Gemma4Assistant.swift` (drafter `bind()`
    /// walks here to cache the target's input embeddings, embed scale, and
    /// per-layer type metadata).
    @ModuleInfo(key: "language_model") var languageModel: Gemma4TextLanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder

    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma4AudioModel?
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma4MultimodalEmbedder?

    public let config: Gemma4Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    // MARK: - StreamableMoE (delegates to the backbone)
    public var streamExperts: Bool {
        get { languageModel.model.streamExperts }
        set { languageModel.model.streamExperts = newValue }
    }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self._visionTower.wrappedValue = Gemma4VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = Gemma4TextLanguageModel(config.textConfiguration)
        self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            embeddingDim: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            eps: config.visionConfiguration.rmsNormEps
        )
        if let acfg = config.audioConfig {
            self._audioTower.wrappedValue = Gemma4AudioModel(config: acfg)
            self._embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: acfg.outputProjDims,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: config.visionConfiguration.rmsNormEps
            )
        }
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        audioValues: MLXArray? = nil,
        audioMask: MLXArray? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray? = nil
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            let imageMask = inputIds .== config.imageTokenId
            let audioMask =
                if let audioTokenId = config.audioTokenId {
                    inputIds .== audioTokenId
                } else {
                    MLXArray.zeros(like: imageMask)
                }
            let textMask = logicalNot(logicalOr(imageMask, audioMask))
            let perLayerTokens = MLX.where(textMask, inputIds, MLXArray.zeros(like: inputIds))
            let perLayerInputsResult = languageModel.model.getPerLayerInputs(perLayerTokens)
            
            // Mask out the PLE for multimodal tokens to preserve original variance
            var multimodalMaskExpanded = logicalNot(textMask)
            multimodalMaskExpanded = multimodalMaskExpanded.reshaped(multimodalMaskExpanded.dim(0), multimodalMaskExpanded.dim(1), 1, 1)
            perLayerInputs = MLX.where(multimodalMaskExpanded, MLXArray.zeros(like: perLayerInputsResult), perLayerInputsResult)
        }

        guard let pixelValues else {
            if let audioValues = audioValues, let audioTower = audioTower, let embedAudio = embedAudio, let audioTokenId = config.audioTokenId {
                let actualAudioMask = audioMask ?? MLXArray.ones(audioValues.shape[0..<2], dtype: .bool)
                print("[ALM Audio] audioValues=\(audioValues.shape) mask=\(actualAudioMask.shape) maskSum=\(actualAudioMask.asType(.int32).sum().item(Int.self))")
                let (audioOutputs, _) = audioTower(audioValues, mask: actualAudioMask)
                eval(audioOutputs)
                let audioFeatures = embedAudio(audioOutputs).asType(inputsEmbeds.dtype)
                eval(audioFeatures)

                let audioTokenMask = inputIds .== audioTokenId
                let audioTokenCount = audioTokenMask.asType(.int32).sum().item(Int.self)
                let audioFeatureCount = audioFeatures.dim(1)
                print("[ALM Audio] audioOutputs=\(audioOutputs.shape) mean=\(audioOutputs.mean().item(Float.self)) tokenCount=\(audioTokenCount) featureCount=\(audioFeatureCount)")
                guard audioTokenCount == audioFeatureCount else {
                    print("[Gemma4] Audio token count mismatch: \(audioTokenCount) tokens vs \(audioFeatureCount) features. Skipping.")
                    return (inputsEmbeds, perLayerInputs)
                }

                let firstAudioPos = MLX.argMax(audioTokenMask[0].asType(.int32), axis: 0).item(Int.self)
                let embedBefore = inputsEmbeds[0, firstAudioPos, 0].item(Float.self)

                var audioMaskExpanded = expandedDimensions(audioTokenMask, axis: -1)
                audioMaskExpanded = broadcast(audioMaskExpanded, to: inputsEmbeds.shape)
                inputsEmbeds = gemma4MaskedScatter(
                    inputTensor: inputsEmbeds,
                    mask: audioMaskExpanded,
                    source: audioFeatures
                )
                eval(inputsEmbeds)
                let embedAfter = inputsEmbeds[0, firstAudioPos, 0].item(Float.self)
                print("[ALM Audio] scatter: pos=\(firstAudioPos) before=\(embedBefore) after=\(embedAfter) changed=\(embedBefore != embedAfter)")
            }
            return (inputsEmbeds, perLayerInputs)
        }

        // Gemma 4 has no separate video encoder — each video frame runs through the
        // same vision tower as images (producing `visionSoftTokensPerImage` pooled
        // tokens per frame) and is then truncated to the smaller per-frame video
        // budget before scattering onto the `<video>` soft-token positions. Video
        // frames come from the processor at a uniform size, so they don't need the
        // per-image aspect-preserving slicing above.
        if let video, let videoTokenId = config.videoTokenId {
            inputsEmbeds = try scatterVideoFeatures(
                into: inputsEmbeds, inputIds: inputIds, videoPixelValues: video.pixels,
                tokenId: videoTokenId, softTokensPerFrame: config.visionSoftTokensPerVideoFrame)
        }

        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)
        inputsEmbeds = gemma4MaskedScatter(
            inputTensor: inputsEmbeds,
            mask: imageMaskExpanded,
            source: imageFeatures
        )

        if let audioValues = audioValues, let audioTower = audioTower, let embedAudio = embedAudio, let audioTokenId = config.audioTokenId {
            let actualAudioMask = audioMask ?? MLXArray.ones(audioValues.shape[0..<2], dtype: .bool)
            print("[Omni Audio] audioValues=\(audioValues.shape) mask=\(actualAudioMask.shape) maskSum=\(actualAudioMask.asType(.int32).sum().item(Int.self))")
            let (audioOutputs, _) = audioTower(audioValues, mask: actualAudioMask)
            eval(audioOutputs)
            let audioFeatures = embedAudio(audioOutputs).asType(inputsEmbeds.dtype)
            eval(audioFeatures)

            let audioTokenMask = inputIds .== audioTokenId
            let audioTokenCount = audioTokenMask.asType(.int32).sum().item(Int.self)
            let audioFeatureCount = audioFeatures.dim(1)
            let audioStd = MLX.sqrt(MLX.variance(audioOutputs)).item(Float.self)
            let audioAbsMean = MLX.abs(audioOutputs).mean().item(Float.self)
            print("[Omni Audio] audioOutputs=\(audioOutputs.shape) mean=\(audioOutputs.mean().item(Float.self)) std=\(audioStd) absMean=\(audioAbsMean) tokenCount=\(audioTokenCount) featureCount=\(audioFeatureCount)")
            guard audioTokenCount == audioFeatureCount else {
                print("[Gemma4] Omni audio token count mismatch: \(audioTokenCount) tokens vs \(audioFeatureCount) features. Skipping.")
                return (inputsEmbeds, perLayerInputs)
            }

            // Sample embed value at first audio token position BEFORE scatter
            let firstAudioPos = MLX.argMax(audioTokenMask[0].asType(.int32), axis: 0).item(Int.self)
            let embedBefore = inputsEmbeds[0, firstAudioPos, 0].item(Float.self)

            var audioMaskExpanded = expandedDimensions(audioTokenMask, axis: -1)
            audioMaskExpanded = broadcast(audioMaskExpanded, to: inputsEmbeds.shape)
            inputsEmbeds = gemma4MaskedScatter(
                inputTensor: inputsEmbeds,
                mask: audioMaskExpanded,
                source: audioFeatures
            )
            eval(inputsEmbeds)
            let embedAfter = inputsEmbeds[0, firstAudioPos, 0].item(Float.self)
            print("[Omni Audio] scatter: pos=\(firstAudioPos) before=\(embedBefore) after=\(embedAfter) changed=\(embedBefore != embedAfter)")
        }

        return (inputsEmbeds, perLayerInputs)
    }

    /// Encode video frames (`[numFrames, C, H, W]`) through the shared vision tower,
    /// keep the first `softTokensPerFrame` pooled tokens of each frame, and scatter the
    /// resulting `numFrames * softTokensPerFrame` soft tokens onto the `tokenId`
    /// positions. Gemma 4 gives video frames a smaller token budget than full images;
    /// the processor resizes frames so those leading tokens carry the frame's content.
    private func scatterVideoFeatures(
        into inputsEmbeds: MLXArray,
        inputIds: MLXArray,
        videoPixelValues: MLXArray,
        tokenId: Int,
        softTokensPerFrame: Int
    ) throws -> MLXArray {
        var features = visionTower(videoPixelValues)
        features = embedVision(features)
        let cap = min(softTokensPerFrame, features.dim(1))
        features = features[0..., 0 ..< cap, 0...]
        features = features.asType(inputsEmbeds.dtype)

        let producedTokens = features.dim(0) * features.dim(1)
        let tokenMask = inputIds .== tokenId
        let expectedTokens = tokenMask.asType(.int32).sum().item(Int.self)
        if expectedTokens != producedTokens {
            throw Gemma4Error.multimodalTokenCountMismatch(
                kind: "video", featureTokens: producedTokens, promptTokens: expectedTokens)
        }

        var maskExpanded = expandedDimensions(tokenMask, axis: -1)
        maskExpanded = broadcast(maskExpanded, to: inputsEmbeds.shape)
        return gemma4MaskedScatter(
            inputTensor: inputsEmbeds, mask: maskExpanded, source: features)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        let convertedCache = cache.map { $0 }
        let hasImage = input.image?.pixels != nil
        let hasAudio = input.audio?.features != nil
        print("[Gemma4 prepare] hasImage=\(hasImage) hasAudio=\(hasAudio) audioTower=\(audioTower != nil) embedAudio=\(embedAudio != nil) audioTokenId=\(String(describing: config.audioTokenId))")
        if hasImage || hasAudio {
            print("[Gemma4 prepare] → multimodal path: inputIds.shape=\(input.text.tokens.shape)")
            if hasAudio {
                print("[Gemma4 prepare] → audio features shape=\(input.audio!.features.shape)")
            }
            let (inputsEmbeds, perLayerInputs) = try getInputEmbeddings(
                inputIds: input.text.tokens, 
                pixelValues: input.image?.pixels, 
                audioValues: input.audio?.features,
                audioMask: input.audio?.mask
            )
            let result = languageModel(
                nil,
                cache: convertedCache,
                inputsEmbeds: inputsEmbeds,
                perLayerInputs: perLayerInputs,
                tokenTypeIds: gemma4TokenTypeIds(
                    inputIds: input.text.tokens,
                    imageTokenId: config.imageTokenId,
                    videoTokenId: config.videoTokenId,
                    audioTokenId: config.audioTokenId)
            )
            let total = inputsEmbeds.dim(1)
            prefill.progress?(total, total)
            return .logits(result)
        } else {
            print("[Gemma4 prepare] → text-only path")
            let result = languageModel(input.text.tokens, cache: convertedCache)
            return .logits(result)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    /// MTP-aware `LanguageModel` entry point. Reads `mtpEmitFlagKey` from
    /// the incoming `state` and threads it through to `Gemma4TextLanguageModel`;
    /// the returned `LMOutput` carries `mtpLastHiddenStatesKey` and
    /// `mtpSharedKVStatesKey` populated when the flag is set, empty otherwise.
    /// Overrides the protocol-extension default at `LanguageModel` which
    /// would discard `state`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emit = state?[mtpEmitFlagKey] ?? false
        return languageModel(
            input.tokens, cache: cache?.map { $0 },
            emitDrafterState: emit
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = languageModel.sanitize(weights: weights)

        if !config.visionConfiguration.useClippedLinears {
            sanitized = sanitized.filter { key, _ in
                !key.contains("input_min")
                    && !key.contains("input_max")
                    && !key.contains("output_min")
                    && !key.contains("output_max")
            }
        }
        
        var finalSanitized: [String: MLXArray] = [:]
        for (key, value) in sanitized {
            var newKey = key
            if newKey.contains("subsampling_conv.layers.") {
                newKey = newKey.replacingOccurrences(of: "subsampling_conv.layers.0", with: "subsample_conv_projection.layer0")
                newKey = newKey.replacingOccurrences(of: "subsampling_conv.layers.1", with: "subsample_conv_projection.layer1")
            }
            finalSanitized[newKey] = value
        }

        return finalSanitized
    }
}

// MARK: - Message Generator

/// Message generator for Gemma4/Gemma3n that places images BEFORE text in the content
/// array, matching the Python reference `apply_chat_template(image_first=True)` behaviour.
/// The Gemma4 Jinja template processes content in array order, so ordering matters:
///   Python:  [image, text, audio]  →  <|image|>\ntext<|audio|>
///   (Wrong): [text, image, audio]  →  text<|image|><|audio|>  (model ignores audio)
public struct Gemma4MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        // System/assistant/tool roles: no media, use plain string content
        if message.role != .user || (message.images.isEmpty && message.audio.isEmpty) {
            var dict: [String: any Sendable] = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if let toolCalls = message.toolCalls { dict["tool_calls"] = toolCalls }
            if let toolCallId = message.toolCallId { dict["tool_call_id"] = toolCallId }
            return dict
        }
        // User messages with media: images FIRST, then text, then audio
        // This matches the Python: apply_chat_template(..., image_first=True)
        var content: [[String: any Sendable]] = []
        content += message.images.map { _ in ["type": "image"] as [String: any Sendable] }
        content.append(["type": "text", "text": message.content])
        content += message.audio.map { _ in ["type": "audio"] as [String: any Sendable] }
        return [
            "role": message.role.rawValue,
            "content": content,
        ]
    }
}

// MARK: - Processor

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processedImage = MediaProcessing.apply(image, processing: processing)
        let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
        let targetSize = config.aspectPreservingTargetSize(for: srgbImage.extent.size)
        let resizedImage =
            srgbImage.extent.size == targetSize
            ? srgbImage
            : MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
        let finalImage =
            if config.doNormalize {
                MediaProcessing.normalize(
                    resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
            } else {
                resizedImage
            }
        let pixelValues = MediaProcessing.asMLXArray(finalImage)

        return (pixelValues, THW(1, Int(targetSize.height), Int(targetSize.width)))
    }

    /// Sample and preprocess the frames of each video into a single
    /// `[totalFrames, C, H, W]` pixel tensor (frames from all videos concatenated),
    /// plus the per-video frame count used to expand the `<video>` placeholders.
    /// Frames are resized to `config.videoFixedSize` so the vision tower's leading
    /// pooled tokens fit the per-frame video budget.
    public func processVideos(_ videos: [UserInput.Video], processing: UserInput.Processing?)
        async throws -> (pixels: MLXArray, frameCounts: [Int])
    {
        let targetSize = config.videoFixedSize
        var allFrames: [MLXArray] = []
        var frameCounts: [Int] = []
        for video in videos {
            let sequence = try await MediaProcessing.asProcessedSequence(
                video,
                processing: processing?.video ?? .init(),
                targetFPS: { _ in 1.0 },
                maxFrames: config.videoMaxFrames
            ) { frame in
                var userProcessing = processing ?? UserInput.Processing()
                userProcessing.resize = targetSize
                var image = MediaProcessing.apply(
                    try frame.image.asCIImage(), processing: userProcessing)
                image = MediaProcessing.inSRGBToneCurveSpace(image)
                image = MediaProcessing.resampleBicubic(image, to: targetSize)
                if config.doNormalize {
                    image = MediaProcessing.normalize(
                        image, mean: config.imageMeanTuple, std: config.imageStdTuple)
                }
                return VideoFrame(image: .ciImage(image), timeStamp: frame.timeStamp)
            }
            allFrames.append(contentsOf: sequence.frames)
            frameCounts.append(sequence.frames.count)
        }
        return (concatenated(allFrames), frameCounts)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)
        
        print("[Omni Debug] Full decoded prompt: \(tokenizer.decode(tokenIds: promptTokens))")

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(image: $0.asCIImage(), processing: input.processing)
            }
            let frames = imagePixelsAndFrames.map { $0.1 }

            // Each image keeps its own aspect-preserving size. ProcessedImage
            // carries one array, so zero-pad every image onto the largest
            // canvas in the request; the model slices the real regions back
            // out using frames.
            let maxHeight = frames.map(\.h).max() ?? 0
            let maxWidth = frames.map(\.w).max() ?? 0
            let paddedPixels = imagePixelsAndFrames.map { pixels, frame in
                frame.h == maxHeight && frame.w == maxWidth
                    ? pixels
                    : MLX.padded(
                        pixels,
                        widths: [
                            0, 0, .init((0, maxHeight - frame.h)), .init((0, maxWidth - frame.w)),
                        ])
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(paddedPixels), frames: frames)

            // Expand the i-th image placeholder to that image's soft token
            // count: numPatches / poolingKernelSize².
            let softTokenCounts = frames.map { config.softTokenCount(height: $0.h, width: $0.w) }
            var expandedTokens: [Int] = []
            var imageIndex = 0
            for token in promptTokens {
                if token == config.imageTokenId {
                    guard imageIndex < softTokenCounts.count else {
                        throw Gemma4Error.imagePlaceholderMismatch(
                            images: softTokenCounts.count, placeholders: imageIndex + 1)
                    }
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: config.imageTokenId,
                            count: softTokenCounts[imageIndex]))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty, let videoTokenId = config.videoTokenId {
            let (videoPixels, frameCounts) = try await processVideos(
                input.videos, processing: input.processing)
            processedVideo = LMInput.ProcessedVideo(pixels: videoPixels)

            // Expand the i-th `<video>` placeholder into one block per sampled frame:
            // BOI + video_token * videoSoftTokensPerFrame + EOI. The model produces the
            // matching count (frames * videoSoftTokensPerFrame) from `videoPixels`.
            var expandedTokens: [Int] = []
            var videoIndex = 0
            for token in promptTokens {
                if token == videoTokenId {
                    let frames = videoIndex < frameCounts.count ? frameCounts[videoIndex] : 0
                    for _ in 0 ..< frames {
                        expandedTokens.append(config.boiTokenId)
                        expandedTokens.append(
                            contentsOf: Array(
                                repeating: videoTokenId, count: config.videoSoftTokensPerFrame))
                        if let eoiTokenId = config.eoiTokenId {
                            expandedTokens.append(eoiTokenId)
                        }
                    }
                    videoIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        var processedAudio: LMInput.ProcessedAudio? = nil
        if let audioInput = input.audio.first {
            let samples = try MediaProcessing.extractAudioSamples(from: audioInput)
            print("[Omni Debug] Extracted \(samples.count) float samples. Min: \(samples.min() ?? 0.0), Max: \(samples.max() ?? 0.0)")
            let processor = Gemma4AudioFeatureExtractor(
                featureSize: 128,
                samplingRate: 16000,
                frameLengthMs: 20.0,
                hopLengthMs: 10.0,
                melFloor: 1e-3
            )
            let (features, mask) = processor.extract(waveform: samples)
            print("[Omni Debug] Extracted \(features.shape) features. Mean: \(features.mean().item(Float.self)), Min: \(features.min().item(Float.self)), Max: \(features.max().item(Float.self))")
            
            // Expected audio token count: mirrors the audio tower's two stride-2 conv2d layers,
            // each with 1-sample symmetric padding (MLX.padded [1,1] in time axis).
            // Layer formula: ceil(T/2) applied twice → ceil(ceil(numMelFrames/2)/2).
            // This avoids off-by-one vs. duration/40ms approximation.
            let numMelFrames = features.dim(1)
            let afterLayer0 = (numMelFrames + 1) / 2  // ceil(T/2)
            var expectedAudioTokens = (afterLayer0 + 1) / 2  // ceil(L1/2)
            expectedAudioTokens = min(expectedAudioTokens, 750)  // audio_seq_length cap
            
            let seqLength = features.dim(1)
            processedAudio = LMInput.ProcessedAudio(features: features, mask: mask, seqLengths: [seqLength])
            
            let audioTokenId = config.audioTokenId ?? 258881
            let gemmaBoa = 256000 // <|audio|>
            let gemmaEoa = 258883 // <audio|>
            
            var audioPadding = [gemmaBoa]
            audioPadding.append(contentsOf: Array(repeating: audioTokenId, count: expectedAudioTokens))
            audioPadding.append(gemmaEoa)
            
            let targetIdx = promptTokens.firstIndex(of: gemmaBoa) ?? promptTokens.firstIndex(of: audioTokenId)
            print("[Omni Debug] Target Index found at: \(String(describing: targetIdx)), gemmaBoa: \(gemmaBoa), audioTokenId: \(audioTokenId)")
            
            var expandedTokens = promptTokens
            expandedTokens.removeAll(where: { $0 == gemmaBoa || $0 == gemmaEoa || $0 == audioTokenId })
            
            if let insertIdx = targetIdx {
                print("[Omni Debug] Inserting audioPadding (\(audioPadding.count) tokens) at \(insertIdx)")
                expandedTokens.insert(contentsOf: audioPadding, at: insertIdx)
            } else {
                print("[Omni Debug] Inserting audioPadding (\(audioPadding.count) tokens) at 0")
                expandedTokens.insert(contentsOf: audioPadding, at: 0)
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: processedImage, audio: processedAudio)
    }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let maxSoftTokens: Int
    public let patchSize: Int
    public let poolingKernelSize: Int

    public let imageTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let boaTokenId: Int
    public let eoaTokenId: Int
    public let audioTokenId: Int?

    public let videoTokenId: Int?
    public let videoSoftTokensPerFrame: Int
    public let videoMaxFrames: Int

    /// Image keys nested under `image_processor` in processor_config.json.
    /// Repos that ship a flat preprocessor_config.json put the same keys at
    /// the top level, which wins when both are present.
    private struct ImageProcessorConfiguration: Codable {
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let imageSeqLength: Int?
        let maxSoftTokens: Int?
        let patchSize: Int?
        let poolingKernelSize: Int?

        enum CodingKeys: String, CodingKey {
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case imageSeqLength = "image_seq_length"
            case maxSoftTokens = "max_soft_tokens"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case maxSoftTokens = "max_soft_tokens"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case imageProcessor = "image_processor"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case videoTokenId = "video_token_id"
        case videoSoftTokensPerFrame = "video_soft_tokens_per_frame"
        case videoMaxFrames = "video_max_frames"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.imageProcessor)
        processorClass = try c.decode(String.self, forKey: CodingKeys.processorClass)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize)
            ?? nested?.doNormalize ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean)
            ?? nested?.imageMean ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd)
            ?? nested?.imageStd ?? [0.5, 0.5, 0.5]
        imageSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength)
            ?? nested?.imageSeqLength ?? 280
        maxSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxSoftTokens)
            ?? nested?.maxSoftTokens ?? 280
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize)
            ?? nested?.patchSize ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize)
            ?? nested?.poolingKernelSize ?? 3
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        videoSoftTokensPerFrame =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoSoftTokensPerFrame) ?? 70
        videoMaxFrames = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoMaxFrames) ?? 32
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(processorClass, forKey: CodingKeys.processorClass)
        try c.encode(doNormalize, forKey: CodingKeys.doNormalize)
        try c.encode(imageMean, forKey: CodingKeys.imageMean)
        try c.encode(imageStd, forKey: CodingKeys.imageStd)
        try c.encode(imageSeqLength, forKey: CodingKeys.imageSeqLength)
        try c.encode(maxSoftTokens, forKey: CodingKeys.maxSoftTokens)
        try c.encode(patchSize, forKey: CodingKeys.patchSize)
        try c.encode(poolingKernelSize, forKey: CodingKeys.poolingKernelSize)
        try c.encode(imageTokenId, forKey: CodingKeys.imageTokenId)
        try c.encode(boiTokenId, forKey: CodingKeys.boiTokenId)
        try c.encodeIfPresent(eoiTokenId, forKey: CodingKeys.eoiTokenId)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    /// Soft tokens the vision tower produces for an image of the given
    /// (already resized) pixel dimensions.
    public func softTokenCount(height: Int, width: Int) -> Int {
        ((height / patchSize) * (width / patchSize)) / (poolingKernelSize * poolingKernelSize)
    }

    /// Port of the Python Gemma4ImageProcessor's aspect-ratio preserving
    /// resize: the largest dimensions that (a) stay within the patch budget
    /// maxSoftTokens * poolingKernelSize², and (b) keep both sides divisible
    /// by poolingKernelSize * patchSize, so the pooling kernel is exact and
    /// the pooled grid covers the image fully at any aspect ratio.
    ///
    /// Note the config's `size` entry is deliberately ignored, as in the
    /// Python reference — models ship a vestigial 224x224 there.
    public func aspectPreservingTargetSize(for imageSize: CGSize) -> CGSize {
        let kernelArea = poolingKernelSize * poolingKernelSize
        let maxPatches = maxSoftTokens * kernelArea
        let sideMultiple = poolingKernelSize * patchSize
        let height = Double(imageSize.height)
        let width = Double(imageSize.width)

        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / max(height * width, 1)).squareRoot()
        var targetHeight = Int((factor * height / Double(sideMultiple)).rounded(.down))
        var targetWidth = Int((factor * width / Double(sideMultiple)).rounded(.down))

        // One side can floor to zero for extreme aspect ratios (both cannot:
        // their product is pinned near maxPatches, far above 1). Clamp it to
        // one pooling cell and cap the long side at the full token budget.
        let maxSideLength = maxSoftTokens
        if targetHeight == 0 {
            targetHeight = 1
            targetWidth = min(Int((width / max(height, 1)).rounded(.down)), maxSideLength)
            targetWidth = max(targetWidth, 1)
        } else if targetWidth == 0 {
            targetWidth = 1
            targetHeight = min(Int((height / max(width, 1)).rounded(.down)), maxSideLength)
            targetHeight = max(targetHeight, 1)
        }

        return CGSize(
            width: targetWidth * sideMultiple, height: targetHeight * sideMultiple)
    }

    /// Video frames use a smaller square (a multiple of patch_size * pooling_kernel_size
    /// = 48) so the vision tower's leading pooled tokens cover the frame within the
    /// ~70-token video budget: 432 → 27x27 patches → 81 pooled tokens, trimmed to
    /// `visionSoftTokensPerVideoFrame` (70) in the model.
    public var videoFixedSize: CGSize { CGSize(width: 432, height: 432) }
}

public struct Gemma4UnifiedProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let doResize: Bool
    public let doRescale: Bool
    public let rescaleFactor: CGFloat
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let audioSeqLength: Int
    public let audioMsPerToken: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let maxSoftTokens: Int
    public let size: Gemma3ProcessorConfiguration.ImageSize?
    public let imageTokenId: Int
    public let audioTokenId: Int
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?

    private struct ImageProcessorConfiguration: Decodable, Sendable {
        let doResize: Bool?
        let doRescale: Bool?
        let rescaleFactor: CGFloat?
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let patchSize: Int?
        let poolingKernelSize: Int?
        let modelPatchSize: Int?
        let maxSoftTokens: Int?
        let numSoftTokens: Int?
        let size: Gemma3ProcessorConfiguration.ImageSize?

        enum CodingKeys: String, CodingKey {
            case doResize = "do_resize"
            case doRescale = "do_rescale"
            case rescaleFactor = "rescale_factor"
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case modelPatchSize = "model_patch_size"
            case maxSoftTokens = "max_soft_tokens"
            case numSoftTokens = "num_soft_tokens"
            case size
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case doResize = "do_resize"
        case doRescale = "do_rescale"
        case rescaleFactor = "rescale_factor"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case audioMsPerToken = "audio_ms_per_token"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case numSoftTokens = "num_soft_tokens"
        case size
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case audioTokenId = "audio_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.imageProcessor)

        processorClass =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.processorClass)
            ?? "Gemma4UnifiedProcessor"
        doResize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doResize)
            ?? imageProcessor?.doResize
            ?? true
        doRescale =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doRescale)
            ?? imageProcessor?.doRescale
            ?? true
        rescaleFactor =
            try c.decodeIfPresent(CGFloat.self, forKey: CodingKeys.rescaleFactor)
            ?? imageProcessor?.rescaleFactor
            ?? (1.0 / 255.0)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize)
            ?? imageProcessor?.doNormalize
            ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean)
            ?? imageProcessor?.imageMean
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd)
            ?? imageProcessor?.imageStd
            ?? [0.5, 0.5, 0.5]
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize)
            ?? imageProcessor?.patchSize
            ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize)
            ?? imageProcessor?.poolingKernelSize
            ?? 3
        modelPatchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.modelPatchSize)
            ?? imageProcessor?.modelPatchSize
            ?? patchSize * poolingKernelSize
        maxSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxSoftTokens)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.numSoftTokens)
            ?? imageProcessor?.maxSoftTokens
            ?? imageProcessor?.numSoftTokens
            ?? 280
        imageSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength) ?? maxSoftTokens
        audioSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSeqLength) ?? 750
        audioMsPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioMsPerToken) ?? 40
        size =
            try c.decodeIfPresent(
                Gemma3ProcessorConfiguration.ImageSize.self, forKey: CodingKeys.size)
            ?? imageProcessor?.size
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boaTokenId) ?? 256_000
        eoaTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenId) ?? 258_883
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var fixedSize: CGSize {
        let patchesPerSide = max(1, Int(floor(sqrt(Double(maxSoftTokens)))))
        let side = patchesPerSide * patchSize * poolingKernelSize
        return CGSize(width: side, height: side)
    }

    public func aspectRatioPreservingSize(for imageSize: CGSize) throws -> CGSize {
        let width = max(1, Int(ceil(imageSize.width)))
        let height = max(1, Int(ceil(imageSize.height)))
        let sideMultiple = max(1, patchSize * poolingKernelSize)
        let maxTokens = max(1, maxSoftTokens)

        let targetPixels = Double(maxTokens * sideMultiple * sideMultiple)
        let resizeFactor = sqrt(targetPixels / Double(width * height))

        var targetWidth =
            Int(floor(Double(width) * resizeFactor / Double(sideMultiple))) * sideMultiple
        var targetHeight =
            Int(floor(Double(height) * resizeFactor / Double(sideMultiple))) * sideMultiple

        if targetWidth == 0 && targetHeight == 0 {
            throw VLMError.processing("Image is too small to resize for Gemma4 unified vision.")
        } else if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(width) / Double(height))) * sideMultiple))
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(height) / Double(width))) * sideMultiple))
        }

        return CGSize(width: targetWidth, height: targetHeight)
    }
}

public struct Gemma4UnifiedProcessor: UserInputProcessor {
    private let config: Gemma4UnifiedProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4UnifiedProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func patchify(_ pixelValues: MLXArray) -> (MLXArray, MLXArray, Int, THW) {
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)
        let patchHeight = height / config.modelPatchSize
        let patchWidth = width / config.modelPatchSize
        let realCount = min(patchHeight * patchWidth, config.maxSoftTokens)
        let patchDim = config.modelPatchSize * config.modelPatchSize * channels

        var patches = pixelValues.reshaped(
            1, channels, patchHeight, config.modelPatchSize, patchWidth, config.modelPatchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(patchHeight * patchWidth, patchDim)
        if realCount < patches.dim(0) {
            patches = patches[..<realCount, 0...]
        }
        if realCount < config.maxSoftTokens {
            patches = padded(patches, widths: [.init((0, config.maxSoftTokens - realCount)), 0])
        }

        var positionValues: [Int32] = []
        positionValues.reserveCapacity(config.maxSoftTokens * 2)
        var emitted = 0
        for y in 0 ..< patchHeight {
            for x in 0 ..< patchWidth where emitted < realCount {
                positionValues.append(Int32(x))
                positionValues.append(Int32(y))
                emitted += 1
            }
        }
        while emitted < config.maxSoftTokens {
            positionValues.append(-1)
            positionValues.append(-1)
            emitted += 1
        }
        let positions = MLXArray(positionValues, [config.maxSoftTokens, 2])
        return (patches, positions, realCount, THW(1, height, width))
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        pixels: MLXArray, positionIds: MLXArray, tokenCounts: [Int], frames: [THW]
    ) {
        var patchRows: [MLXArray] = []
        var positionRows: [MLXArray] = []
        var tokenCounts: [Int] = []
        var frames: [THW] = []

        for image in images {
            let processedImage = MediaProcessing.apply(image, processing: processing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage =
                if config.doResize {
                    try MediaProcessing.resampleBicubic(
                        srgbImage,
                        to: config.aspectRatioPreservingSize(for: srgbImage.extent.size))
                } else {
                    srgbImage
                }

            var pixelValues = MediaProcessing.asMLXArray(resizedImage)
            let rescaleMultiplier = Float(config.doRescale ? config.rescaleFactor * 255 : 255)
            if rescaleMultiplier != 1 {
                pixelValues = pixelValues * MLXArray(rescaleMultiplier, dtype: pixelValues.dtype)
            }
            if config.doNormalize {
                let mean = MLXArray(
                    config.imageMean.map { Float($0) }, [1, config.imageMean.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                let std = MLXArray(
                    config.imageStd.map { Float($0) }, [1, config.imageStd.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                pixelValues = (pixelValues - mean) / std
            }
            let (patches, positions, tokenCount, frame) = patchify(pixelValues)
            patchRows.append(patches)
            positionRows.append(positions)
            tokenCounts.append(tokenCount)
            frames.append(frame)
        }

        return (
            pixels: stacked(patchRows, axis: 0),
            positionIds: stacked(positionRows, axis: 0),
            tokenCounts: tokenCounts,
            frames: frames
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageData = try preprocess(
                images: input.images.map { try $0.asCIImage() },
                processing: input.processing
            )
            processedImage = LMInput.ProcessedImage(
                pixels: imageData.pixels,
                positionIds: imageData.positionIds,
                frames: imageData.frames
            )

            var imageIndex = 0
            var expandedTokens: [Int] = []
            expandedTokens.reserveCapacity(
                promptTokens.count + imageData.tokenCounts.reduce(0, +))
            for token in promptTokens {
                if token == config.imageTokenId {
                    let count =
                        imageIndex < imageData.tokenCounts.count
                        ? imageData.tokenCounts[imageIndex]
                        : config.imageSeqLength
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(repeating: config.imageTokenId, count: count))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: processedImage)
    }
}

// MARK: - Chat conventions

extension Gemma4 {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }
}

extension Gemma4Unified {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }
}
