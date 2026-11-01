import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// ops_colqwen3 retrieval encoder: Qwen3-VL backbone, linear projection to `dims`, L2 normalization.
///
/// Not a generator, so it does not conform to `LanguageModel`. `embed` returns one
/// unit-length vector per token. Weight keys: `vision_tower.*`, `language_model.*`,
/// `embedding_proj_layer.*`.
///
/// UNVERIFIED: the head key `embedding_proj_layer` and its bias follow colpali-engine's
/// ColQwen layout. The OpenSearch checkpoint is not on disk, so the key is not confirmed.
public final class SharpAIOpsColqwen3Model: Module {

    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen3VLLanguage.LanguageModel
    @ModuleInfo(key: "embedding_proj_layer") var embeddingProjLayer: Linear

    public let config: SharpAIOpsColqwen3Configuration

    public var dims: Int { config.dims }

    public init(_ config: SharpAIOpsColqwen3Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.backbone.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config.backbone)
        _embeddingProjLayer.wrappedValue = Linear(
            config.backbone.textConfiguration.hiddenSize, config.dims)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIOpsColqwen3Sanitize.sanitize(weights: weights, config: config)
    }

    /// Per-token multi-vectors `[batch, seq, dims]`, unit-norm, zero where `attentionMask` is 0.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seq]` token ids.
    ///   - pixelValues: flattened image patches for `Qwen3VLVision`, if any.
    ///   - imageGridTHW: one grid per image, matching `pixelValues`.
    ///   - attentionMask: `[batch, seq]` with 1 for real tokens and 0 for padding.
    public func embed(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        imageGridTHW: [THW]? = nil,
        attentionMask: MLXArray? = nil
    ) throws -> MLXArray {
        let textEmbeds = languageModel.model.embedTokens(inputIds)
        var hidden = textEmbeds
        var visualIndices: [Int]? = nil
        var deepstackEmbeds: [MLXArray]? = nil
        var visualGrids: [THW]? = nil

        if let pixelValues {
            guard let grids = imageGridTHW, !grids.isEmpty else {
                throw SharpAIOpsColqwen3Error.missingImageGrid
            }
            visualGrids = grids
            let dtype = visionModel.patchEmbed.proj.weight.dtype
            let (visionHidden, deepstackOutputs) = visionModel(
                pixelValues.asType(dtype), gridTHW: grids)

            let mergeSize = config.backbone.visionConfiguration.spatialMergeSize
            let splits = grids.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let featureSlices = visionHidden.split(indices: splitIndices)
            let flattenedFeatures = concatenated(featureSlices).asType(textEmbeds.dtype)

            let (merged, indices) = try scatterImageFeatures(
                features: flattenedFeatures, embeds: textEmbeds, inputIds: inputIds)
            hidden = merged
            visualIndices = indices

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let slices = layerFeatures.split(indices: splitIndices)
                    return concatenated(slices).asType(textEmbeds.dtype)
                }
            }
        }

        let (positionIds, _) = Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIds,
            imageGridTHW: visualGrids,
            videoGridTHW: nil,
            spatialMergeSize: config.backbone.visionConfiguration.spatialMergeSize,
            imageTokenId: config.backbone.imageTokenIndex,
            videoTokenId: config.backbone.videoTokenIndex,
            visionStartTokenId: config.backbone.visionStartTokenId,
            attentionMask: attentionMask)

        let mask = try attentionMask.map { try causalPaddingMask($0, length: inputIds.dim(1)) }

        let lastHidden = languageModel.model(
            nil,
            cache: nil,
            inputEmbeddings: hidden,
            mask: mask,
            positionIds: positionIds,
            visualIndices: visualIndices,
            deepstackEmbeds: deepstackEmbeds)

        let projected = embeddingProjLayer(lastHidden.asType(embeddingProjLayer.weight.dtype))
        let norms = sqrt((projected * projected).sum(axis: -1, keepDims: true))
        var normalized = projected / norms

        if let attentionMask {
            let valid = attentionMask.expandedDimensions(axis: -1).asType(normalized.dtype)
            normalized = normalized * valid
        }
        return normalized
    }

    private func scatterImageFeatures(
        features: MLXArray, embeds: MLXArray, inputIds: MLXArray
    ) throws -> (MLXArray, [Int]) {
        let imageMask = (inputIds .== MLXArray(config.backbone.imageTokenIndex))
        let visualIndices = nonZeroIndices(imageMask)

        let hiddenSize = embeds.dim(-1)
        guard visualIndices.count * hiddenSize == features.size else {
            throw SharpAIOpsColqwen3Error.featureTokenMismatch(
                expected: visualIndices.count, actual: features.dim(0))
        }
        guard !visualIndices.isEmpty else { return (embeds, visualIndices) }

        let result = embeds.reshaped([-1, hiddenSize])
        result[MLXArray(visualIndices.map { Int32($0) }), 0...] =
            features.reshaped([visualIndices.count, hiddenSize])

        return (result.reshaped(embeds.shape), visualIndices)
    }

    /// Causal mask that also hides padded keys. Each position may attend to itself,
    /// so a left-padded query row never has every key masked.
    private func causalPaddingMask(_ attentionMask: MLXArray, length: Int) throws -> MLXArray {
        guard attentionMask.ndim == 2, attentionMask.dim(1) == length else {
            throw SharpAIOpsColqwen3Error.invalidAttentionMask(shape: attentionMask.shape)
        }
        let batch = attentionMask.dim(0)
        let positions = MLXArray(0 ..< length).asType(.int32)
        let keys = positions[.newAxis, 0...]
        let queries = positions[0..., .newAxis]

        // Element [i, j] is about query i and key j; both broadcast to [batch, 1, length, length].
        let causal = keys .<= queries
        let diagonal = keys .== queries
        let keep = attentionMask.asType(.bool).reshaped([batch, 1, 1, length])
        return (causal .&& keep) .|| diagonal
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    private func nonZeroIndices(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }
}
