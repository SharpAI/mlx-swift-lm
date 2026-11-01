import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// ColQwen3 retrieval encoder: Qwen3-VL backbone, linear projection, L2 normalization.
///
/// Not a generator, so it does not conform to `LanguageModel`. `embed` returns one
/// unit-length vector per token. Weight keys: `vision_tower.*`, `language_model.*`,
/// `embedding_proj_layer.*`.
///
/// UNVERIFIED: the projection bias, causal attention and eps-free normalization follow
/// transformers' ColQwen2. The tomoro ColQwen3 remote code is not on disk to confirm them.
public final class SharpAIColqwen3Model: Module {

    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen3VLLanguage.LanguageModel
    @ModuleInfo(key: "embedding_proj_layer") var embeddingProjLayer: Linear

    public let config: SharpAIColqwen3Configuration

    public var embedDim: Int { config.embedDim }

    public init(_ config: SharpAIColqwen3Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.backbone.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config.backbone)
        _embeddingProjLayer.wrappedValue = Linear(
            config.backbone.textConfiguration.hiddenSize, config.embedDim)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIColqwen3Sanitize.sanitize(weights: weights, config: config)
    }

    /// Per-token multi-vectors `[batch, seq, embedDim]`, unit-norm, zero where `attentionMask` is 0.
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
                throw SharpAIColqwen3Error.missingImageGrid
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

        let positionIds = ropePositionIds(
            inputIds: inputIds, grids: visualGrids, attentionMask: attentionMask)

        let mask = try attentionMask.map { try causalPaddingMask($0, length: inputIds.dim(1)) }

        let lastHidden = runBackbone(
            hidden: hidden, mask: mask, positionIds: positionIds,
            visualIndices: visualIndices, deepstackEmbeds: deepstackEmbeds)

        let projected = embeddingProjLayer(lastHidden.asType(embeddingProjLayer.weight.dtype))
        let norms = sqrt((projected * projected).sum(axis: -1, keepDims: true))
        var normalized = projected / norms

        if let attentionMask {
            let valid = attentionMask.expandedDimensions(axis: -1).asType(normalized.dtype)
            normalized = normalized * valid
        }
        return normalized
    }

    /// M-RoPE positions `[3, batch, seq]`. `getRopeIndex` restarts its image cursor at 0 for
    /// every row and leaves rows without vision tokens at all-ones, so a batch with images is
    /// computed one row at a time, each row receiving only its own grids (nil for a text row).
    private func ropePositionIds(
        inputIds: MLXArray, grids: [THW]?, attentionMask: MLXArray?
    ) -> MLXArray {
        let backbone = config.backbone
        let mergeSize = backbone.visionConfiguration.spatialMergeSize
        let batch = inputIds.dim(0)

        guard batch > 1, let grids else {
            return Qwen3VLLanguage.getRopeIndex(
                inputIds: inputIds,
                imageGridTHW: grids,
                videoGridTHW: nil,
                spatialMergeSize: mergeSize,
                imageTokenId: backbone.imageTokenIndex,
                videoTokenId: backbone.videoTokenIndex,
                visionStartTokenId: backbone.visionStartTokenId,
                attentionMask: attentionMask
            ).0
        }

        let length = inputIds.dim(1)
        let flatIds = inputIds.asType(.int32).asArray(Int32.self)
        let imageToken = Int32(backbone.imageTokenIndex)
        var cursor = 0
        var rowPositions: [MLXArray] = []
        for row in 0 ..< batch {
            let imageTokens = flatIds[row * length ..< (row + 1) * length]
                .filter { $0 == imageToken }.count
            var rowGrids: [THW] = []
            var covered = 0
            while covered < imageTokens, cursor < grids.count {
                rowGrids.append(grids[cursor])
                covered += grids[cursor].product / (mergeSize * mergeSize)
                cursor += 1
            }
            let (positions, _) = Qwen3VLLanguage.getRopeIndex(
                inputIds: inputIds[row ..< row + 1, 0...],
                imageGridTHW: rowGrids.isEmpty ? nil : rowGrids,
                videoGridTHW: nil,
                spatialMergeSize: mergeSize,
                imageTokenId: backbone.imageTokenIndex,
                videoTokenId: backbone.videoTokenIndex,
                visionStartTokenId: backbone.visionStartTokenId,
                attentionMask: attentionMask.map { $0[row ..< row + 1, 0...] })
            rowPositions.append(positions)
        }
        return concatenated(rowPositions, axis: 1)
    }

    /// Runs the language backbone. Deepstack injection indexes the sequence axis, while
    /// `visualIndices` are flat positions over `[batch * seq]`, so a batch with images is run
    /// one row at a time with row-local indices and that row's slice of the deepstack features.
    private func runBackbone(
        hidden: MLXArray, mask: MLXArray?, positionIds: MLXArray,
        visualIndices: [Int]?, deepstackEmbeds: [MLXArray]?
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        guard batch > 1, let visualIndices, !visualIndices.isEmpty, let deepstackEmbeds else {
            return languageModel.model(
                nil,
                cache: nil,
                inputEmbeddings: hidden,
                mask: mask,
                positionIds: positionIds,
                visualIndices: visualIndices,
                deepstackEmbeds: deepstackEmbeds)
        }

        var rows: [MLXArray] = []
        var offset = 0
        for row in 0 ..< batch {
            let local = visualIndices.filter { $0 / length == row }.map { $0 - row * length }
            let count = local.count
            let rowDeepstack = deepstackEmbeds.map { $0[offset ..< offset + count, 0...] }
            offset += count
            rows.append(
                languageModel.model(
                    nil,
                    cache: nil,
                    inputEmbeddings: hidden[row ..< row + 1, 0..., 0...],
                    mask: mask.map { $0[row ..< row + 1, 0..., 0..., 0...] },
                    positionIds: positionIds[0..., row ..< row + 1, 0...],
                    visualIndices: local,
                    deepstackEmbeds: rowDeepstack))
        }
        return concatenated(rows, axis: 0)
    }

    private func scatterImageFeatures(
        features: MLXArray, embeds: MLXArray, inputIds: MLXArray
    ) throws -> (MLXArray, [Int]) {
        let imageMask = (inputIds .== MLXArray(config.backbone.imageTokenIndex))
        let visualIndices = nonZeroIndices(imageMask)

        let hiddenSize = embeds.dim(-1)
        guard visualIndices.count * hiddenSize == features.size else {
            throw SharpAIColqwen3Error.featureTokenMismatch(
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
            throw SharpAIColqwen3Error.invalidAttentionMask(shape: attentionMask.shape)
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
