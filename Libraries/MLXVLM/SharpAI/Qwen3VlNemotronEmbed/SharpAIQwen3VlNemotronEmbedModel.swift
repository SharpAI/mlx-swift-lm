import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3-VL backbone used as a ColBERT-style multi-vector embedder.
///
/// Verified against: https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2/resolve/main/modeling_qwen3_vl_nemotron_embed.py
/// (`Qwen3VLNemotronEmbedTextModel` sets `is_causal = False` on every layer, `Qwen3VLNemotronEmbedForConditionalGeneration.forward`
/// returns the inner model's `last_hidden_state` with no projection, `_extract_embeddings` multiplies by the attention mask and
/// applies `F.normalize(dim=-1)`). The text stack skips its final RMSNorm. The checkpoint has no embedding head.
///
/// Not a generator, so it does not conform to `LanguageModel`. `embed` returns one unit-length vector per token,
/// `[batch, seq, hidden]`, with padded positions set to zero. Weight keys: `vision_tower.*`, `language_model.*`.
public final class SharpAIQwen3VlNemotronEmbedModel: Module {

    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen3VLLanguage.LanguageModel

    public let config: SharpAIQwen3VlNemotronEmbedConfiguration

    /// Width of each output vector. The reference emits the text hidden size, not `config.embedDim`.
    public var embedDim: Int { config.backbone.textConfiguration.hiddenSize }

    public init(_ config: SharpAIQwen3VlNemotronEmbedConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.backbone.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config.backbone)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIQwen3VlNemotronEmbedSanitize.sanitize(weights: weights, config: config)
    }

    /// One unit-length vector per real token, `[batch, seq, hidden]`. Padded positions are zero.
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
            // DeepStack injection and the image scatter use flat positions, which only hold for one row.
            guard inputIds.dim(0) == 1 else {
                throw SharpAIQwen3VlNemotronEmbedError.unsupportedBatchWithImages(batch: inputIds.dim(0))
            }
            guard let grids = imageGridTHW, !grids.isEmpty else {
                throw SharpAIQwen3VlNemotronEmbedError.missingImageGrid
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

        let mask = try attentionMask.map { try bidirectionalPaddingMask($0, length: inputIds.dim(1)) }
        let attentionMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            attentionMode = .array(mask)
        } else {
            attentionMode = .none
        }

        let lastHidden = decodeWithoutFinalNorm(
            hidden,
            mask: attentionMode,
            positionIds: positionIds,
            visualIndices: visualIndices,
            deepstackEmbeds: deepstackEmbeds)

        // Reference order: zero the padded positions, then L2-normalize each token vector.
        let tokens: MLXArray
        if let attentionMask {
            tokens = lastHidden * attentionMask.asType(lastHidden.dtype).expandedDimensions(axis: -1)
        } else {
            tokens = lastHidden
        }
        // Normalize in float32: squared sums overflow float16, and the clamp below underflows to 0 there.
        let tokens32 = tokens.asType(.float32)
        let norms = sqrt((tokens32 * tokens32).sum(axis: -1, keepDims: true))
        // Same clamp as torch.nn.functional.normalize (eps = 1e-12), so zero rows stay zero.
        let normalized = tokens32 / maximum(norms, MLXArray(Float(1e-12)))
        return normalized.asType(lastHidden.dtype)
    }

    /// The Qwen3-VL text stack in `Qwen3VLLanguage.Model` applies `norm` after its last layer. The
    /// reference embedder returns the output before that norm, so this repeats the same decoder
    /// loop, including DeepStack injection, and stops before `norm`.
    private func decodeWithoutFinalNorm(
        _ embeds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        positionIds: MLXArray,
        visualIndices: [Int]?,
        deepstackEmbeds: [MLXArray]?
    ) -> MLXArray {
        var hidden = embeds
        let indices = visualIndices ?? []
        let deepstackIndices: MLXArray? =
            indices.isEmpty ? nil : MLXArray(indices.map { UInt32($0) })

        for (index, layer) in languageModel.model.layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: nil, positionIds: positionIds)

            if let deepstackEmbeds, index < deepstackEmbeds.count, let deepstackIndices {
                let injected = hidden[0..., deepstackIndices, 0...] + deepstackEmbeds[index]
                hidden[0..., deepstackIndices, 0...] = injected
            }
        }
        return hidden
    }

    private func scatterImageFeatures(
        features: MLXArray, embeds: MLXArray, inputIds: MLXArray
    ) throws -> (MLXArray, [Int]) {
        let imageMask = (inputIds .== MLXArray(config.backbone.imageTokenIndex))
        let visualIndices = nonZeroIndices(imageMask)

        let hiddenSize = embeds.dim(-1)
        guard visualIndices.count * hiddenSize == features.size else {
            throw SharpAIQwen3VlNemotronEmbedError.featureTokenMismatch(
                expected: visualIndices.count, actual: features.dim(0))
        }
        guard !visualIndices.isEmpty else { return (embeds, visualIndices) }

        let result = embeds.reshaped([-1, hiddenSize])
        result[MLXArray(visualIndices.map { Int32($0) }), 0...] =
            features.reshaped([visualIndices.count, hiddenSize])

        return (result.reshaped(embeds.shape), visualIndices)
    }

    /// Bidirectional mask: each real token attends to every other real token, and padded keys are hidden.
    /// The diagonal stays open so a padded query row is never fully masked. Those rows are zeroed in `embed`.
    /// Verified against: the reference `_create_bidirectional_mask` (non-causal key-padding mask).
    private func bidirectionalPaddingMask(_ attentionMask: MLXArray, length: Int) throws -> MLXArray {
        guard attentionMask.ndim == 2, attentionMask.dim(1) == length else {
            throw SharpAIQwen3VlNemotronEmbedError.invalidAttentionMask(shape: attentionMask.shape)
        }
        let batch = attentionMask.dim(0)
        let positions = MLXArray(0 ..< length).asType(.int32)
        let keys = positions[.newAxis, 0...]
        let queries = positions[0..., .newAxis]

        // Element [i, j] is about query i and key j; both broadcast to [batch, 1, length, length].
        let diagonal = keys .== queries
        let keep = attentionMask.asType(.bool).reshaped([batch, 1, 1, length])
        return keep .|| diagonal
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
