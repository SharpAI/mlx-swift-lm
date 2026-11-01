// SharpAI fork overlay: bagel (BAGEL-7B-MoT) vision-language model. Not part of upstream
// mlx-swift-lm.
//
// Scope: the understanding path only. A SigLIP tower encodes one image, a connector maps it to
// the text width, and the features replace a contiguous run of image placeholder tokens. A
// Qwen2 decoder with QK norm runs over the merged embeddings.
//
// Not implemented (gaps): the generation path (VAE, flow matching, `_moe_gen` experts, latent
// and timestep embedders), packed multi-image NaViT batching, batch sizes above 1, and any
// bidirectional attention over image tokens. UNVERIFIED: the placeholder-token convention and
// the image-to-prompt layout. No processor is registered for this model.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIBagel: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "language_model") var languageModel: SharpAIBagelLanguageModel
    @ModuleInfo(key: "vit_model") var vitModel: SharpAIBagelViTModel
    @ModuleInfo var connector: SharpAIBagelConnector

    public let config: SharpAIBagelConfiguration

    /// Checkpoint prefixes this understanding-only model loads. Other keys are dropped.
    static let loadedPrefixes = ["language_model.", "vit_model.", "connector."]

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }

    public var kvHeads: [Int] {
        let text = config.textConfiguration
        return (0 ..< text.hiddenLayers).map { _ in text.kvHeads }
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: SharpAIBagelConfiguration) {
        self.config = config
        let text = config.textConfiguration
        let vision = config.visionConfiguration
        self._languageModel.wrappedValue = SharpAIBagelLanguageModel(text)
        self._vitModel.wrappedValue = SharpAIBagelViTModel(
            vision, gridSize: config.vitMaxNumPatchPerSide)
        self._connector.wrappedValue = SharpAIBagelConnector(
            visionHidden: vision.hiddenSize, textHidden: text.hiddenSize)
    }

    /// Encodes one image in NCHW layout (1, C, H, W) into text-space features (1, N, textHidden).
    public func imageFeatures(_ pixels: MLXArray) throws -> MLXArray {
        guard pixels.ndim == 4, pixels.dim(0) == 1 else {
            throw VLMError.processing("bagel: expected one image in NCHW layout")
        }
        let patch = config.visionConfiguration.patchSize
        let rows = pixels.dim(2) / patch
        let cols = pixels.dim(3) / patch
        guard rows > 0, cols > 0 else {
            throw VLMError.imageProcessingFailure("bagel: image is smaller than one patch")
        }
        guard rows <= config.vitMaxNumPatchPerSide, cols <= config.vitMaxNumPatchPerSide else {
            throw VLMError.imageProcessingFailure(
                "bagel: image grid exceeds vit_max_num_patch_per_side")
        }

        let nhwc = pixels.transposed(0, 2, 3, 1)
        return connector(vitModel(nhwc))
    }

    /// Writes `features` into the run of image placeholder tokens in `embeddings`.
    private func mergeImageFeatures(
        _ features: MLXArray, into embeddings: MLXArray, inputIds: MLXArray
    ) throws -> MLXArray {
        guard inputIds.dim(0) == 1 else {
            throw VLMError.processing("bagel: batch size must be 1")
        }

        let ids: [Int32] = inputIds[0].asArray(Int32.self)
        let imageToken = Int32(config.imageTokenIndex)
        guard let start = ids.firstIndex(of: imageToken) else {
            throw VLMError.processing("bagel: image given but the prompt has no image token")
        }

        let count = features.dim(1)
        let run = ids[start ..< min(ids.count, start + count)]
        guard run.count == count, run.allSatisfy({ $0 == imageToken }) else {
            throw VLMError.processing(
                "bagel: image tokens must be contiguous and match the \(count) image features")
        }

        let placeholderCount = ids.reduce(0) { $1 == imageToken ? $0 + 1 : $0 }
        guard placeholderCount == count else {
            throw VLMError.processing(
                "bagel: prompt has \(placeholderCount) image tokens but the image gives \(count) features")
        }

        let seqLen = embeddings.dim(1)
        let placed = padded(features, widths: [0, .init((start, seqLen - start - count)), 0])
        let imageMask = expandedDimensions(inputIds .== imageToken, axis: -1)
        return which(imageMask, placed, embeddings)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        var embeddings = languageModel.model.embedTokens(inputIds)

        if let image = input.image {
            let features = try imageFeatures(image.pixels.asType(embeddings.dtype))
            embeddings = try mergeImageFeatures(
                features.asType(embeddings.dtype), into: embeddings, inputIds: inputIds)
        }

        let logits = languageModel(embeddings, cache: cache)

        let total = embeddings.dim(1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(languageModel.model.embedTokens(inputs), cache: cache)
    }

    /// Maps HF checkpoint keys to this module tree. Keys are kept as written in the checkpoint.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            guard Self.loadedPrefixes.contains(where: { key.hasPrefix($0) }) else { continue }

            // Generation path, not loaded here.
            if key.contains("_moe_gen") { continue }

            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") { continue }

            if config.textConfiguration.tieWordEmbeddings, key.hasPrefix("language_model.lm_head.")
            {
                continue
            }

            if key.contains("patch_embedding.weight"), value.ndim == 4,
                !Self.isMLXConvWeight(value)
            {
                // PyTorch layout (out, in, kH, kW) to MLX layout (out, kH, kW, in).
                sanitized[key] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[key] = value
            }
        }

        return sanitized
    }

    private static func isMLXConvWeight(_ array: MLXArray) -> Bool {
        let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
        return outChannels >= kH && outChannels >= kW && kH == kW
    }
}
