// SharpAI fork overlay: qwen3_omni_moe thinker (text, vision and audio input; text output).
// Not part of upstream mlx-swift-lm.
// The text decoder is Qwen3VLMoELanguage.LanguageModel and the ViT is Qwen3VLVision.VisionModel,
// both reused as written. The audio tower is SharpAIQwen3OmniMoeAudioEncoder.
// Known gaps: talker and code2wav (speech output) are not built. Audio tokens take plain text
// positions in M-RoPE, so audio-in-video interleaving is not handled. UNVERIFIED against a
// checkpoint: the deepstack merger key name and the M-RoPE section layout.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class SharpAIQwen3OmniMoeThinker: Module, VLMModel, KVCacheDimensionProvider {

    public let config: SharpAIQwen3OmniMoeConfiguration
    let languageConfig: Qwen3VLMoEConfiguration

    @ModuleInfo(key: "audio_tower") private var audioTower: SharpAIQwen3OmniMoeAudioEncoder
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Qwen3VLMoELanguage.LanguageModel

    public var vocabularySize: Int { config.textConfiguration.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: SharpAIQwen3OmniMoeConfiguration) throws {
        let language = try config.languageConfiguration()
        self.config = config
        self.languageConfig = language
        _audioTower.wrappedValue = SharpAIQwen3OmniMoeAudioEncoder(config.audioConfiguration)
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLMoELanguage.LanguageModel(language)
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let dtype = visionModel.patchEmbed.proj.weight.dtype
        var textEmbeds = languageModel.model.embedTokens(inputIds)

        var pixelParts: [MLXArray] = []
        var imageGrids: [THW]? = nil
        var videoGrids: [THW]? = nil
        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageGrids = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoGrids = video.frames
        }
        let grids = (imageGrids ?? []) + (videoGrids ?? [])

        var pixelValues: MLXArray? = nil
        var visualMask: MLXArray? = nil
        var deepstackEmbeds: [MLXArray]? = nil
        if !pixelParts.isEmpty, !grids.isEmpty {
            let pixels = concatenated(pixelParts)
            pixelValues = pixels
            let (visionHidden, deepstackOutputs) = visionModel(pixels, gridTHW: grids)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = grids.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let flattened = concatenated(visionHidden.split(indices: splitIndices)).asType(
                textEmbeds.dtype)

            let (merged, mask) = try scatter(
                features: flattened, into: textEmbeds, inputIds: inputIds,
                tokenIds: [config.imageTokenId, config.videoTokenId])
            textEmbeds = merged
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let slices = layerFeatures.split(indices: splitIndices)
                    return concatenated(slices).asType(textEmbeds.dtype)
                }
            }
        }

        if let audio = input.audio {
            guard inputIds.ndim == 2, inputIds.dim(0) == 1 else {
                throw SharpAIQwen3OmniMoeError.invalidInput("audio input requires batch size 1")
            }
            let audioEmbeds = try audioTower(audio.features, featureMask: audio.mask)
            let (merged, _) = try scatter(
                features: audioEmbeds.asType(textEmbeds.dtype), into: textEmbeds,
                inputIds: inputIds, tokenIds: [config.audioTokenId])
            textEmbeds = merged
        }

        let output = languageModel(
            inputIds,
            cache: castCache(cache),
            state: state,
            inputEmbeddings: textEmbeds,
            mask: nil,
            positionIds: nil,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: pixelValues,
            imageGridTHW: imageGrids,
            videoGridTHW: videoGrids)

        let total = inputIds.dim(-1)
        prefill.progress?(total, total)
        return .logits(output)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        languageModel(
            input.tokens,
            cache: castCacheOptional(cache),
            state: state,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIQwen3OmniMoeSanitize.sanitize(weights: weights, config: config)
    }

    // MARK: - Helpers

    private func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    private func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    /// Writes `features` (`[count, hidden]`) into the rows of `embeds` whose token id is in
    /// `tokenIds`, in order. Returns the merged embeddings and the matching `[batch, seq]` mask.
    private func scatter(
        features: MLXArray, into embeds: MLXArray, inputIds: MLXArray, tokenIds: [Int]
    ) throws -> (MLXArray, MLXArray) {
        let ids = inputIds.asArray(Int.self)
        let positions = ids.indices.filter { tokenIds.contains(ids[$0]) }
        let hidden = embeds.dim(-1)

        guard positions.count * hidden == features.size else {
            throw SharpAIQwen3OmniMoeError.featureTokenMismatch(
                placeholders: positions.count, features: features.size / max(hidden, 1))
        }

        var flat = embeds.reshaped([-1, hidden])
        if !positions.isEmpty {
            flat[MLXArray(positions.map { Int32($0) }), 0...] =
                features.reshaped([positions.count, hidden])
        }
        let maskValues = ids.map { tokenIds.contains($0) ? Int32(1) : Int32(0) }
        let mask = MLXArray(maskValues, inputIds.shape).asType(.bool)
        return (flat.reshaped(embeds.shape), mask)
    }
}

// MARK: - Chat conventions

extension SharpAIQwen3OmniMoeThinker {
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
