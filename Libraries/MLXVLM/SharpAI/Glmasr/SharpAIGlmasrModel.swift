// SharpAI fork overlay: glmasr (GLM-ASR) speech model. Not part of upstream mlx-swift-lm.
// Audio features come from `SharpAIGlmasrFeatureExtractor` (not wired into a processor here).
// The encoder runs on the full padded input, as in the reference. Mask only sets token counts.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Tokens emitted for `convLength` encoder frames: groups of 4 after the conv stem.
func sharpAIGlmasrAudioTokenCount(frames: Int) -> Int {
    guard frames > 0 else { return 0 }
    let convLength = (frames - 1) / 2 + 1
    guard convLength >= 4 else { return 0 }
    return (convLength - 4) / 4 + 1
}

public class SharpAIGlmasr: Module, VLMModel, KVCacheDimensionProvider {
    public let config: SharpAIGlmasrConfiguration

    @ModuleInfo(key: "audio_tower") private var audioTower: SharpAIGlmasrAudioEncoder
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector: SharpAIGlmasrProjector
    @ModuleInfo(key: "language_model") private var languageModel: SharpAIGlmasrLanguageModel

    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: SharpAIGlmasrConfiguration) {
        self.config = config
        self._audioTower.wrappedValue = SharpAIGlmasrAudioEncoder(config.audioConfiguration)
        self._multiModalProjector.wrappedValue = SharpAIGlmasrProjector(
            audioIntermediate: config.audioConfiguration.intermediateSize,
            textHidden: config.textConfiguration.hiddenSize)
        self._languageModel.wrappedValue = SharpAIGlmasrLanguageModel(config.textConfiguration)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        guard tokens.dim(0) == 1 else {
            throw SharpAIGlmasrError.invalidInput("batch size must be 1, got \(tokens.dim(0))")
        }

        let embeddings = try inputEmbeddings(tokens: tokens, audio: input.audio)
        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel.logits(embeddings: embeddings[0..., range, 0...], cache: cache)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let result = languageModel.logits(
            embeddings: embeddings[0..., processed..., 0...], cache: cache)
        prefill.progress?(totalPositions, totalPositions)
        return .logits(LMOutput(logits: result))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel.logits(embeddings: languageModel.embed(inputs), cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIGlmasrSanitize.sanitize(weights: weights, config: config)
    }

    // MARK: - Audio

    private func inputEmbeddings(tokens: MLXArray, audio: LMInput.ProcessedAudio?) throws
        -> MLXArray
    {
        let textEmbeddings = languageModel.embed(tokens)
        guard let audio else { return textEmbeddings }

        let audioEmbeddings = try encodeAudio(features: audio.features, mask: audio.mask)
        return try mergeAudio(
            textEmbeddings: textEmbeddings,
            tokenIds: tokens[0].asArray(Int.self),
            audioEmbeddings: audioEmbeddings)
    }

    /// Encodes `[batch, numMelBins, frames]` features and keeps only the valid tokens.
    /// Returns `[totalTokens, textHidden]`, with rows concatenated in order.
    private func encodeAudio(features: MLXArray, mask: MLXArray?) throws -> MLXArray {
        let audio = config.audioConfiguration
        guard features.ndim == 3, features.dim(1) == audio.numMelBins else {
            throw SharpAIGlmasrError.invalidInput(
                "audio features must be [batch, \(audio.numMelBins), frames]")
        }
        let batch = features.dim(0)
        let frames = features.dim(2)
        let convLength = frames > 0 ? (frames - 1) / 2 + 1 : 0
        guard convLength >= 4, convLength % 4 == 0 else {
            throw SharpAIGlmasrError.invalidInput(
                "encoder length \(convLength) must be a positive multiple of 4")
        }

        let validFrames: [Int]
        if let mask {
            guard mask.ndim == 2, mask.dim(0) == batch, mask.dim(1) == frames else {
                throw SharpAIGlmasrError.invalidInput("feature mask must be [batch, frames]")
            }
            validFrames = mask.asType(.int32).sum(axis: -1).asArray(Int.self)
        } else {
            validFrames = Array(repeating: frames, count: batch)
        }

        // Stack 4 consecutive encoder frames per token, as the reference reshape does.
        let encoded = audioTower(features)
        let grouped = encoded.reshaped(batch, convLength / 4, audio.intermediateSize)
        let projected = multiModalProjector(grouped)

        var pieces: [MLXArray] = []
        for (row, count) in validFrames.enumerated() {
            let tokens = sharpAIGlmasrAudioTokenCount(frames: count)
            if tokens > 0 {
                pieces.append(projected[row, 0 ..< tokens])
            }
        }
        guard !pieces.isEmpty else {
            throw SharpAIGlmasrError.invalidInput("audio has no valid frames")
        }
        return concatenated(pieces, axis: 0)
    }

    /// Replaces each run of `audioTokenId` placeholders with the next audio embeddings, in order.
    private func mergeAudio(
        textEmbeddings: MLXArray, tokenIds: [Int], audioEmbeddings: MLXArray
    ) throws -> MLXArray {
        let audioTokenId = config.audioTokenId
        let placeholders = tokenIds.filter { $0 == audioTokenId }.count
        guard placeholders == audioEmbeddings.dim(0) else {
            throw SharpAIGlmasrError.audioTokenMismatch(
                placeholders: placeholders, features: audioEmbeddings.dim(0))
        }

        let audio = audioEmbeddings.asType(textEmbeddings.dtype)
        var parts: [MLXArray] = []
        var textStart = 0
        var audioOffset = 0
        var index = 0
        while index < tokenIds.count {
            guard tokenIds[index] == audioTokenId else {
                index += 1
                continue
            }
            if textStart < index {
                parts.append(textEmbeddings[0..., textStart ..< index, 0...])
            }
            var runEnd = index
            while runEnd < tokenIds.count, tokenIds[runEnd] == audioTokenId {
                runEnd += 1
            }
            let runLength = runEnd - index
            parts.append(
                audio[audioOffset ..< (audioOffset + runLength)].expandedDimensions(axis: 0))
            audioOffset += runLength
            textStart = runEnd
            index = runEnd
        }
        if textStart < tokenIds.count {
            parts.append(textEmbeddings[0..., textStart ..< tokenIds.count, 0...])
        }
        return concatenated(parts, axis: 1)
    }
}
