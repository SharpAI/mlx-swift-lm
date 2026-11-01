// SharpAI fork overlay: audioflamingo3 (audio + text) model. Not part of upstream mlx-swift-lm.
// Input audio must already be log-mel features. The feature extractor and <sound> prompt
// expansion are not written here, so no processor is registered.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIAudioflamingo3: Module, VLMModel, KVCacheDimensionProvider {
    public let config: SharpAIAudioflamingo3Configuration

    @ModuleInfo(key: "audio_tower") private var audioTower: SharpAIAudioflamingo3AudioEncoder
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        SharpAIAudioflamingo3Projector
    @ModuleInfo(key: "language_model") private var languageModel:
        SharpAIAudioflamingo3LanguageModel

    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: SharpAIAudioflamingo3Configuration) {
        self.config = config
        self._audioTower.wrappedValue = SharpAIAudioflamingo3AudioEncoder(
            config.audioConfiguration)
        self._multiModalProjector.wrappedValue = SharpAIAudioflamingo3Projector(
            audioHidden: config.audioConfiguration.hiddenSize,
            textHidden: config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
        self._languageModel.wrappedValue = SharpAIAudioflamingo3LanguageModel(
            config.textConfiguration)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        guard tokens.dim(0) == 1 else {
            throw SharpAIAudioflamingo3Error.invalidInput(
                "batch size must be 1, got \(tokens.dim(0))")
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
        SharpAIAudioflamingo3Sanitize.sanitize(weights: weights, config: config)
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

    /// Encodes `[chunks, numMelBins, frames]` features and keeps only the valid pooled tokens.
    /// Returns `[totalTokens, textHidden]`, with chunks concatenated in order.
    private func encodeAudio(features: MLXArray, mask: MLXArray?) throws -> MLXArray {
        let encoder = config.audioConfiguration
        guard features.ndim == 3, features.dim(1) == encoder.numMelBins else {
            throw SharpAIAudioflamingo3Error.invalidInput(
                "audio features must be [chunks, \(encoder.numMelBins), frames]")
        }
        let chunks = features.dim(0)
        let frames = features.dim(2)
        guard frames >= 3, (frames - 1) / 2 + 1 <= encoder.maxSourcePositions else {
            throw SharpAIAudioflamingo3Error.invalidInput(
                "audio frame count \(frames) is outside the encoder range")
        }

        let validFrames: [Int]
        if let mask {
            guard mask.ndim == 2, mask.dim(0) == chunks, mask.dim(1) == frames else {
                throw SharpAIAudioflamingo3Error.invalidInput(
                    "feature mask must be [chunks, frames]")
            }
            validFrames = mask.asType(.int32).sum(axis: -1).asArray(Int.self)
        } else {
            validFrames = Array(repeating: frames, count: chunks)
        }

        let projected = multiModalProjector(audioTower(features, featureMask: mask))
        var pieces: [MLXArray] = []
        for (chunk, count) in validFrames.enumerated() {
            let tokens = sharpAIAudioflamingo3PooledLength(count)
            if tokens > 0 {
                pieces.append(projected[chunk, 0 ..< tokens])
            }
        }
        guard !pieces.isEmpty else {
            throw SharpAIAudioflamingo3Error.invalidInput("audio has no valid frames")
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
            throw SharpAIAudioflamingo3Error.audioTokenMismatch(
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
