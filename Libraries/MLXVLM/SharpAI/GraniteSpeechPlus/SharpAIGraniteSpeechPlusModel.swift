// SharpAI fork overlay: granite_speech_plus model for the LanguageModel pipeline. Not part of
// upstream mlx-swift-lm.
//
// Structure follows GraniteSpeechForConditionalGeneration (transformers granite_speech):
// encoder -> projector -> merge into the placeholder rows -> Granite decoder -> logits.
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/modeling_granite_speech.py
//
// prepare(_:cache:state:prefill:) takes LMInput.audio as log-mel features (batch, time, input_dim).
// Feature extraction is not done here. Audio placeholder tokens are matched in order, one embedding
// row per placeholder. The LoRA adapter the checkpoint ships is NOT applied (see the remaining list).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIGraniteSpeechPlusForConditionalGeneration:
    Module, LanguageModel, KVCacheDimensionProvider
{

    public let configuration: SharpAIGraniteSpeechPlusConfiguration
    public let kvHeads: [Int]

    @ModuleInfo(key: "encoder") var encoder: SharpAIGraniteSpeechPlusEncoder
    @ModuleInfo(key: "projector") var projector: SharpAIGraniteSpeechPlusProjector
    @ModuleInfo(key: "language_model") var languageModel: SharpAIGraniteSpeechPlusLanguageModel

    public init(_ configuration: SharpAIGraniteSpeechPlusConfiguration) {
        let text = configuration.textConfiguration
        self.configuration = configuration
        self.kvHeads = Array(repeating: text.kvHeads, count: text.hiddenLayers)
        self._encoder.wrappedValue = SharpAIGraniteSpeechPlusEncoder(
            configuration.encoderConfiguration)
        self._projector.wrappedValue = SharpAIGraniteSpeechPlusProjector(
            configuration.projectorConfiguration,
            encoderWidth: configuration.encoderConfiguration.outputWidth,
            textWidth: text.hiddenSize,
            windowSize: configuration.windowSize,
            downsampleRate: configuration.downsampleRate)
        self._languageModel.wrappedValue = SharpAIGraniteSpeechPlusLanguageModel(text)
        super.init()
    }

    /// Audio features (batch, time, input_dim) -> projected audio embeddings
    /// (batch, blocks * queries per block, text hidden size).
    public func encodeAudio(_ features: MLXArray) -> MLXArray {
        projector(encoder(features))
    }

    /// Replaces the audio placeholder rows, in order, with the audio embedding rows. The result is
    /// raw token embeddings; the decoder applies the embedding multiplier afterwards.
    func mergeAudio(tokens: MLXArray, audioEmbeddings: MLXArray) throws -> MLXArray {
        let (b, t) = (tokens.dim(0), tokens.dim(1))
        let hidden = audioEmbeddings.dim(-1)
        let audioId = Int32(configuration.audioTokenIndex)

        let flatTokens = tokens.asType(.int32).reshaped(-1).asArray(Int32.self)
        let placeholders = flatTokens.indices.filter { flatTokens[$0] == audioId }
        let features = audioEmbeddings.reshaped(-1, hidden)
        guard placeholders.count == features.dim(0) else {
            throw SharpAIGraniteSpeechPlusError.audioTokenMismatch(
                tokens: placeholders.count, features: features.dim(0))
        }

        // Placeholders are looked up as token 0 and then overwritten, so the lookup stays in range.
        let isAudio = tokens.asType(.int32) .== MLXArray(audioId)
        let safeTokens = MLX.where(isAudio, MLXArray(Int32(0)), tokens.asType(.int32))
        let textEmbeddings = languageModel.model.embedTokens(safeTokens).reshaped(-1, hidden)

        if placeholders.isEmpty {
            return textEmbeddings.reshaped(b, t, hidden)
        }

        // For each flat position: which audio row feeds it, and whether it is a placeholder.
        var sourceRow = [Int32](repeating: 0, count: flatTokens.count)
        var isPlaceholder = [Bool](repeating: false, count: flatTokens.count)
        for (row, position) in placeholders.enumerated() {
            sourceRow[position] = Int32(row)
            isPlaceholder[position] = true
        }
        let gathered = take(features, MLXArray(sourceRow), axis: 0)
        let selector = MLXArray(isPlaceholder).reshaped(-1, 1)
        let merged = MLX.where(selector, gathered, textEmbeddings)
        return merged.reshaped(b, t, hidden)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var sanitized = SharpAIGraniteSpeechPlusSanitizer.sanitize(weights)
        // Tied embeddings have no separate head, so a stray head tensor has no module to load into.
        if configuration.textConfiguration.tieWordEmbeddings {
            sanitized["language_model.lm_head.weight"] = nil
        }
        return sanitized
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        let embeddings: MLXArray
        if let audio = input.audio {
            var features = audio.features
            if features.ndim == 2 {
                features = features.expandedDimensions(axis: 0)
            }
            embeddings = try mergeAudio(
                tokens: tokens, audioEmbeddings: encodeAudio(features))
        } else {
            embeddings = languageModel.model.embedTokens(tokens)
        }

        let logits = languageModel.logits(embeddings: embeddings, cache: cache)
        prefill.progress?(tokens.dim(1), tokens.dim(1))
        return .logits(LMOutput(logits: logits, state: nil))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        var tokens = input.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let logits = languageModel.logits(
            embeddings: languageModel.model.embedTokens(tokens), cache: cache)
        return LMOutput(logits: logits, state: state)
    }

    /// Text-only forward: token ids in, logits out.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var tokens = inputs
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        return languageModel.logits(
            embeddings: languageModel.model.embedTokens(tokens), cache: cache)
    }
}
