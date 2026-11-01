// SharpAI fork overlay: qwen3_asr models for the LanguageModel pipeline. Not part of upstream
// mlx-swift-lm.
//
// prepare(_:cache:state:prefill:) encodes LMInput.audio, writes the encoder rows over the
// audio placeholder tokens, and runs the decoder over the whole prompt. Later steps need only
// the cache, so no LMOutput.State is carried.
// Limits: LMInput.audio must already hold log-mel features (1, mel, frames). The prompt is
// decoded in one pass with no prefill chunking. Only the ASR head and the forced aligner head
// are built. The streaming path and the server routes are not in this file.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Shared prompt embedding for the ASR model and the forced aligner.
enum SharpAIQwen3AsrInputs {

    /// Text embeddings with the encoder rows written over each audio placeholder run.
    /// Returns (1, length, hidden).
    static func embeddings(
        tokens rawTokens: MLXArray,
        audio: LMInput.ProcessedAudio?,
        audioTower: SharpAIQwen3AsrAudioEncoder,
        model: SharpAIQwen3AsrTextModel,
        configuration: SharpAIQwen3AsrConfiguration
    ) throws -> MLXArray {
        var tokens = rawTokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        guard tokens.dim(0) == 1 else {
            throw SharpAIQwen3AsrError.unsupportedBatch(tokens.dim(0))
        }

        let embeds = model.embed(tokens)
        let ids = tokens.asArray(Int32.self)
        let placeholder = Int32(configuration.audioTokenId)
        let positions = ids.indices.filter { ids[$0] == placeholder }

        guard let audio else {
            if !positions.isEmpty {
                throw SharpAIQwen3AsrError.unexpectedAudio
            }
            return embeds
        }

        var features = audio.features
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }
        let frames = features.dim(2)
        let valid =
            audio.seqLengths?.first
            ?? audio.mask?.asType(.int32).sum(axis: -1).asArray(Int.self).first
            ?? frames

        let encoded = try audioTower(features, validFrames: valid).asType(embeds.dtype)
        guard encoded.dim(0) == positions.count else {
            throw SharpAIQwen3AsrError.audioTokenMismatch(
                placeholders: positions.count, encoderRows: encoded.dim(0))
        }
        guard encoded.dim(1) == embeds.dim(2) else {
            throw SharpAIQwen3AsrError.invalidInput(
                "audio width \(encoded.dim(1)) does not match hidden size \(embeds.dim(2))")
        }

        // Splice audio rows in place of each contiguous placeholder run.
        let length = embeds.dim(1)
        var pieces: [MLXArray] = []
        var cursor = 0
        var audioCursor = 0
        var index = 0
        while index < positions.count {
            let runStart = positions[index]
            var runEnd = index
            while runEnd + 1 < positions.count && positions[runEnd + 1] == positions[runEnd] + 1 {
                runEnd += 1
            }
            let runLength = runEnd - index + 1
            if runStart > cursor {
                pieces.append(embeds[0, cursor ..< runStart])
            }
            pieces.append(encoded[audioCursor ..< (audioCursor + runLength)])
            audioCursor += runLength
            cursor = runStart + runLength
            index = runEnd + 1
        }
        if cursor < length {
            pieces.append(embeds[0, cursor ..< length])
        }
        return MLX.concatenated(pieces, axis: 0).expandedDimensions(axis: 0)
    }

    /// Token ids for a step after the prompt. Adds the batch axis when it is missing.
    static func batchedTokens(_ tokens: MLXArray) -> MLXArray {
        tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
    }
}

/// Qwen3-ASR speech-to-text model. Hub key layout: audio_tower.*, model.*, lm_head.*.
public class SharpAIQwen3AsrForConditionalGeneration:
    Module, LanguageModel, KVCacheDimensionProvider
{

    public let configuration: SharpAIQwen3AsrConfiguration
    public let kvHeads: [Int]

    @ModuleInfo(key: "audio_tower") var audioTower: SharpAIQwen3AsrAudioEncoder
    @ModuleInfo(key: "model") var model: SharpAIQwen3AsrTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIQwen3AsrConfiguration) {
        self.configuration = configuration
        self.kvHeads = Array(
            repeating: configuration.textConfiguration.kvHeads,
            count: configuration.textConfiguration.hiddenLayers)
        self._audioTower.wrappedValue = SharpAIQwen3AsrAudioEncoder(
            configuration.audioConfiguration)
        self._model.wrappedValue = SharpAIQwen3AsrTextModel(configuration.textConfiguration)
        if configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(
                configuration.textConfiguration.hiddenSize,
                configuration.textConfiguration.vocabularySize, bias: false)
        }
        super.init()
    }

    func projectLogits(_ hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: configuration)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let embeds = try SharpAIQwen3AsrInputs.embeddings(
            tokens: input.text.tokens,
            audio: input.audio,
            audioTower: audioTower,
            model: model,
            configuration: configuration)
        let length = embeds.dim(1)
        let hidden = model(embeds, cache: cache)
        prefill.progress?(length, length)
        return .logits(LMOutput(logits: projectLogits(hidden)))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = SharpAIQwen3AsrInputs.batchedTokens(input.tokens)
        let hidden = model(model.embed(tokens), cache: cache)
        return LMOutput(logits: projectLogits(hidden), state: state)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = SharpAIQwen3AsrInputs.batchedTokens(inputs)
        return projectLogits(model(model.embed(tokens), cache: cache))
    }
}

/// Qwen3-ASR forced aligner. It shares the audio tower and decoder with the ASR model and has a
/// classification head in place of the LM head. The head predicts time bins at timestamp tokens.
/// UNVERIFIED: the LanguageModel conformance exists so the creator can return it. The host
/// should read the logits as bin scores, not as next-token logits.
public class SharpAIQwen3AsrForcedAlignerForTokenClassification:
    Module, LanguageModel, KVCacheDimensionProvider
{

    public let configuration: SharpAIQwen3AsrConfiguration
    public let kvHeads: [Int]

    @ModuleInfo(key: "audio_tower") var audioTower: SharpAIQwen3AsrAudioEncoder
    @ModuleInfo(key: "model") var model: SharpAIQwen3AsrTextModel
    @ModuleInfo(key: "classifier") var classifier: Linear

    public init(_ configuration: SharpAIQwen3AsrConfiguration) {
        self.configuration = configuration
        self.kvHeads = Array(
            repeating: configuration.textConfiguration.kvHeads,
            count: configuration.textConfiguration.hiddenLayers)
        self._audioTower.wrappedValue = SharpAIQwen3AsrAudioEncoder(
            configuration.audioConfiguration)
        self._model.wrappedValue = SharpAIQwen3AsrTextModel(configuration.textConfiguration)
        self._classifier.wrappedValue = Linear(
            configuration.textConfiguration.hiddenSize, configuration.classifyNum, bias: false)
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIQwen3AsrSanitizer.sanitize(weights, configuration: configuration)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let embeds = try SharpAIQwen3AsrInputs.embeddings(
            tokens: input.text.tokens,
            audio: input.audio,
            audioTower: audioTower,
            model: model,
            configuration: configuration)
        let length = embeds.dim(1)
        let hidden = model(embeds, cache: cache)
        prefill.progress?(length, length)
        return .logits(LMOutput(logits: classifier(hidden)))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let tokens = SharpAIQwen3AsrInputs.batchedTokens(input.tokens)
        let hidden = model(model.embed(tokens), cache: cache)
        return LMOutput(logits: classifier(hidden), state: state)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = SharpAIQwen3AsrInputs.batchedTokens(inputs)
        return classifier(model(model.embed(tokens), cache: cache))
    }
}
