// SharpAI fork overlay: voxtral_realtime model for the LanguageModel pipeline. Not part of
// upstream mlx-swift-lm.
//
// prepare(_:cache:state:prefill:) encodes LMInput.audio, then decodes the prompt tokens. Token
// position p gets the embedding of token p plus audio frame p, when that frame exists. The
// projected audio is kept in LMOutput.State, so later decoding steps can add the next frames.
// Limits: LMInput.audio must already hold log-mel features (no feature extractor here), there is
// no padding mask, and prefill runs in chunks.
// Only the last prompt position's logits are returned from prepare.
// UNVERIFIED: the per-position alignment of prompt tokens and audio frames is not checked
// against transformers' realtime generation loop.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAIVoxtralRealtimeStateKeys {
    static let audioEmbeddings = LMOutput.Key<MLXArray>("sharpai.voxtral_realtime.audio")
}

/// Audio frames for token positions `offset ..< offset + length`. Positions past the audio get
/// zeros. Returns nil when no audio frame falls in the range.
// UNVERIFIED: token p paired with audio frame p is an assumption. No source confirms it, and the
// realtime delay-token handling is not implemented.
enum SharpAIVoxtralRealtimeAlignment {
    static func frames(_ audio: MLXArray, offset: Int, length: Int) -> MLXArray? {
        let available = audio.dim(1)
        guard offset < available else {
            return nil
        }
        let end = min(offset + length, available)
        var frames = audio[0..., offset ..< end]
        if end - offset < length {
            frames = MLX.padded(frames, widths: [0, .init((0, length - (end - offset))), 0])
        }
        return frames
    }
}

public class SharpAIVoxtralRealtimeForConditionalGeneration:
    Module, LanguageModel, KVCacheDimensionProvider
{

    public let configuration: SharpAIVoxtralRealtimeConfiguration
    public let kvHeads: [Int]

    @ModuleInfo(key: "audio_tower") var audioTower: SharpAIVoxtralRealtimeAudioEncoder
    @ModuleInfo(key: "multi_modal_projector") var projector: SharpAIVoxtralRealtimeProjector
    @ModuleInfo(key: "language_model") var languageModel: SharpAIVoxtralRealtimeLanguageModel

    public init(_ configuration: SharpAIVoxtralRealtimeConfiguration) {
        self.configuration = configuration
        self.kvHeads = Array(
            repeating: configuration.textConfiguration.kvHeads,
            count: configuration.textConfiguration.hiddenLayers)
        self._audioTower.wrappedValue = SharpAIVoxtralRealtimeAudioEncoder(
            configuration.audioConfiguration)
        self._projector.wrappedValue = SharpAIVoxtralRealtimeProjector(configuration)
        self._languageModel.wrappedValue = SharpAIVoxtralRealtimeLanguageModel(
            configuration.textConfiguration, adaNormDimension: configuration.adaNormDimension)
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIVoxtralRealtimeSanitizer.sanitize(weights, configuration: configuration)
    }

    /// The decoder attends over a sliding window, so every layer gets a rotating cache.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try (0 ..< configuration.textConfiguration.hiddenLayers).map { _ in
            try makeSlidingWindowKVCache(
                parameters: parameters, window: configuration.textConfiguration.slidingWindow)
        }
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let audio = input.audio else {
            throw SharpAIVoxtralRealtimeError.missingAudioInput
        }
        var features = audio.features
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }
        let audioEmbeddings = projector(audioTower(features))
        guard audioEmbeddings.dim(1) > 0 else {
            throw SharpAIVoxtralRealtimeError.missingAudioInput
        }

        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let length = tokens.dim(1)
        let processed = try prefill.forEachChunk(total: length) { range in
            _ = decode(tokens[0..., range], audio: audioEmbeddings, cache: cache)
            asyncEval(cache)
        }
        if processed > 0 {
            eval(cache)
        }
        let hidden = decode(
            tokens[0..., processed ..< length], audio: audioEmbeddings, cache: cache)
        prefill.progress?(length, length)

        var state = LMOutput.State()
        state[SharpAIVoxtralRealtimeStateKeys.audioEmbeddings] = audioEmbeddings
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        return .logits(LMOutput(logits: logits(last), state: state))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let audio = state?[SharpAIVoxtralRealtimeStateKeys.audioEmbeddings]
        var tokens = input.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let hidden = decode(tokens, audio: audio, cache: cache)
        return LMOutput(logits: logits(hidden), state: state)
    }

    /// Text-only path. No audio is added, so the output has no audio context.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var tokens = inputs
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        return logits(decode(tokens, audio: nil, cache: cache))
    }

    /// Embeds tokens, adds the aligned audio frames, and runs the decoder with the delay
    /// condition.
    private func decode(_ tokens: MLXArray, audio: MLXArray?, cache: [KVCache]?) -> MLXArray {
        let decoder = languageModel.model
        var embeddings = decoder.embedTokens(tokens)
        let dtype = embeddings.dtype
        if let audio {
            let offset = cache?.first?.offset ?? 0
            if let frames = SharpAIVoxtralRealtimeAlignment.frames(
                audio, offset: offset, length: tokens.dim(1))
            {
                embeddings = (embeddings + frames).asType(dtype)
            }
        }
        let condition = SharpAIVoxtralRealtimeDelay.embedding(
            configuration.defaultNumDelayTokens,
            dimension: configuration.textConfiguration.hiddenSize
        ).asType(dtype)
        return decoder(embeddings, condition: condition, cache: cache)
    }

    /// Tied output head: the embedding matrix is reused.
    private func logits(_ hidden: MLXArray) -> MLXArray {
        languageModel.model.embedTokens.asLinear(hidden)
    }
}
