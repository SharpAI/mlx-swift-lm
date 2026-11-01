// SharpAI fork overlay: cohere_asr model for the LanguageModel pipeline. Not part of upstream
// mlx-swift-lm.
//
// prepare(_:cache:state:prefill:) runs the encoder on LMInput.audio, decodes the prompt tokens,
// and stores the encoder output in LMOutput.State. Later decoding steps read it from there.
// The model has no mutable per-request field, so inference stays read-only.
// Limits: the prompt is decoded in one pass (no prefill chunking); there is no padding mask,
// and mel features are not extracted here, so LMInput.audio must already hold log-mel frames.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAICohereAsrStateKeys {
    static let encoderStates = LMOutput.Key<MLXArray>("sharpai.cohere_asr.encoder_states")
}

/// The encoder and decoder, under the HF `model.` prefix.
final class SharpAICohereAsrCore: Module {

    @ModuleInfo(key: "encoder") var encoder: SharpAICohereAsrEncoder
    @ModuleInfo(key: "decoder") var decoder: SharpAICohereAsrDecoder

    init(_ configuration: SharpAICohereAsrConfiguration) throws {
        self._encoder.wrappedValue = try SharpAICohereAsrEncoder(
            configuration.encoderConfiguration)
        self._decoder.wrappedValue = try SharpAICohereAsrDecoder(configuration)
        super.init()
    }
}

public class SharpAICohereAsrForConditionalGeneration:
    Module, LanguageModel, KVCacheDimensionProvider
{

    public let configuration: SharpAICohereAsrConfiguration
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: SharpAICohereAsrCore
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(_ configuration: SharpAICohereAsrConfiguration) throws {
        self.configuration = configuration
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self._model.wrappedValue = try SharpAICohereAsrCore(configuration)
        self._projOut.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: true)
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAICohereAsrSanitizer.sanitize(weights, configuration: configuration)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let audio = input.audio else {
            throw SharpAICohereAsrError.missingAudioInput
        }
        var features = audio.features
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }

        let encoderStates = model.encoder(features)
        let encoderLimit = configuration.encoderConfiguration.maxPositionEmbeddings
        if encoderStates.dim(1) > encoderLimit {
            throw SharpAICohereAsrError.sequenceTooLong(
                length: encoderStates.dim(1), limit: encoderLimit)
        }

        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let length = tokens.dim(1)
        let offset = cache.first?.offset ?? 0
        let decoderLimit = configuration.maxPositionEmbeddings
        if offset + length > decoderLimit {
            throw SharpAICohereAsrError.sequenceTooLong(
                length: offset + length, limit: decoderLimit)
        }

        let hidden = model.decoder(tokens, encoderStates: encoderStates, cache: cache)
        let logits = projOut(hidden)
        prefill.progress?(length, length)

        var state = LMOutput.State()
        state[SharpAICohereAsrStateKeys.encoderStates] = encoderStates
        return .logits(LMOutput(logits: logits, state: state))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        guard let encoderStates = state?[SharpAICohereAsrStateKeys.encoderStates] else {
            preconditionFailure(SharpAICohereAsrError.missingEncoderStates.localizedDescription)
        }
        var tokens = input.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let hidden = model.decoder(tokens, encoderStates: encoderStates, cache: cache)
        return LMOutput(logits: projOut(hidden), state: state)
    }

    /// Text-only calls have no audio context, so this path is not supported. Use `prepare`.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        preconditionFailure(SharpAICohereAsrError.missingEncoderStates.localizedDescription)
    }
}
