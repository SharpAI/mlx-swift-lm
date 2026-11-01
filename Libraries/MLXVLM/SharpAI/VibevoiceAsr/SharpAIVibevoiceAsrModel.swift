// SharpAI fork overlay: vibevoice_asr model for the LanguageModel pipeline. Not part of
// upstream mlx-swift-lm.
//
// prepare(_:cache:state:prefill:) reads raw 24 kHz samples from LMInput.audio, encodes them with
// both tokenizers, and writes the projected features over the audio placeholder tokens. Then it
// runs the decoder over the whole prompt. Decoding steps are text-only, so no state is carried.
// Limits: one prompt per call; no padding mask; a prompt with a non-empty cache is re-encoded
// from its audio, not appended to the cache. UNVERIFIED: the sampled acoustic noise below is
// copied from the reference's get_audio_features and was not checked against its output.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `multi_modal_projector`: acoustic and semantic latents, each projected to the text width, then summed.
final class SharpAIVibevoiceAsrProjector: Module {

    @ModuleInfo(key: "acoustic_linear_1") var acousticLinear1: Linear
    @ModuleInfo(key: "acoustic_norm") var acousticNorm: RMSNorm
    @ModuleInfo(key: "acoustic_linear_2") var acousticLinear2: Linear
    @ModuleInfo(key: "semantic_linear_1") var semanticLinear1: Linear
    @ModuleInfo(key: "semantic_norm") var semanticNorm: RMSNorm
    @ModuleInfo(key: "semantic_linear_2") var semanticLinear2: Linear

    init(acousticSize: Int, semanticSize: Int, textSize: Int) {
        self._acousticLinear1.wrappedValue = Linear(acousticSize, textSize, bias: true)
        self._acousticNorm.wrappedValue = RMSNorm(dimensions: textSize, eps: 1e-6)
        self._acousticLinear2.wrappedValue = Linear(textSize, textSize, bias: true)
        self._semanticLinear1.wrappedValue = Linear(semanticSize, textSize, bias: true)
        self._semanticNorm.wrappedValue = RMSNorm(dimensions: textSize, eps: 1e-6)
        self._semanticLinear2.wrappedValue = Linear(textSize, textSize, bias: true)
        super.init()
    }

    func callAsFunction(acoustic: MLXArray, semantic: MLXArray) -> MLXArray {
        let a = acousticLinear2(acousticNorm(acousticLinear1(acoustic)))
        let s = semanticLinear2(semanticNorm(semanticLinear1(semantic)))
        return a + s
    }
}

public class SharpAIVibevoiceAsrForConditionalGeneration:
    Module, LanguageModel, KVCacheDimensionProvider
{

    public let configuration: SharpAIVibevoiceAsrConfiguration
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: SharpAIVibevoiceAsrCausalLM
    @ModuleInfo(key: "multi_modal_projector") var projector: SharpAIVibevoiceAsrProjector
    @ModuleInfo(key: "acoustic_tokenizer_encoder")
    var acousticEncoder: SharpAIVibevoiceAsrTokenizerEncoder
    @ModuleInfo(key: "semantic_tokenizer_encoder")
    var semanticEncoder: SharpAIVibevoiceAsrTokenizerEncoder

    public init(_ configuration: SharpAIVibevoiceAsrConfiguration) throws {
        self.configuration = configuration
        self.kvHeads = Array(
            repeating: configuration.text.kvHeads, count: configuration.text.hiddenLayers)
        self._languageModel.wrappedValue = SharpAIVibevoiceAsrCausalLM(configuration.text)
        self._projector.wrappedValue = SharpAIVibevoiceAsrProjector(
            acousticSize: configuration.acousticTokenizer.hiddenSize,
            semanticSize: configuration.semanticTokenizer.hiddenSize,
            textSize: configuration.text.hiddenSize)
        self._acousticEncoder.wrappedValue = try SharpAIVibevoiceAsrTokenizerEncoder(
            configuration.acousticTokenizer)
        self._semanticEncoder.wrappedValue = try SharpAIVibevoiceAsrTokenizerEncoder(
            configuration.semanticTokenizer)
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIVibevoiceAsrSanitizer.sanitize(weights, configuration: configuration)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let audio = input.audio else {
            return .tokens(input.text)
        }

        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        let audioFeatures = try encodeAudio(audio.features)
        let embeddings = try mergeAudio(
            tokens: tokens,
            embeddings: languageModel.model.embedTokens(tokens),
            audioFeatures: audioFeatures)

        let hidden = languageModel.model(embeddings: embeddings, cache: cache)
        let length = tokens.dim(1)
        prefill.progress?(length, length)
        return .logits(LMOutput(logits: languageModel.logits(hidden)))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        var tokens = input.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let embeddings = languageModel.model.embedTokens(tokens)
        let hidden = languageModel.model(embeddings: embeddings, cache: cache)
        return LMOutput(logits: languageModel.logits(hidden), state: state)
    }

    /// Text-only step with no audio. Use `prepare` for prompts that contain audio.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var tokens = inputs
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let embeddings = languageModel.model.embedTokens(tokens)
        return languageModel.logits(languageModel.model(embeddings: embeddings, cache: cache))
    }

    /// Converts raw samples to (batch, samples, 1). Accepts (samples), (batch, samples) or (batch, 1, samples).
    static func waveformNLC(_ samples: MLXArray) throws -> MLXArray {
        switch samples.ndim {
        case 1:
            return samples.reshaped(1, samples.dim(0), 1)
        case 2:
            return samples.expandedDimensions(axis: 2)
        case 3 where samples.dim(1) == 1:
            return samples.transposed(0, 2, 1)
        default:
            throw SharpAIVibevoiceAsrError.unsupportedAudioShape(samples.shape)
        }
    }

    /// Returns one row per audio frame: (batch * frames, textHiddenSize).
    func encodeAudio(_ samples: MLXArray) throws -> MLXArray {
        var waveform = try Self.waveformNLC(samples)

        // The reference feature extractor zero-pads to a multiple of the hop, so frames = samples / hop.
        let hop = configuration.acousticTokenizer.hopLength
        let remainder = waveform.dim(1) % hop
        if remainder != 0 {
            let padding = MLXArray.zeros(
                [waveform.dim(0), hop - remainder, 1], dtype: waveform.dtype)
            waveform = concatenated([waveform, padding], axis: 1)
        }

        let acoustic = acousticEncoder(waveform)
        let semantic = semanticEncoder(waveform)

        // Reference: noise_std = vae_std * randn(batch), then acoustic += noise_std * randn_like.
        let batch = acoustic.dim(0)
        let noiseScale = MLXRandom.normal([batch]) * configuration.acousticTokenizer.vaeStd
        let noisy = acoustic + noiseScale.reshaped(batch, 1, 1) * MLXRandom.normal(acoustic.shape)

        let features = projector(acoustic: noisy, semantic: semantic)
        return features.reshaped(-1, features.dim(2))
    }

    /// Replaces each audio placeholder row with the next audio frame, in row-major order.
    /// Same result as the reference's masked_scatter.
    func mergeAudio(tokens: MLXArray, embeddings: MLXArray, audioFeatures: MLXArray) throws
        -> MLXArray
    {
        let isAudio = tokens .== MLXArray(configuration.audioTokenId)
        let placeholders = isAudio.asType(.int32).sum().asArray(Int.self).first ?? 0
        let frames = audioFeatures.dim(0)
        guard placeholders == frames else {
            throw SharpAIVibevoiceAsrError.audioTokenMismatch(
                placeholders: placeholders, frames: frames)
        }

        let flatIsAudio = isAudio.asType(.int32).reshaped(-1)
        let order = (cumsum(flatIsAudio, axis: 0) - 1).reshaped(tokens.dim(0), tokens.dim(1))
        let rowIndex = maximum(order, MLXArray(0))
        let gathered = take(audioFeatures, rowIndex, axis: 0).asType(embeddings.dtype)
        let mask = isAudio.expandedDimensions(axis: -1)
        return MLX.where(mask, gathered, embeddings)
    }
}
