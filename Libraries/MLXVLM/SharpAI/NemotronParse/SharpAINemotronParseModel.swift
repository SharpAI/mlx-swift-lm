// SharpAI fork overlay: nemotron_parse conditional generation model. Not part of upstream mlx-swift-lm.
//
// `encoder` (RADIO ViT + neck) produces decoder cross-attention states from pixels. `decoder` is the
// mBART-style stack. `lm_head` is untied (`tie_word_embeddings: false`). The reference's extra
// prediction heads are absent here (`num_extra_heads` defaults to 0, and the checkpoint has no
// `extra_heads` weights). Training losses are not ported.
//
// Generation: the encoder runs once in `prepare`, and the states are kept for the decode steps.
// The caller must start the token sequence with `decoder_start_token_id` (2 in this checkpoint).
// The image preprocessor (resize to a multiple of 16, normalization) is not part of this module.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Holds the encoder states between `prepare` and the decode steps. It is not a `Module`, so the
/// weight loader does not see it.
final class SharpAINemotronParseEncoderStateBox: @unchecked Sendable {
    var states: MLXArray?
}

public final class SharpAINemotronParse: Module, VLMModel, KVCacheDimensionProvider {

    public let config: SharpAINemotronParseConfiguration

    @ModuleInfo var encoder: SharpAINemotronParseVision.Neck
    @ModuleInfo var decoder: SharpAINemotronParseDecoder
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let encoderState = SharpAINemotronParseEncoderStateBox()

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: config.decoderAttentionHeads, count: config.decoderLayers) }
    public var loraLayers: [Module] { decoder.layers }

    public init(_ config: SharpAINemotronParseConfiguration) {
        self.config = config
        _encoder.wrappedValue = SharpAINemotronParseVision.Neck(
            config.encoder, width: config.decoderHiddenSize)
        _decoder.wrappedValue = SharpAINemotronParseDecoder(
            vocabularySize: config.vocabularySize,
            dim: config.decoderHiddenSize,
            heads: config.decoderAttentionHeads,
            layers: config.decoderLayers,
            ffnDim: config.ffnDim)
        _lmHead.wrappedValue = Linear(config.decoderHiddenSize, config.vocabularySize, bias: false)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAINemotronParseSanitize.sanitize(weights)
    }

    /// Encodes NCHW pixels (already normalized) into the decoder's cross-attention states
    /// `[batch, tokens + 1, decoder d_model]`.
    public func encode(pixelValues: MLXArray) throws -> MLXArray {
        try encoder(pixelValues)
    }

    /// Teacher-forced logits `[batch, length, vocabulary]` for `tokens` given `pixelValues`.
    /// No cache is used. This is the reference forward without loss.
    public func logits(_ tokens: MLXArray, pixelValues: MLXArray) throws -> MLXArray {
        let states = try encode(pixelValues: pixelValues)
        return lmHead(decoder(tokens, encoderStates: states, cache: nil))
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let pixels = input.image?.pixels else {
            throw SharpAINemotronParseError.missingImage
        }
        let tokens = input.text.tokens
        let states = try encode(pixelValues: pixels)
        encoderState.states = states
        let hidden = decoder(tokens, encoderStates: states, cache: cache)
        let total = tokens.dim(1)
        prefill.progress?(total, total)
        // Only the last position's logits are needed to pick the next token.
        let last = lmHead(hidden[0..., (total - 1) ..< total, 0...])
        return .logits(LMOutput(logits: last))
    }

    /// Decode step. Uses the encoder states from `prepare`, with the self-attention cache.
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        guard let states = encoderState.states else {
            preconditionFailure("nemotron_parse: prepare(_:cache:state:prefill:) must run before decoding")
        }
        return lmHead(decoder(inputs, encoderStates: states, cache: cache))
    }
}
