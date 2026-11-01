// SharpAI fork overlay: moss_audio model (audio encoder, adapter, DeepStack, Qwen3 backbone).
// Not part of upstream mlx-swift-lm.
//
// Checkpoint names follow vLLM's moss_audio.py load mapping: `audio_encoder.*`, `audio_adapter.*`,
// `deepstack_audio_merger_list.*`, `language_model.embed_tokens`, `language_model.layers.N.*`,
// `language_model.norm`, and a top-level `lm_head`.
// The text decoder is written here because MLXLLM's Qwen3 blocks are internal and cannot take
// embeddings or DeepStack injections.
// DeepStack: after decoder layer i (i < number of mergers), the i-th merged tap is added at the
// audio positions. Text positions get zeros, so they are unchanged.
// UNVERIFIED: the adapter and merger biases, and the DeepStack merger hidden size (taken as
// adapter_hidden_size). Time markers (enable_time_marker) and the processor are not written.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Gated MLP `down(silu(gate(x)) * up(x))`, used for the adapter and the DeepStack mergers.
final class SharpAIMossAudioGatedMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        _gate.wrappedValue = Linear(inputSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(inputSize, hiddenSize, bias: false)
        _down.wrappedValue = Linear(hiddenSize, outputSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIMossAudioTextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAIMossAudioTextConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope = RoPE(
            dimensions: config.headDim, traditional: false, base: config.ropeTheta, scale: 1)
        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qNorm(qProj(x).reshaped(batch, length, heads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return oProj(output)
    }
}

final class SharpAIMossAudioTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIMossAudioTextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIMossAudioTextAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIMossAudioTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIMossAudioTextConfiguration) {
        _attention.wrappedValue = SharpAIMossAudioTextAttention(config)
        _mlp.wrappedValue = SharpAIMossAudioTextMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The `language_model.*` part of the Qwen3 backbone. It takes embeddings and DeepStack taps.
final class SharpAIMossAudioTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIMossAudioTextLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIMossAudioTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIMossAudioTextLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func embed(_ tokens: MLXArray) -> MLXArray {
        embedTokens(tokens)
    }

    /// `deepstack[i]` is `[1, length, hidden]` and is added after decoder layer `i`.
    func forward(_ embeddings: MLXArray, cache: [KVCache]?, deepstack: [MLXArray]) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index])
            if index < deepstack.count {
                h = h + deepstack[index].asType(h.dtype)
            }
        }
        return norm(h)
    }
}

public class SharpAIMossAudio: Module, VLMModel, KVCacheDimensionProvider {
    /// `<|AUDIO|>` placeholder id from vLLM's processor defaults.
    public static let defaultAudioTokenId = 151654

    public let config: SharpAIMossAudioConfiguration

    @ModuleInfo(key: "audio_encoder") private var audioEncoder: SharpAIMossAudioEncoder
    @ModuleInfo(key: "audio_adapter") private var audioAdapter: SharpAIMossAudioGatedMLP
    @ModuleInfo(key: "deepstack_audio_merger_list") private var deepstackMergers:
        [SharpAIMossAudioGatedMLP]
    @ModuleInfo(key: "language_model") private var languageModel: SharpAIMossAudioTextModel
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public var kvHeads: [Int] {
        let text = config.textConfiguration
        return (0 ..< text.hiddenLayers).map { _ in text.kvHeads }
    }

    public var loraLayers: [Module] { languageModel.layers }

    public init(_ config: SharpAIMossAudioConfiguration) {
        self.config = config
        let text = config.textConfiguration
        let audio = config.audioConfiguration
        _audioEncoder.wrappedValue = SharpAIMossAudioEncoder(audio)
        _audioAdapter.wrappedValue = SharpAIMossAudioGatedMLP(
            inputSize: audio.outputDim, hiddenSize: config.adapterHiddenSize,
            outputSize: text.hiddenSize)
        _deepstackMergers.wrappedValue = (0 ..< config.deepstackMergerCount).map { _ in
            SharpAIMossAudioGatedMLP(
                inputSize: audio.dModel, hiddenSize: config.adapterHiddenSize,
                outputSize: text.hiddenSize)
        }
        _languageModel.wrappedValue = SharpAIMossAudioTextModel(text)
        if !text.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabularySize, bias: false)
        }
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let total = tokens.dim(1)
        let output: MLXArray
        if let audio = input.audio {
            output = try multimodalLogits(
                tokens: tokens, audioFeatures: audio.features, cache: cache)
        } else {
            output = project(
                languageModel.forward(
                    languageModel.embed(tokens), cache: cache, deepstack: []))
        }
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: output))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        project(languageModel.forward(languageModel.embed(inputs), cache: cache, deepstack: []))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = SharpAIMossAudioSanitize.sanitize(weights: weights)
        if config.textConfiguration.tieWordEmbeddings {
            result = result.filter { !$0.key.hasPrefix("lm_head.") }
        }
        return result
    }

    /// Prefill logits `[1, length, vocab]` for `tokens` `[1, length]`, with the encoded
    /// `audioFeatures` `[1, numMelBins, frames]` spliced in at the `audioTokenId` placeholders.
    public func multimodalLogits(
        tokens: MLXArray, audioFeatures: MLXArray,
        audioTokenId: Int = SharpAIMossAudio.defaultAudioTokenId,
        cache: [any KVCache]? = nil
    ) throws -> MLXArray {
        guard tokens.ndim == 2, tokens.dim(0) == 1 else {
            throw SharpAIMossAudioError.invalidInput(
                "tokens must be [1, length], got \(tokens.shape)")
        }
        let ids = tokens[0].asArray(Int.self)
        let textEmbeddings = languageModel.embed(tokens)
        let encoded = try encodeAudio(audioFeatures)
        let embeddings = try scatter(
            ids, audioTokenId: audioTokenId, values: encoded.main, base: textEmbeddings)
        let zeros = MLXArray.zeros(textEmbeddings.shape, dtype: textEmbeddings.dtype)
        var deepstack: [MLXArray] = []
        for tap in encoded.taps {
            deepstack.append(
                try scatter(ids, audioTokenId: audioTokenId, values: tap, base: zeros))
        }
        return project(
            languageModel.forward(embeddings, cache: cache, deepstack: deepstack))
    }

    private func project(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return languageModel.embedTokens.asLinear(hidden)
    }

    /// Returns the main audio rows `[tokens, hidden]` and one DeepStack row block per merger.
    private func encodeAudio(_ features: MLXArray) throws -> (main: MLXArray, taps: [MLXArray]) {
        let audio = config.audioConfiguration
        guard features.ndim == 3, features.dim(0) == 1, features.dim(1) == audio.numMelBins else {
            throw SharpAIMossAudioError.invalidInput(
                "audio features must be [1, \(audio.numMelBins), frames], got \(features.shape)")
        }
        let tapLayers = Array(audio.deepstackEncoderLayerIndexes.prefix(deepstackMergers.count))
        let encoded = audioEncoder(features, deepstackLayers: tapLayers)
        guard encoded.output.dim(1) > 0 else {
            throw SharpAIMossAudioError.invalidInput("audio has no encoder tokens")
        }
        let main = audioAdapter(encoded.output)[0]
        var taps: [MLXArray] = []
        for (index, merger) in deepstackMergers.enumerated() where index < encoded.taps.count {
            taps.append(merger(encoded.taps[index])[0])
        }
        return (main, taps)
    }

    /// Replaces each run of `audioTokenId` positions in `base` `[1, length, hidden]` with the next
    /// rows of `values` `[count, hidden]`, in order.
    private func scatter(
        _ ids: [Int], audioTokenId: Int, values: MLXArray, base: MLXArray
    ) throws -> MLXArray {
        let placeholders = ids.filter { $0 == audioTokenId }.count
        guard placeholders == values.dim(0) else {
            throw SharpAIMossAudioError.audioTokenMismatch(
                placeholders: placeholders, features: values.dim(0))
        }
        let rows = values.asType(base.dtype)
        var parts: [MLXArray] = []
        var textStart = 0
        var audioOffset = 0
        var index = 0
        while index < ids.count {
            guard ids[index] == audioTokenId else {
                index += 1
                continue
            }
            if textStart < index {
                parts.append(base[0..., textStart ..< index, 0...])
            }
            var runEnd = index
            while runEnd < ids.count, ids[runEnd] == audioTokenId {
                runEnd += 1
            }
            let runLength = runEnd - index
            parts.append(
                rows[audioOffset ..< (audioOffset + runLength)].expandedDimensions(axis: 0))
            audioOffset += runLength
            textStart = runEnd
            index = runEnd
        }
        if textStart < ids.count {
            parts.append(base[0..., textStart ..< ids.count, 0...])
        }
        return concatenated(parts, axis: 1)
    }
}
