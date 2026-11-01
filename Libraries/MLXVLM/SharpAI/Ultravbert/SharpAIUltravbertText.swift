// SharpAI fork overlay: UltraBERT text encoder for UltraVBERT. Not part of upstream mlx-swift-lm.
//
// Sources (transformers v5.x, main branch):
//   - `models/ultrabert/modular_ultrabert.py`: subclasses EuroBERT; RMSNorm; `layer_types` cycle
//     `sliding_attention` unless `(i + 1) % 3 == 0`; RoPE bases 10_000 (sliding) and
//     2_560_000 (full); bidirectional masks; no KV cache; final norm named `norm`.
//   - `models/llama/modeling_llama.py` (EuroBERT inherits Llama): pre-norm layers with
//     `input_layernorm` and `post_attention_layernorm`; bias-free q/k/v/o unless
//     `attention_bias`; `scaling = head_dim ** -0.5`; `down(silu(gate(x)) * up(x))`;
//     `rotate_half` applied to q and k only.
//   - `models/modernbert/modeling_modernbert.py`: `rotate_half` split-in-half formula and
//     `inv_freq = 1 / base ** (arange(0, dim, 2) / dim)`.
//
// Layer 0 reads the raw token embeddings (no embedding norm). Each layer still applies its own
// pre-attention norm, as in Llama.
//
// UNVERIFIED: the sliding-window boundary. The config docstring says an inclusive radius of
// `sliding_window` tokens on each side, so this uses `|i - j| <= sliding_window`. The comparison
// inside `create_bidirectional_sliding_window_mask` was not read.

import Foundation
import MLX
import MLXNN

/// Text-tower hyperparameters for `model_type` "ultrabert". Defaults are the UltrabertConfig
/// defaults. Only `text_config` keys are read; the caller passes that nested object.
public struct SharpAIUltravbertTextConfiguration: Decodable, Sendable {

    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let rmsNormEps: Float
    public let slidingWindow: Int
    public let attentionBias: Bool
    public let mlpBias: Bool
    /// `rope_parameters` for `sliding_attention` (default in UltrabertConfig: 10_000).
    public let slidingRopeTheta: Float
    /// `rope_parameters` for `full_attention` (default in UltrabertConfig: 2_560_000).
    public let fullRopeTheta: Float

    public init(
        vocabularySize: Int = 32000,
        hiddenSize: Int = 4096,
        intermediateSize: Int = 11008,
        hiddenLayers: Int = 32,
        attentionHeads: Int = 32,
        rmsNormEps: Float = 1e-6,
        slidingWindow: Int = 128,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        slidingRopeTheta: Float = 10_000,
        fullRopeTheta: Float = 2_560_000
    ) {
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.slidingWindow = slidingWindow
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.slidingRopeTheta = slidingRopeTheta
        self.fullRopeTheta = fullRopeTheta
    }

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: any Decoder) throws {
        let defaults = SharpAIUltravbertTextConfiguration()
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? defaults.vocabularySize
        hiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? defaults.hiddenSize
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? defaults.intermediateSize
        hiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? defaults.hiddenLayers
        attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? defaults.attentionHeads
        rmsNormEps =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? defaults.rmsNormEps
        slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
            ?? defaults.slidingWindow
        attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
            ?? defaults.attentionBias
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? defaults.mlpBias
        slidingRopeTheta = defaults.slidingRopeTheta
        fullRopeTheta = defaults.fullRopeTheta
    }
}

/// Bidirectional UltraBERT encoder. Checkpoint paths: `embed_tokens`, `layers.N.self_attn.*`,
/// `layers.N.mlp.*`, `norm` (relative to the text model root).
final class SharpAIUltravbertTextModel: Module {

    /// Layers at 0-based index 2, 5, 8, ... attend globally. The rest use the sliding window.
    static func isGlobalLayer(_ index: Int) -> Bool {
        (index + 1) % 3 == 0
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float
        let ropeBase: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: SharpAIUltravbertTextConfiguration, isGlobal: Bool) {
            self.heads = config.attentionHeads
            self.headDim = config.hiddenSize / config.attentionHeads
            self.scale = pow(Float(config.hiddenSize / config.attentionHeads), -0.5)
            self.ropeBase = isGlobal ? config.fullRopeTheta : config.slidingRopeTheta
            self._qProj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
            self._kProj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
            self._vProj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
            self._oProj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
        }

        /// `mask` is additive, broadcastable to `[batch, heads, seq, seq]`, or nil.
        func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            var keys = kProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)

            // traditional: false is the half-split rotate_half form used by the source.
            queries = MLXFast.RoPE(
                queries, dimensions: headDim, traditional: false, base: ropeBase, scale: 1,
                offset: 0)
            keys = MLXFast.RoPE(
                keys, dimensions: headDim, traditional: false, base: ropeBase, scale: 1,
                offset: 0)

            let output: MLXArray
            if let mask {
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale,
                    mask: .array(mask.asType(queries.dtype)))
            } else {
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: .none)
            }

            return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    /// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
    final class MLP: Module {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: SharpAIUltravbertTextConfiguration) {
            self._gateProj.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
            self._upProj.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
            self._downProj.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
        @ModuleInfo var mlp: MLP

        let isGlobal: Bool

        init(_ config: SharpAIUltravbertTextConfiguration, layerIndex: Int) {
            let global = SharpAIUltravbertTextModel.isGlobalLayer(layerIndex)
            self.isGlobal = global
            self._inputLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._selfAttn.wrappedValue = Attention(config, isGlobal: global)
            self._postAttentionLayernorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
            var h = x + selfAttn(inputLayernorm(x), mask: mask)
            h = h + mlp(postAttentionLayernorm(h))
            return h
        }
    }

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let slidingWindow: Int

    init(_ config: SharpAIUltravbertTextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { index in
            DecoderLayer(config, layerIndex: index)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.slidingWindow = config.slidingWindow
    }

    /// Encodes token ids `[batch, seq]` into hidden states `[batch, seq, hidden]`.
    /// `paddingMask` is Bool `[batch, seq]`, true for real tokens. Nil means no padding.
    func callAsFunction(_ inputIds: MLXArray, paddingMask: MLXArray?) -> MLXArray {
        let (B, L) = (inputIds.dim(0), inputIds.dim(1))
        let pad = paddingMask.map { $0.reshaped(B, 1, 1, L) }

        let indices = MLXArray(0 ..< L)
        let distance = abs(
            indices.expandedDimensions(axis: 1) - indices.expandedDimensions(axis: 0))
        let band = (distance .<= MLXArray(slidingWindow)).reshaped(1, 1, L, L)

        let globalMask: MLXArray? = pad.map { additiveMask($0) }
        let localMask = additiveMask(pad.map { band & $0 } ?? band)

        var h = embedTokens(inputIds)
        for layer in layers {
            h = layer(h, mask: layer.isGlobal ? globalMask : localMask)
        }
        return norm(h)
    }

    /// Disallowed positions get a large negative value that stays finite after the cast to
    /// float16, so a row with no allowed key softmaxes to uniform instead of NaN.
    static let maskedValue: Float = -30000

    static func additiveMask(_ allowed: MLXArray) -> MLXArray {
        which(allowed, MLXArray(Float(0)), MLXArray(maskedValue))
    }

    private func additiveMask(_ allowed: MLXArray) -> MLXArray {
        Self.additiveMask(allowed)
    }
}
