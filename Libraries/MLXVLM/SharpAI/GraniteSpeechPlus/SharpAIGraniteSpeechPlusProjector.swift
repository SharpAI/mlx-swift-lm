// SharpAI fork overlay: granite_speech_plus projector, a BLIP-2 Q-Former over fixed windows.
// Not part of upstream mlx-swift-lm.
//
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/modeling_granite_speech.py (GraniteSpeechEncoderProjector)
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py (Blip2QFormerModel, Blip2QFormerLayer)
//
// Only the query path is used, so the text-input branch of Blip2QFormerLayer (use_qformer_text_input)
// is not built. Dropout is the identity at inference. Q-Former attention has no mask.

import Foundation
import MLX
import MLXNN

/// Dense + GELU, as `Blip2QFormerIntermediate`.
final class SharpAIGraniteSpeechPlusQFormerIntermediate: Module {

    @ModuleInfo(key: "dense") var dense: Linear

    init(hidden: Int, intermediate: Int) {
        self._dense.wrappedValue = Linear(hidden, intermediate)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gelu(dense(x))
    }
}

/// Dense, then residual add and LayerNorm, as `Blip2QFormerOutput` and `Blip2QFormerSelfOutput`.
final class SharpAIGraniteSpeechPlusQFormerOutput: Module {

    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(inputWidth: Int, hidden: Int, eps: Float) {
        self._dense.wrappedValue = Linear(inputWidth, hidden)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, residual: MLXArray) -> MLXArray {
        layerNorm(dense(hidden) + residual)
    }
}

/// Multi-head attention. Keys and values come from `context` when it is given, else from `hidden`.
final class SharpAIGraniteSpeechPlusQFormerMultiHead: Module {

    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    init(hidden: Int, heads: Int, contextWidth: Int) {
        self.heads = heads
        self.headDim = hidden / heads
        self.scale = pow(Float(hidden / heads), -0.5)
        self._query.wrappedValue = Linear(hidden, hidden)
        self._key.wrappedValue = Linear(contextWidth, hidden)
        self._value.wrappedValue = Linear(contextWidth, hidden)
        super.init()
    }

    /// Self-attention when `context` is nil. Cross-attention otherwise.
    func callAsFunction(_ hidden: MLXArray, context: MLXArray?) -> MLXArray {
        let source = context ?? hidden
        let (b, t) = (hidden.dim(0), hidden.dim(1))
        let s = source.dim(1)
        let q = query(hidden).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let k = key(source).reshaped(b, s, heads, headDim).transposed(0, 2, 1, 3)
        let v = value(source).reshaped(b, s, heads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
        return out.transposed(0, 2, 1, 3).reshaped(b, t, -1)
    }
}

/// `Blip2QFormerAttention`: attention, then the residual output projection.
final class SharpAIGraniteSpeechPlusQFormerAttention: Module {

    @ModuleInfo(key: "attention") var attention: SharpAIGraniteSpeechPlusQFormerMultiHead
    @ModuleInfo(key: "output") var output: SharpAIGraniteSpeechPlusQFormerOutput

    init(hidden: Int, heads: Int, contextWidth: Int, eps: Float) {
        self._attention.wrappedValue = SharpAIGraniteSpeechPlusQFormerMultiHead(
            hidden: hidden, heads: heads, contextWidth: contextWidth)
        self._output.wrappedValue = SharpAIGraniteSpeechPlusQFormerOutput(
            inputWidth: hidden, hidden: hidden, eps: eps)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, context: MLXArray?) -> MLXArray {
        output(attention(hidden, context: context), residual: hidden)
    }
}

/// One Q-Former layer: self-attention, optional cross-attention, then the query feed-forward.
final class SharpAIGraniteSpeechPlusQFormerLayer: Module {

    @ModuleInfo(key: "attention") var attention: SharpAIGraniteSpeechPlusQFormerAttention
    @ModuleInfo(key: "crossattention") var crossAttention: SharpAIGraniteSpeechPlusQFormerAttention?
    @ModuleInfo(key: "intermediate_query") var intermediateQuery: SharpAIGraniteSpeechPlusQFormerIntermediate
    @ModuleInfo(key: "output_query") var outputQuery: SharpAIGraniteSpeechPlusQFormerOutput

    init(_ config: SharpAIGraniteSpeechPlusProjectorConfiguration, layerIndex: Int, encoderWidth: Int)
    {
        let hidden = config.hiddenSize
        let eps = config.layerNormEps
        self._attention.wrappedValue = SharpAIGraniteSpeechPlusQFormerAttention(
            hidden: hidden, heads: config.attentionHeads, contextWidth: hidden, eps: eps)
        if layerIndex % config.crossAttentionFrequency == 0 {
            self._crossAttention.wrappedValue = SharpAIGraniteSpeechPlusQFormerAttention(
                hidden: hidden, heads: config.attentionHeads, contextWidth: encoderWidth, eps: eps)
        } else {
            self._crossAttention.wrappedValue = nil
        }
        self._intermediateQuery.wrappedValue = SharpAIGraniteSpeechPlusQFormerIntermediate(
            hidden: hidden, intermediate: config.intermediateSize)
        self._outputQuery.wrappedValue = SharpAIGraniteSpeechPlusQFormerOutput(
            inputWidth: config.intermediateSize, hidden: hidden, eps: eps)
        super.init()
    }

    /// `queries` is (batch, query_count, hidden). `encoderStates` is (batch, window, encoder width).
    func callAsFunction(_ queries: MLXArray, encoderStates: MLXArray) -> MLXArray {
        var h = attention(queries, context: nil)
        if let cross = crossAttention {
            h = cross(h, context: encoderStates)
        }
        return outputQuery(intermediateQuery(h), residual: h)
    }
}

/// The Q-Former stack: an input LayerNorm, then the layers.
final class SharpAIGraniteSpeechPlusQFormer: Module {

    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: SharpAIGraniteSpeechPlusQFormerEncoder

    init(_ config: SharpAIGraniteSpeechPlusProjectorConfiguration, encoderWidth: Int) {
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._encoder.wrappedValue = SharpAIGraniteSpeechPlusQFormerEncoder(
            config, encoderWidth: encoderWidth)
        super.init()
    }

    func callAsFunction(_ queries: MLXArray, encoderStates: MLXArray) -> MLXArray {
        var h = layerNorm(queries)
        for layer in encoder.layers {
            h = layer(h, encoderStates: encoderStates)
        }
        return h
    }
}

final class SharpAIGraniteSpeechPlusQFormerEncoder: Module {

    @ModuleInfo(key: "layer") var layers: [SharpAIGraniteSpeechPlusQFormerLayer]

    init(_ config: SharpAIGraniteSpeechPlusProjectorConfiguration, encoderWidth: Int) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { index in
            SharpAIGraniteSpeechPlusQFormerLayer(
                config, layerIndex: index, encoderWidth: encoderWidth)
        }
        super.init()
    }
}

/// `GraniteSpeechEncoderProjector`: windows the encoder states, runs the Q-Former with the learned
/// queries on each window, then maps the queries to the language model width.
public final class SharpAIGraniteSpeechPlusProjector: Module {

    let windowSize: Int
    let queryCount: Int
    let qformerWidth: Int

    @ParameterInfo(key: "query") var query: MLXArray
    @ModuleInfo(key: "qformer") var qformer: SharpAIGraniteSpeechPlusQFormer
    @ModuleInfo(key: "linear") var linear: Linear

    public init(
        _ config: SharpAIGraniteSpeechPlusProjectorConfiguration,
        encoderWidth: Int, textWidth: Int, windowSize: Int, downsampleRate: Int
    ) {
        let queryCount = windowSize / downsampleRate
        self.windowSize = windowSize
        self.queryCount = queryCount
        self.qformerWidth = config.hiddenSize
        self._query.wrappedValue = MLXArray.zeros([1, queryCount, config.hiddenSize])
        self._qformer.wrappedValue = SharpAIGraniteSpeechPlusQFormer(
            config, encoderWidth: encoderWidth)
        self._linear.wrappedValue = Linear(config.hiddenSize, textWidth)
        super.init()
    }

    /// (batch, time, encoder width) -> (batch, blocks * queryCount, text width)
    public func callAsFunction(_ states: MLXArray) -> MLXArray {
        let (b, seq, dim) = (states.dim(0), states.dim(1), states.dim(2))
        let blocks = (seq + windowSize - 1) / windowSize
        let pad = blocks * windowSize - seq
        var padded = states
        if pad > 0 {
            padded = MLX.padded(states, widths: [0, .init((0, pad)), 0])
        }
        let windows = padded.reshaped(b * blocks, windowSize, dim)
        let queries = MLX.broadcast(query, to: [b * blocks, queryCount, qformerWidth])
        let encoded = qformer(queries, encoderStates: windows)
        return linear(encoded.reshaped(b, blocks * queryCount, qformerWidth))
    }
}
