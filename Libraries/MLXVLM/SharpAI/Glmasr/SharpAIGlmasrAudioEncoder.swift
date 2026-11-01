// SharpAI fork overlay: glmasr audio encoder. Not part of upstream mlx-swift-lm.
// Follows transformers' GlmAsrEncoder: conv1 (k3, s1) and conv2 (k3, s2) with gelu, then pre-LN
// blocks with partial RoPE (half of each head) and bidirectional, unmasked attention, then a
// final LayerNorm. Module keys match the HF names under `audio_tower.`.
// UNVERIFIED: the encoder RoPE base and layout (rotate-half vs interleaved pairs) are not in the sources.
// The repo's GLM4 uses interleaved RoPE, so check modeling_glmasr.py before using real checkpoints.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Cos and sin tables `[1, 1, positions, rotaryDim]` for the first `rotaryDim` channels.
func sharpAIGlmasrRotaryTables(positions: Int, rotaryDim: Int, theta: Float) -> (MLXArray, MLXArray) {
    let half = rotaryDim / 2
    var cosValues = [Float](repeating: 0, count: positions * rotaryDim)
    var sinValues = [Float](repeating: 0, count: positions * rotaryDim)
    for position in 0 ..< positions {
        for i in 0 ..< half {
            let inverse = pow(Double(theta), -Double(2 * i) / Double(rotaryDim))
            let angle = Double(position) * inverse
            let c = Float(cos(angle))
            let s = Float(sin(angle))
            // UNVERIFIED: both halves share one angle (rotate-half). See header.
            cosValues[position * rotaryDim + i] = c
            cosValues[position * rotaryDim + i + half] = c
            sinValues[position * rotaryDim + i] = s
            sinValues[position * rotaryDim + i + half] = s
        }
    }
    return (
        MLXArray(cosValues, [1, 1, positions, rotaryDim]),
        MLXArray(sinValues, [1, 1, positions, rotaryDim])
    )
}

/// Rotates the first `cos.dim(3)` channels of `[batch, heads, positions, headDim]`.
func sharpAIGlmasrApplyPartialRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let rotaryDim = cos.dim(3)
    let headDim = x.dim(3)
    let rot = x[0..., 0..., 0..., 0 ..< rotaryDim]
    let half = rotaryDim / 2
    let first = rot[0..., 0..., 0..., 0 ..< half]
    let second = rot[0..., 0..., 0..., half ..< rotaryDim]
    let rotated = concatenated([-second, first], axis: -1)
    let embedded = rot * cos.asType(x.dtype) + rotated * sin.asType(x.dtype)
    guard rotaryDim < headDim else { return embedded }
    let pass = x[0..., 0..., 0..., rotaryDim ..< headDim]
    return concatenated([embedded, pass], axis: -1)
}

final class SharpAIGlmasrEncoderAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIGlmasrAudioConfiguration) {
        let hidden = config.hiddenSize
        self.heads = config.attentionHeads
        self.headDim = hidden / config.attentionHeads
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(hidden, hidden, bias: true)
        // UNVERIFIED: k_proj bias is not in the sources. A mismatch fails the strict load
        // (Load.swift, verify .all).
        _kProj.wrappedValue = Linear(hidden, hidden, bias: false)
        _vProj.wrappedValue = Linear(hidden, hidden, bias: true)
        _oProj.wrappedValue = Linear(hidden, hidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)

        queries = sharpAIGlmasrApplyPartialRotary(queries, cos: cos, sin: sin)
        keys = sharpAIGlmasrApplyPartialRotary(keys, cos: cos, sin: sin)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: nil, scale: scale)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAIGlmasrEncoderMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: SharpAIGlmasrAudioConfiguration) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

final class SharpAIGlmasrEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SharpAIGlmasrEncoderAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIGlmasrEncoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    init(_ config: SharpAIGlmasrAudioConfiguration) {
        _selfAttention.wrappedValue = SharpAIGlmasrEncoderAttention(config)
        _mlp.wrappedValue = SharpAIGlmasrEncoderMLP(config)
        _inputLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var h = x + selfAttention(inputLayerNorm(x), cos: cos, sin: sin)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}

/// Whisper-style stem plus partial-RoPE encoder. Input `[batch, numMelBins, frames]`.
final class SharpAIGlmasrAudioEncoder: Module {
    let numMelBins: Int
    let rotaryDim: Int
    let ropeTheta: Float

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    let layers: [SharpAIGlmasrEncoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: SharpAIGlmasrAudioConfiguration) {
        let hidden = config.hiddenSize
        self.numMelBins = config.numMelBins
        let headDim = hidden / config.attentionHeads
        self.rotaryDim = Int(Float(headDim) * config.partialRotaryFactor)
        self.ropeTheta = config.ropeTheta

        _conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins, outputChannels: hidden, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(
            inputChannels: hidden, outputChannels: hidden, kernelSize: 3, stride: 2, padding: 1)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIGlmasrEncoderLayer(config)
        }
        _norm.wrappedValue = LayerNorm(dimensions: hidden, eps: 1e-5)
    }

    /// Returns `[batch, convLength, hidden]`, where `convLength = (frames - 1) / 2 + 1`.
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let dtype = conv1.weight.dtype
        var x = features.transposed(0, 2, 1).asType(dtype)
        x = gelu(conv1(x))
        x = gelu(conv2(x))

        let (cos, sin) = sharpAIGlmasrRotaryTables(
            positions: x.dim(1), rotaryDim: rotaryDim, theta: ropeTheta)
        for layer in layers {
            x = layer(x, cos: cos, sin: sin)
        }
        return norm(x)
    }
}

/// Two-layer MLP from `intermediate_size` (4 stacked encoder frames) to the text hidden size.
final class SharpAIGlmasrProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(audioIntermediate: Int, textHidden: Int) {
        _linear1.wrappedValue = Linear(audioIntermediate, textHidden * 2, bias: true)
        _linear2.wrappedValue = Linear(textHidden * 2, textHidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}
