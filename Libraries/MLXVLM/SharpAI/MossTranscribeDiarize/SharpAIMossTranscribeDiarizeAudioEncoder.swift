// SharpAI fork overlay: moss_transcribe_diarize audio encoder. Not part of upstream mlx-swift-lm.
// UNVERIFIED: the MOSS encoder is assumed to be Whisper-style: conv stem, a learned position
// table, pre-LN blocks with a bias-free key projection, and a final LayerNorm. The spec tags
// the encoder as "whisper" only. No reference implementation is on disk.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Audio tokens for `frames` mel frames: conv stride 2, then groups of `downsampleFactor`.
func sharpAIMossTranscribeDiarizeTokenCount(frames: Int, downsampleFactor: Int) -> Int {
    guard frames > 0, downsampleFactor > 0 else { return 0 }
    return ((frames - 1) / 2 + 1) / downsampleFactor
}

final class SharpAIMossTranscribeDiarizeEncoderAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SharpAIMossTranscribeDiarizeAudioConfiguration) {
        let hidden = config.hiddenSize
        self.heads = config.attentionHeads
        self.headDim = hidden / config.attentionHeads
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(hidden, hidden, bias: true)
        _kProj.wrappedValue = Linear(hidden, hidden, bias: false)
        _vProj.wrappedValue = Linear(hidden, hidden, bias: true)
        _outProj.wrappedValue = Linear(hidden, hidden, bias: true)
    }

    /// Bidirectional attention over every encoder position.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let keys = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: nil, scale: scale)
        return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAIMossTranscribeDiarizeEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SharpAIMossTranscribeDiarizeEncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: SharpAIMossTranscribeDiarizeAudioConfiguration) {
        _selfAttention.wrappedValue = SharpAIMossTranscribeDiarizeEncoderAttention(config)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: 1e-5)
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + selfAttention(selfAttentionLayerNorm(x))
        h = h + fc2(gelu(fc1(finalLayerNorm(h))))
        return h
    }
}

/// Whisper-style stem and blocks. Input `[batch, numMelBins, frames]`.
final class SharpAIMossTranscribeDiarizeAudioEncoder: Module {
    let numMelBins: Int

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    let layers: [SharpAIMossTranscribeDiarizeEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: SharpAIMossTranscribeDiarizeAudioConfiguration) {
        let hidden = config.hiddenSize
        self.numMelBins = config.numMelBins
        _conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins, outputChannels: hidden, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(
            inputChannels: hidden, outputChannels: hidden, kernelSize: 3, stride: 2, padding: 1)
        _embedPositions.wrappedValue = Embedding(
            embeddingCount: config.maxSourcePositions, dimensions: hidden)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIMossTranscribeDiarizeEncoderLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: 1e-5)
    }

    /// Returns `[batch, convLength, hidden]`, where `convLength = (frames - 1) / 2 + 1`.
    /// The caller checks that `convLength` does not exceed `maxSourcePositions`.
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let dtype = conv1.weight.dtype
        var x = features.transposed(0, 2, 1).asType(dtype)
        x = gelu(conv1(x))
        x = gelu(conv2(x))

        x = x + embedPositions.weight[0 ..< x.dim(1)]
        for layer in layers {
            x = layer(x)
        }
        return layerNorm(x)
    }
}

/// Stacks `downsampleFactor` encoder frames per token, then maps them to the text hidden size.
final class SharpAIMossTranscribeDiarizeProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDim: Int, textHidden: Int) {
        _linear1.wrappedValue = Linear(inputDim, textHidden, bias: true)
        _linear2.wrappedValue = Linear(textHidden, textHidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}
