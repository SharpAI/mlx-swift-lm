// SharpAI fork overlay: moss_audio audio encoder. Not part of upstream mlx-swift-lm.
//
// Ports vLLM's moss_audio.py `MossAudioEncoder`: three Conv2d stages (3x3, stride 2, GELU) over
// `[mel, frames]`, a projection of the flattened stem to d_model, sinusoidal positions, pre-norm
// layers with bidirectional attention, a final LayerNorm, and an optional output projection.
// Audio is cut into chunks of 2 * n_window mel frames, and each chunk is encoded on its own. Attention
// therefore never crosses a chunk, which matches the reference's padding mask.
// UNVERIFIED (no checkpoint and no HF modeling file read): the stem flatten order (frequency vs
// channel), the sinusoid layout (sin|cos), the bias of k_proj, stem_proj and out_proj, the DeepStack
// tap convention (output of layer i, 0-based), and the DeepStack merger hidden size.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Frames after the three stride-2 stages, each `(n - 1) / 2 + 1`.
public func sharpAIMossAudioEncoderFrames(_ frames: Int) -> Int {
    var n = frames
    for _ in 0 ..< 3 {
        guard n > 0 else { return 0 }
        n = (n - 1) / 2 + 1
    }
    return n
}

/// Audio tokens for `frames` mel frames cut into chunks of `chunkFrames`.
public func sharpAIMossAudioTokenCount(frames: Int, chunkFrames: Int) -> Int {
    guard frames > 0, chunkFrames > 0 else { return 0 }
    var total = 0
    var start = 0
    while start < frames {
        total += sharpAIMossAudioEncoderFrames(min(chunkFrames, frames - start))
        start += chunkFrames
    }
    return total
}

/// Mel bins left after the three stride-2 stages.
func sharpAIMossAudioStemBins(_ mel: Int) -> Int {
    var n = mel
    for _ in 0 ..< 3 {
        n = (n - 1) / 2 + 1
    }
    return n
}

/// Sinusoidal table `[length, channels]`: sin in the first half, cos in the second.
func sharpAIMossAudioSinusoids(length: Int, channels: Int) -> MLXArray {
    let half = channels / 2
    var values = [Float](repeating: 0, count: length * channels)
    for position in 0 ..< length {
        for i in 0 ..< half {
            let inverse = pow(10000.0, -Double(i) / Double(half - 1))
            let angle = Double(position) * inverse
            values[position * channels + i] = Float(sin(angle))
            values[position * channels + half + i] = Float(cos(angle))
        }
    }
    return MLXArray(values, [length, channels])
}

final class SharpAIMossAudioEncoderAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SharpAIMossAudioEncoderConfiguration) {
        let hidden = config.dModel
        self.heads = config.attentionHeads
        self.headDim = hidden / config.attentionHeads
        self.scale = pow(Float(hidden / config.attentionHeads), -0.5)
        _qProj.wrappedValue = Linear(hidden, hidden, bias: true)
        // UNVERIFIED: k_proj bias is not in the sources.
        _kProj.wrappedValue = Linear(hidden, hidden, bias: false)
        _vProj.wrappedValue = Linear(hidden, hidden, bias: true)
        _outProj.wrappedValue = Linear(hidden, hidden, bias: true)
    }

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

final class SharpAIMossAudioEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SharpAIMossAudioEncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: SharpAIMossAudioEncoderConfiguration) {
        _selfAttention.wrappedValue = SharpAIMossAudioEncoderAttention(config)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.dModel, eps: config.layerNormEps)
        _fc1.wrappedValue = Linear(config.dModel, config.ffnDim, bias: true)
        _fc2.wrappedValue = Linear(config.ffnDim, config.dModel, bias: true)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
    }

    /// Pre-norm residual block: attention, then the GELU feed-forward.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + selfAttention(selfAttentionLayerNorm(x))
        h = h + fc2(gelu(fc1(finalLayerNorm(h))))
        return h
    }
}

final class SharpAIMossAudioEncoder: Module {
    let dModel: Int
    let stemBins: Int
    let chunkFrames: Int
    let channels: Int

    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv3") var conv3: Conv2d
    @ModuleInfo(key: "stem_proj") var stemProj: Linear
    @ModuleInfo(key: "layers") var layers: [SharpAIMossAudioEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear?

    init(_ config: SharpAIMossAudioEncoderConfiguration) {
        self.dModel = config.dModel
        self.stemBins = sharpAIMossAudioStemBins(config.numMelBins)
        self.chunkFrames = 2 * config.nWindow
        self.channels = config.downsampleHiddenSize

        _conv1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: config.downsampleHiddenSize, kernelSize: 3,
            stride: 2, padding: 1)
        _conv2.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize, outputChannels: config.downsampleHiddenSize,
            kernelSize: 3, stride: 2, padding: 1)
        _conv3.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize, outputChannels: config.downsampleHiddenSize,
            kernelSize: 3, stride: 2, padding: 1)
        _stemProj.wrappedValue = Linear(
            config.downsampleHiddenSize * sharpAIMossAudioStemBins(config.numMelBins),
            config.dModel, bias: true)
        _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in
            SharpAIMossAudioEncoderLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        if config.outputDim != config.dModel {
            _outProj.wrappedValue = Linear(config.dModel, config.outputDim, bias: true)
        }
    }

    /// `features` is `[1, numMelBins, frames]`. Returns the encoder output `[1, tokens, outputDim]`
    /// and one DeepStack tap `[1, tokens, dModel]` for each entry of `deepstackLayers`.
    func callAsFunction(_ features: MLXArray, deepstackLayers: [Int]) -> (
        output: MLXArray, taps: [MLXArray]
    ) {
        let frames = features.dim(2)
        var outputs: [MLXArray] = []
        var tapParts: [[MLXArray]] = Array(repeating: [], count: deepstackLayers.count)
        var start = 0
        while start < frames {
            let end = min(start + chunkFrames, frames)
            let chunk = features[0..., 0..., start ..< end]
            let encoded = encodeChunk(chunk, deepstackLayers: deepstackLayers)
            outputs.append(encoded.output)
            for index in encoded.taps.indices {
                tapParts[index].append(encoded.taps[index])
            }
            start = end
        }
        let taps = tapParts.map { concatenated($0, axis: 1) }
        return (concatenated(outputs, axis: 1), taps)
    }

    /// Encodes one chunk `[1, numMelBins, len]`.
    private func encodeChunk(_ chunk: MLXArray, deepstackLayers: [Int]) -> (
        output: MLXArray, taps: [MLXArray]
    ) {
        // NHWC: height is the mel (frequency) axis, width is time.
        var x = chunk.expandedDimensions(axis: -1).asType(conv1.weight.dtype)
        x = gelu(conv1(x))
        x = gelu(conv2(x))
        x = gelu(conv3(x))  // [1, stemBins, tokens, channels]
        let tokens = x.dim(2)
        // UNVERIFIED: flatten order is (frequency, channel).
        x = x.transposed(0, 2, 1, 3).reshaped(1, tokens, stemBins * channels)

        var h = stemProj(x)
        h = h + sharpAIMossAudioSinusoids(length: tokens, channels: dModel).expandedDimensions(
            axis: 0)

        var tapByLayer: [Int: MLXArray] = [:]
        for (index, layer) in layers.enumerated() {
            h = layer(h)
            if deepstackLayers.contains(index) {
                tapByLayer[index] = h
            }
        }
        let normed = layerNorm(h)
        let output: MLXArray
        if let outProj {
            output = outProj(normed)
        } else {
            output = normed
        }
        let taps = deepstackLayers.compactMap { tapByLayer[$0] }
        return (output, taps)
    }
}
