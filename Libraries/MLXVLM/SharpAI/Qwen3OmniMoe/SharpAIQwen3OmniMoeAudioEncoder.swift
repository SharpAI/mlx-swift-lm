// SharpAI fork overlay: qwen3_omni_moe audio tower. Not part of upstream mlx-swift-lm.
// Whisper-style conv stem (three stride-2 Conv2d over mel x time), sinusoidal positions,
// pre-LN transformer blocks with attention limited to n_window_infer windows, then a
// two-layer projector to the text hidden size.
// UNVERIFIED against a checkpoint: follows transformers' Qwen3OmniMoeAudioEncoder as recalled.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Output tokens for `frames` mel frames: three stride-2 convolutions, each rounding up.
func sharpAIQwen3OmniMoeAudioTokenCount(_ frames: Int) -> Int {
    guard frames > 0 else { return 0 }
    var length = frames
    for _ in 0 ..< 3 {
        length = (length + 1) / 2
    }
    return length
}

/// Frequency bins left after the three stride-2 convolutions over `numMelBins`.
func sharpAIQwen3OmniMoeAudioFrequencyBins(_ numMelBins: Int) -> Int {
    var bins = numMelBins
    for _ in 0 ..< 3 {
        bins = (bins + 1) / 2
    }
    return bins
}

/// Sinusoidal table `[length, channels]`: sin block first, then cos block.
func sharpAIQwen3OmniMoeSinusoids(length: Int, channels: Int) -> MLXArray {
    let half = channels / 2
    let logIncrement = log(10_000.0) / Double(max(half - 1, 1))
    var values = [Float](repeating: 0, count: length * channels)
    for position in 0 ..< length {
        for index in 0 ..< half {
            let angle = Double(position) * exp(-logIncrement * Double(index))
            values[position * channels + index] = Float(sin(angle))
            values[position * channels + half + index] = Float(cos(angle))
        }
    }
    return MLXArray(values, [length, channels])
}

final class SharpAIQwen3OmniMoeAudioAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, heads: Int) {
        let headDim = hiddenSize / heads
        self.heads = heads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    /// `mask` is additive and broadcastable to `[batch, heads, len, len]`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let keys = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            attentionMask = .array(mask)
        } else {
            attentionMask = .none
        }
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: nil, scale: scale,
            mask: attentionMask)
        return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAIQwen3OmniMoeAudioEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SharpAIQwen3OmniMoeAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: SharpAIQwen3OmniMoeAudioConfiguration) {
        _selfAttention.wrappedValue = SharpAIQwen3OmniMoeAudioAttention(
            hiddenSize: config.dModel, heads: config.encoderAttentionHeads)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _fc1.wrappedValue = Linear(config.dModel, config.encoderFfnDim)
        _fc2.wrappedValue = Linear(config.encoderFfnDim, config.dModel)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + selfAttention(selfAttentionLayerNorm(x), mask: mask)
        h = h + fc2(gelu(fc1(finalLayerNorm(h))))
        return h
    }
}

/// Audio tower. Input is log-mel features `[chunks, numMelBins, frames]`, one chunk per row,
/// with `frames <= 2 * nWindow`. Output is the valid pooled tokens `[totalTokens, outputDim]`.
final class SharpAIQwen3OmniMoeAudioEncoder: Module {
    let config: SharpAIQwen3OmniMoeAudioConfiguration

    @ModuleInfo(key: "conv2d1") var conv2d1: Conv2d
    @ModuleInfo(key: "conv2d2") var conv2d2: Conv2d
    @ModuleInfo(key: "conv2d3") var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    let layers: [SharpAIQwen3OmniMoeAudioEncoderLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear

    init(_ config: SharpAIQwen3OmniMoeAudioConfiguration) {
        self.config = config
        let channels = config.downsampleHiddenSize
        _conv2d1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: channels, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1))
        _conv2d2.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1))
        _conv2d3.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: IntOrPair(3),
            stride: IntOrPair(2), padding: IntOrPair(1))
        _convOut.wrappedValue = Linear(
            channels * sharpAIQwen3OmniMoeAudioFrequencyBins(config.numMelBins),
            config.dModel, bias: false)
        self.layers = (0 ..< config.encoderLayers).map { _ in
            SharpAIQwen3OmniMoeAudioEncoderLayer(config)
        }
        _lnPost.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _proj1.wrappedValue = Linear(config.dModel, config.dModel)
        _proj2.wrappedValue = Linear(config.dModel, config.outputDim)
    }

    /// - Parameters:
    ///   - features: `[chunks, numMelBins, frames]` log-mel features.
    ///   - featureMask: `[chunks, frames]`, 1 for valid frames. `nil` means every frame is valid.
    /// - Returns: `[totalTokens, outputDim]`, valid tokens of every chunk in order.
    func callAsFunction(_ features: MLXArray, featureMask: MLXArray?) throws -> MLXArray {
        guard features.ndim == 3, features.dim(1) == config.numMelBins else {
            throw SharpAIQwen3OmniMoeError.invalidInput(
                "audio features must be [chunks, \(config.numMelBins), frames]")
        }
        let chunks = features.dim(0)
        let frames = features.dim(2)
        guard chunks > 0, frames > 0, frames <= 2 * config.nWindow else {
            throw SharpAIQwen3OmniMoeError.invalidInput(
                "audio chunk length \(frames) must be between 1 and \(2 * config.nWindow) frames")
        }

        let validFrames: [Int]
        if let featureMask {
            guard featureMask.ndim == 2, featureMask.dim(0) == chunks,
                featureMask.dim(1) == frames
            else {
                throw SharpAIQwen3OmniMoeError.invalidInput(
                    "feature mask must be [chunks, frames]")
            }
            validFrames = featureMask.asType(.int32).sum(axis: -1).asArray(Int.self)
        } else {
            validFrames = Array(repeating: frames, count: chunks)
        }

        // Conv stem on [chunks, mel, frames, 1]: frequency is H, time is W, as in PyTorch.
        let dtype = conv2d1.weight.dtype
        var x = features.asType(dtype).expandedDimensions(axis: -1)
        x = gelu(conv2d1(x))
        x = gelu(conv2d2(x))
        x = gelu(conv2d3(x))

        // [chunks, freq, time, channels] -> [chunks, time, channels, freq] -> [chunks, time, c*f].
        // This is the (channel, frequency) order that conv_out was trained with.
        let (freq, time, channels) = (x.dim(1), x.dim(2), x.dim(3))
        x = x.transposed(0, 2, 3, 1).reshaped(chunks, time, channels * freq)
        x = convOut(x)
        x = x + sharpAIQwen3OmniMoeSinusoids(length: time, channels: config.dModel).asType(
            x.dtype)

        var pieces: [MLXArray] = []
        var windowIds: [Int32] = []
        let chunksPerWindow = max(1, config.nWindowInfer / (2 * config.nWindow))
        for (chunk, length) in validFrames.enumerated() {
            let count = min(sharpAIQwen3OmniMoeAudioTokenCount(length), time)
            guard count > 0 else { continue }
            pieces.append(x[chunk, 0 ..< count])
            windowIds += Array(repeating: Int32(chunk / chunksPerWindow), count: count)
        }
        guard !pieces.isEmpty else {
            throw SharpAIQwen3OmniMoeError.invalidInput("audio has no valid frames")
        }

        var hidden = concatenated(pieces, axis: 0).expandedDimensions(axis: 0)
        let attentionMask: MLXArray? =
            Set(windowIds).count > 1
            ? windowMask(windowIds).asType(hidden.dtype) : nil
        for layer in layers {
            hidden = layer(hidden, mask: attentionMask)
        }
        hidden = lnPost(hidden)
        return proj2(gelu(proj1(hidden)))[0]
    }

    /// Additive mask `[1, 1, n, n]`: 0 inside a window, -inf across windows.
    private func windowMask(_ windowIds: [Int32]) -> MLXArray {
        let total = windowIds.count
        var values = [Float](repeating: 0, count: total * total)
        for query in 0 ..< total {
            for key in 0 ..< total where windowIds[query] != windowIds[key] {
                values[query * total + key] = -Float.infinity
            }
        }
        return MLXArray(values, [1, 1, total, total])
    }
}
