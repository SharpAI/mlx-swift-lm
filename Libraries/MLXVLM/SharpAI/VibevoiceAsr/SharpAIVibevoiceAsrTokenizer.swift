// SharpAI fork overlay: vibevoice acoustic tokenizer encoder (used twice: acoustic and semantic).
// Not part of upstream mlx-swift-lm.
//
// Layout is NLC: (batch, samples, channels). The reference is NCL. Every convolution here is
// causal, so running the whole waveform in one pass gives the same frames as the reference's
// chunked, cached run. UNVERIFIED numerically: no reference output was available to compare.
// The decoder half of the tokenizer is not built, because ASR only runs the encoders.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Causal Conv1d. Pads on the left only. Weights use MLX layout (out, kernel, in / groups).
final class SharpAIVibevoiceAsrCausalConv1d: Module {

    @ModuleInfo(key: "conv") var conv: Conv1d
    let leftPad: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, groups: Int = 1) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize,
            stride: stride, padding: 0, groups: groups, bias: true)
        self.leftPad = max(0, kernelSize - stride)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if leftPad > 0 {
            let zeros = MLXArray.zeros([x.dim(0), leftPad, x.dim(2)], dtype: x.dtype)
            h = concatenated([zeros, x], axis: 1)
        }
        return conv(h)
    }
}

final class SharpAIVibevoiceAsrFeedForward: Module {

    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    let activation: (MLXArray) -> MLXArray

    init(hiddenSize: Int, expansion: Int, activation: @escaping (MLXArray) -> MLXArray) {
        self._linear1.wrappedValue = Linear(hiddenSize, expansion * hiddenSize, bias: true)
        self._linear2.wrappedValue = Linear(expansion * hiddenSize, hiddenSize, bias: true)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear1(x)))
    }
}

/// ConvNeXt-style block: a depthwise causal conv mixer and a feed-forward, each with layer scale.
final class SharpAIVibevoiceAsrConvNextBlock: Module {

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: SharpAIVibevoiceAsrCausalConv1d
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: SharpAIVibevoiceAsrFeedForward
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    @ParameterInfo(key: "ffn_gamma") var ffnGamma: MLXArray

    init(
        channels: Int, config: SharpAIVibevoiceAsrTokenizerConfiguration,
        activation: @escaping (MLXArray) -> MLXArray
    ) {
        self._norm.wrappedValue = RMSNorm(dimensions: channels, eps: config.rmsNormEps)
        self._mixer.wrappedValue = SharpAIVibevoiceAsrCausalConv1d(
            inChannels: channels, outChannels: channels, kernelSize: config.kernelSize,
            groups: channels)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: channels, eps: config.rmsNormEps)
        self._ffn.wrappedValue = SharpAIVibevoiceAsrFeedForward(
            hiddenSize: channels, expansion: config.ffnExpansion, activation: activation)
        self._gamma.wrappedValue =
            MLXArray.zeros([channels], dtype: .float32) + config.layerScaleInitValue
        self._ffnGamma.wrappedValue =
            MLXArray.zeros([channels], dtype: .float32) + config.layerScaleInitValue
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + mixer(norm(x)) * gamma
        h = h + ffn(ffnNorm(h)) * ffnGamma
        return h
    }
}

/// One stage of the encoder: a causal conv, then its ConvNeXt blocks.
/// The stem is a stage with stride 1. Each downsampling stage uses stride equal to its ratio.
final class SharpAIVibevoiceAsrEncoderStage: Module {

    @ModuleInfo(key: "conv") var conv: SharpAIVibevoiceAsrCausalConv1d
    @ModuleInfo(key: "stage") var stage: [SharpAIVibevoiceAsrConvNextBlock]

    init(
        inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, blocks: Int,
        config: SharpAIVibevoiceAsrTokenizerConfiguration,
        activation: @escaping (MLXArray) -> MLXArray
    ) {
        self._conv.wrappedValue = SharpAIVibevoiceAsrCausalConv1d(
            inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize,
            stride: stride)
        self._stage.wrappedValue = (0 ..< blocks).map { _ in
            SharpAIVibevoiceAsrConvNextBlock(
                channels: outChannels, config: config, activation: activation)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        for block in stage {
            h = block(h)
        }
        return h
    }
}

/// `VibeVoiceAcousticTokenizerEncoderModel`. Input (batch, samples, 1), output (batch, frames, hiddenSize).
final class SharpAIVibevoiceAsrTokenizerEncoder: Module {

    @ModuleInfo(key: "stem") var stem: SharpAIVibevoiceAsrEncoderStage
    @ModuleInfo(key: "conv_layers") var convLayers: [SharpAIVibevoiceAsrEncoderStage]
    @ModuleInfo(key: "head") var head: SharpAIVibevoiceAsrCausalConv1d

    let configuration: SharpAIVibevoiceAsrTokenizerConfiguration

    init(_ configuration: SharpAIVibevoiceAsrTokenizerConfiguration) throws {
        try configuration.validate()
        let activation = try SharpAIVibevoiceAsrActivation.resolve(configuration.hiddenAct)
        let filters = configuration.numFilters
        let ratios = configuration.downsamplingRatios
        let depths = configuration.depths

        self._stem.wrappedValue = SharpAIVibevoiceAsrEncoderStage(
            inChannels: configuration.channels, outChannels: filters,
            kernelSize: configuration.kernelSize, stride: 1, blocks: depths[0],
            config: configuration, activation: activation)

        var layers: [SharpAIVibevoiceAsrEncoderStage] = []
        for (stageIndex, ratio) in ratios.enumerated() {
            layers.append(
                SharpAIVibevoiceAsrEncoderStage(
                    inChannels: filters * (1 << stageIndex),
                    outChannels: filters * (1 << (stageIndex + 1)),
                    kernelSize: 2 * ratio, stride: ratio, blocks: depths[stageIndex + 1],
                    config: configuration, activation: activation))
        }
        self._convLayers.wrappedValue = layers

        self._head.wrappedValue = SharpAIVibevoiceAsrCausalConv1d(
            inChannels: filters * (1 << ratios.count), outChannels: configuration.hiddenSize,
            kernelSize: configuration.kernelSize)

        self.configuration = configuration
        super.init()
    }

    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        var h = stem(waveform)
        for layer in convLayers {
            h = layer(h)
        }
        return head(h)
    }
}
