// SharpAI fork overlay: cohere_asr encoder (transformers ParakeetEncoder / FastConformer).
// Not part of upstream mlx-swift-lm.
//
// Input is log-mel features shaped (batch, time, mel bins). Output is (batch, time', hidden).
// Padding masks are not implemented: every input is treated as full length.
// UNVERIFIED: no checkpoint was run. Layout follows modeling_parakeet.py in transformers 5.x.

import Foundation
import MLX
import MLXNN

/// Two-dimensional conv subsampling: stride-2 convolutions over (time, mel), then a linear map.
/// Layer order matches HF `subsampling.layers`, which is conv, relu, then depthwise, pointwise,
/// relu pairs. The relu layers have no weights, so the Swift array keeps only the convs.
final class SharpAICohereAsrSubsampling: Module {

    @ModuleInfo(key: "layers") var layers: [Conv2d]
    @ModuleInfo(key: "linear") var linear: Linear

    init(_ config: SharpAICohereAsrEncoderConfiguration) {
        let channels = config.subsamplingConvChannels
        let kernel = config.subsamplingConvKernelSize
        let stride = config.subsamplingConvStride
        let padding = (kernel - 1) / 2
        let numLayers = Int(log2(Double(config.subsamplingFactor)))

        var convs: [Conv2d] = [
            Conv2d(
                inputChannels: 1, outputChannels: channels,
                kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                padding: IntOrPair(padding))
        ]
        for _ in 0 ..< max(numLayers - 1, 0) {
            convs.append(
                Conv2d(
                    inputChannels: channels, outputChannels: channels,
                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                    padding: IntOrPair(padding), groups: channels))
            convs.append(
                Conv2d(
                    inputChannels: channels, outputChannels: channels,
                    kernelSize: IntOrPair(1)))
        }

        let melAfter = config.numMelBins / Int(pow(Double(stride), Double(numLayers)))
        self._layers.wrappedValue = convs
        self._linear.wrappedValue = Linear(channels * melAfter, config.hiddenSize, bias: true)
        super.init()
    }

    /// (batch, time, mel) -> (batch, time', hidden)
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        // Channels-last: (batch, time, mel, 1). Relu follows layers 0, 2 and 4.
        var x = features.expandedDimensions(axis: -1)
        for (index, conv) in layers.enumerated() {
            x = conv(x)
            if index % 2 == 0 {
                x = relu(x)
            }
        }
        // HF flattens (channels, mel) with channels first, so move channels ahead of mel.
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let flat = x.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
        return linear(flat)
    }
}

/// Dense layer with an activation between two projections.
final class SharpAICohereAsrFeedForward: Module {

    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    let activation: (MLXArray) -> MLXArray

    init(_ config: SharpAICohereAsrEncoderConfiguration, activation: @escaping (MLXArray) -> MLXArray)
    {
        self._linear1.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.attentionBias)
        self._linear2.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.attentionBias)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear1(x)))
    }
}

/// Multi-head attention with relative position terms (Transformer-XL style, bias_u and bias_v).
final class SharpAICohereAsrRelativeAttention: Module {

    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ParameterInfo(key: "bias_u") var biasU: MLXArray
    @ParameterInfo(key: "bias_v") var biasV: MLXArray

    init(_ config: SharpAICohereAsrEncoderConfiguration) {
        let heads = config.attentionHeads
        let headDim = config.hiddenSize / heads
        self.heads = heads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        let width = heads * headDim
        self._qProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(width, config.hiddenSize, bias: config.attentionBias)
        self._relativeKProj.wrappedValue = Linear(config.hiddenSize, width, bias: false)
        self._biasU.wrappedValue = MLXArray.zeros([heads, headDim])
        self._biasV.wrappedValue = MLXArray.zeros([heads, headDim])
        super.init()
    }

    /// `x` is (batch, time, hidden). `positions` is (1, 2 * time - 1, hidden).
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))

        let q = qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)

        let qU = q + biasU.reshaped(1, heads, 1, headDim)
        let qV = q + biasV.reshaped(1, heads, 1, headDim)

        // (1, heads, head_dim, 2t - 1)
        let relK = relativeKProj(positions).reshaped(1, -1, heads, headDim)
            .transposed(0, 2, 3, 1)
        // Term (b) and (d). The shift aligns relative distances with each query row.
        let bd = relativeShift(matmul(qV, relK))[.ellipsis, ..<t] * scale

        // softmax((qU k^T) * scale + bd) v, the same as HF's eager path with mask = bd.
        let out = MLXFast.scaledDotProductAttention(
            queries: qU, keys: k, values: v, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.array(bd))
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }

    /// Shaw et al. relative shift for (batch, heads, t, 2t - 1) scores.
    private func relativeShift(_ scores: MLXArray) -> MLXArray {
        let (b, h, t, p) = (scores.dim(0), scores.dim(1), scores.dim(2), scores.dim(3))
        var x = MLX.padded(scores, widths: [0, 0, 0, .init((1, 0))])
        x = x.reshaped(b, h, p + 1, t)
        x = x[0..., 0..., 1..., 0...]
        return x.reshaped(b, h, t, p)
    }
}

/// Depthwise conv module with a GLU gate. Works on (batch, time, channels), so no transposes.
final class SharpAICohereAsrConvolutionModule: Module {

    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d
    let activation: (MLXArray) -> MLXArray

    init(
        _ config: SharpAICohereAsrEncoderConfiguration,
        activation: @escaping (MLXArray) -> MLXArray
    ) {
        let channels = config.hiddenSize
        let kernel = config.convKernelSize
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: 2 * channels, kernelSize: 1,
            stride: 1, padding: 0, bias: config.convolutionBias)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernel,
            stride: 1, padding: (kernel - 1) / 2, groups: channels,
            bias: config.convolutionBias)
        self._norm.wrappedValue = BatchNorm(featureCount: channels)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 1,
            stride: 1, padding: 0, bias: config.convolutionBias)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // GLU: first half times sigmoid of second half, over channels.
        let halves = pointwiseConv1(x).split(parts: 2, axis: -1)
        var h = halves[0] * MLX.sigmoid(halves[1])
        h = activation(norm(depthwiseConv(h)))
        return pointwiseConv2(h)
    }
}

/// One FastConformer block: half-step feed-forward, attention, conv, half-step feed-forward.
final class SharpAICohereAsrEncoderBlock: Module {

    @ModuleInfo(key: "feed_forward1") var feedForward1: SharpAICohereAsrFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAICohereAsrRelativeAttention
    @ModuleInfo(key: "conv") var conv: SharpAICohereAsrConvolutionModule
    @ModuleInfo(key: "feed_forward2") var feedForward2: SharpAICohereAsrFeedForward
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ config: SharpAICohereAsrEncoderConfiguration, activation: @escaping (MLXArray) -> MLXArray)
    {
        let width = config.hiddenSize
        self._feedForward1.wrappedValue = SharpAICohereAsrFeedForward(
            config, activation: activation)
        self._selfAttn.wrappedValue = SharpAICohereAsrRelativeAttention(config)
        self._conv.wrappedValue = SharpAICohereAsrConvolutionModule(
            config, activation: activation)
        self._feedForward2.wrappedValue = SharpAICohereAsrFeedForward(
            config, activation: activation)
        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        self._normConv.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        self._normOut.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        var h = x + feedForward1(normFeedForward1(x)) * 0.5
        h = h + selfAttn(normSelfAtt(h), positions: positions)
        h = h + conv(normConv(h))
        h = h + feedForward2(normFeedForward2(h)) * 0.5
        return normOut(h)
    }
}

/// Subsampling, relative positions, then the conformer blocks.
final class SharpAICohereAsrEncoder: Module {

    let inputScale: Float
    let hiddenSize: Int

    @ModuleInfo(key: "subsampling") var subsampling: SharpAICohereAsrSubsampling
    @ModuleInfo(key: "layers") var layers: [SharpAICohereAsrEncoderBlock]

    init(_ config: SharpAICohereAsrEncoderConfiguration) throws {
        let activation = try SharpAICohereAsrActivation.resolve(config.hiddenAct)
        self.hiddenSize = config.hiddenSize
        self.inputScale = config.scaleInput ? Float(config.hiddenSize).squareRoot() : 1
        self._subsampling.wrappedValue = SharpAICohereAsrSubsampling(config)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            SharpAICohereAsrEncoderBlock(config, activation: activation)
        }
        super.init()
    }

    /// (batch, time, mel) -> (batch, time', hidden)
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var hidden = subsampling(features) * inputScale
        let positions = Self.relativePositions(length: hidden.dim(1), dimensions: hiddenSize)
            .asType(hidden.dtype)
        for layer in layers {
            hidden = layer(hidden, positions: positions)
        }
        return hidden
    }

    /// Sinusoids for distances `length - 1` down to `-(length - 1)`, with sin and cos
    /// interleaved per frequency, as in HF `ParakeetEncoderRelPositionalEncoding`.
    static func relativePositions(length: Int, dimensions: Int) -> MLXArray {
        let distances = MLXArray((0 ..< (2 * length - 1)).map { Float(length - 1 - $0) })
        let halfDims = dimensions / 2
        let inverseFrequencies = MLXArray(
            (0 ..< halfDims).map { Float(1.0 / pow(10_000.0, Double(2 * $0) / Double(dimensions))) })
        let angles = distances.expandedDimensions(axis: -1) * inverseFrequencies
        let interleaved = MLX.stacked([sin(angles), cos(angles)], axis: -1)
            .reshaped(2 * length - 1, dimensions)
        return interleaved.expandedDimensions(axis: 0)
    }
}
