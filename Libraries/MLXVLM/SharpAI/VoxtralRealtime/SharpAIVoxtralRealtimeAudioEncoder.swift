// SharpAI fork overlay: voxtral_realtime audio tower and projector. Not part of upstream
// mlx-swift-lm.
//
// Input is log-mel features shaped (batch, mel, time), as the HF feature extractor returns them.
// Both convs pad on the left only, so the encoder stays causal. Output is (batch, time', hidden)
// with time' = ceil(time / 2). The projector merges `downsample_factor` frames per token.
// UNVERIFIED: no checkpoint was run. Activation placement, bias use and key names follow the
// config and the HF module layout, not a reference run.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Pads `amount` frames on the left of a (batch, time, channels) array.
func sharpAIVoxtralRealtimeLeftPad(_ x: MLXArray, _ amount: Int) -> MLXArray {
    MLX.padded(x, widths: [0, .init((amount, 0)), 0])
}

/// Causal self-attention with RoPE. Keys and values use the same head count as queries.
final class SharpAIVoxtralRealtimeAudioAttention: Module {

    let heads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIVoxtralRealtimeAudioConfiguration) {
        let heads = config.attentionHeads
        let headDim = config.headDim
        self.heads = heads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        self.rope = initializeRope(
            dims: headDim, base: config.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: nil)

        let width = heads * headDim
        self._qProj.wrappedValue = Linear(config.hiddenSize, width, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, width, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, width, bias: false)
        self._oProj.wrappedValue = Linear(width, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)

        q = rope(q, offset: 0)
        k = rope(k, offset: 0)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

/// Gated MLP: `down(silu(gate(x)) * up(x))`. Shared by the audio tower and the text decoder.
final class SharpAIVoxtralRealtimeMLP: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hidden: Int, intermediate: Int) {
        self._gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class SharpAIVoxtralRealtimeAudioLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIVoxtralRealtimeAudioAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIVoxtralRealtimeMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIVoxtralRealtimeAudioConfiguration) {
        self._selfAttn.wrappedValue = SharpAIVoxtralRealtimeAudioAttention(config)
        self._mlp.wrappedValue = SharpAIVoxtralRealtimeMLP(
            hidden: config.hiddenSize, intermediate: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}

/// Causal conv stem (two convs, the second with stride 2), then sliding-window layers.
final class SharpAIVoxtralRealtimeAudioEncoder: Module {

    let slidingWindow: Int

    // UNVERIFIED: conv stem layout, kernel sizes, key names and the RoPE offset are not in any
    // source. The fixture config has none of them.
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "layers") var layers: [SharpAIVoxtralRealtimeAudioLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIVoxtralRealtimeAudioConfiguration) {
        self.slidingWindow = config.slidingWindow
        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins, outputChannels: config.hiddenSize,
            kernelSize: 3)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize, outputChannels: config.hiddenSize,
            kernelSize: 3, stride: 2)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            SharpAIVoxtralRealtimeAudioLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// `features` is (batch, mel, time). Returns (batch, time', hidden).
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = features.transposed(0, 2, 1)
        x = gelu(conv1(sharpAIVoxtralRealtimeLeftPad(x, 2)))
        x = gelu(conv2(sharpAIVoxtralRealtimeLeftPad(x, 2)))

        let mask: MLXFast.ScaledDotProductAttentionMaskMode = .array(
            createCausalMask(n: x.dim(1), offset: 0, windowSize: slidingWindow))
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return norm(x)
    }
}

/// Merges `downsampleFactor` consecutive frames, then maps them to the text width with a
/// two-layer GELU MLP.
final class SharpAIVoxtralRealtimeProjector: Module {

    let downsampleFactor: Int

    // UNVERIFIED: the two-layer GELU projector, its bias-free layers and key names are assumed.
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: SharpAIVoxtralRealtimeConfiguration) {
        self.downsampleFactor = config.downsampleFactor
        let inputWidth = config.audioConfiguration.hiddenSize * config.downsampleFactor
        let textWidth = config.textConfiguration.hiddenSize
        self._linear1.wrappedValue = Linear(inputWidth, textWidth, bias: false)
        self._linear2.wrappedValue = Linear(textWidth, textWidth, bias: false)
        super.init()
    }

    /// `frames` is (batch, time', audioHidden). Trailing frames that do not fill a group are dropped.
    func callAsFunction(_ frames: MLXArray) -> MLXArray {
        let (b, t, c) = (frames.dim(0), frames.dim(1), frames.dim(2))
        let usable = t - t % downsampleFactor
        let grouped = frames[0..., 0 ..< usable].reshaped(
            b, usable / downsampleFactor, c * downsampleFactor)
        return linear2(gelu(linear1(grouped)))
    }
}
