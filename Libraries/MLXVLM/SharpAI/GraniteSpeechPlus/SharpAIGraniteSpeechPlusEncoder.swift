// SharpAI fork overlay: granite_speech_plus audio encoder (a block-attention conformer).
// Not part of upstream mlx-swift-lm.
//
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/modeling_granite_speech.py
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech_plus/modular_granite_speech_plus.py
//
// Input is (batch, time, input_dim) mel features. Output is (batch, time, hidden * (1 + count(cat))).
// Inference only: dropout is the identity. Padding masks are not implemented: every input is full
// length, as in the HF path with no `input_features_mask`.
// Layout: Conv1d and BatchNorm work on (batch, time, channels), so no transposes are needed.
// The checkpoint's PyTorch conv weights are transposed to MLX layout in the sanitizer.

import Foundation
import MLX
import MLXNN

/// Half-step feed-forward: pre-norm, up projection, SiLU, down projection.
final class SharpAIGraniteSpeechPlusFeedForward: Module {

    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hidden: Int, multiplier: Int) {
        self._preNorm.wrappedValue = LayerNorm(dimensions: hidden)
        self._upProj.wrappedValue = Linear(hidden, hidden * multiplier)
        self._downProj.wrappedValue = Linear(hidden * multiplier, hidden)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(upProj(preNorm(x))))
    }
}

/// Block attention with Shaw relative-position scores, computed in blocks of `contextSize`.
/// The last block is right-padded and its padded keys are masked, as in the HF code.
final class SharpAIGraniteSpeechPlusBlockAttention: Module {

    let heads: Int
    let headDim: Int
    let contextSize: Int
    let scale: Float

    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "rel_pos_emb") var relPosEmb: Embedding

    init(_ config: SharpAIGraniteSpeechPlusEncoderConfiguration) {
        let heads = config.numHeads
        let headDim = config.resolvedDimHead
        self.heads = heads
        self.headDim = headDim
        self.contextSize = config.contextSize
        self.scale = pow(Float(headDim), -0.5)

        let inner = heads * headDim
        self._preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenDim)
        self._toQ.wrappedValue = Linear(config.hiddenDim, inner, bias: false)
        self._toKV.wrappedValue = Linear(config.hiddenDim, inner * 2, bias: false)
        self._toOut.wrappedValue = Linear(inner, config.hiddenDim, bias: true)
        self._relPosEmb.wrappedValue = Embedding(
            embeddingCount: 2 * config.maxPosEmb + 1, dimensions: headDim)
        super.init()
    }

    /// `x` is (batch, time, hidden). `attentionDists` is (context, context) indices into `relPosEmb`.
    func callAsFunction(_ x: MLXArray, attentionDists: MLXArray) -> MLXArray {
        let h = preNorm(x)
        let bsz = h.dim(0)
        let numFeatures = h.dim(1)
        let numBlocks = (numFeatures + contextSize - 1) / contextSize
        let remainder = numFeatures % contextSize

        var padded = h
        if remainder > 0 {
            padded = MLX.padded(h, widths: [0, .init((0, contextSize - remainder)), 0])
        }

        let q = toQ(padded)
        let kv = toKV(padded).split(parts: 2, axis: -1)
        let flat = bsz * numBlocks

        // (batch * blocks, heads, context, head_dim). Same memory order as HF's reshape.
        func blocks(_ t: MLXArray) -> MLXArray {
            t.reshaped(flat, contextSize, heads, headDim).transposed(0, 2, 1, 3)
        }
        let qh = blocks(q)
        let kh = blocks(kv[0])
        let vh = blocks(kv[1])

        // Shaw term: einsum("b h c d, c r d -> b h c r"), then scaled.
        let rel = relPosEmb(attentionDists)  // (context, context, head_dim)
        let qc = qh.transposed(0, 2, 1, 3)  // (flat, context, heads, head_dim)
        let relT = rel.transposed(0, 2, 1)  // (context, head_dim, context)
        var posAttn = matmul(qc, relT).transposed(0, 2, 1, 3) * scale

        if remainder > 0 {
            // Only the last block is partial. Inside it, keys past `remainder` are masked.
            // 1 where the entry is a masked padded key of the last block; 0 inside the real keys.
            var maskValues = [Int32](repeating: 1, count: contextSize * contextSize)
            for r in 0 ..< remainder {
                for c in 0 ..< remainder {
                    maskValues[r * contextSize + c] = 0
                }
            }
            let padMask = MLXArray(maskValues, [1, 1, contextSize, contextSize])
            let lastBlock = MLXArray(
                (0 ..< flat).map { Int32($0 % numBlocks == numBlocks - 1 ? 1 : 0) },
                [flat, 1, 1, 1])
            let masked = (lastBlock * padMask) .>= MLXArray(Int32(1))
            posAttn = MLX.where(masked, MLXArray(-Float.greatestFiniteMagnitude), posAttn)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.array(posAttn))
        let merged = out.transposed(0, 2, 1, 3).reshaped(bsz, numBlocks * contextSize, -1)
        return toOut(merged[0..., ..<numFeatures, 0...])
    }
}

/// Depthwise 1D convolution with HF's (possibly asymmetric) "same" padding.
final class SharpAIGraniteSpeechPlusDepthwiseConv: Module {

    let padLeft: Int
    let padRight: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernelSize: Int) {
        let pad = kernelSize / 2
        let padOffset = (kernelSize + 1) % 2
        self.padLeft = pad
        self.padRight = pad - padOffset
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            stride: 1, padding: 0, groups: channels, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(MLX.padded(x, widths: [0, .init((padLeft, padRight)), 0]))
    }
}

/// Conformer conv module: pointwise conv, GLU, depthwise conv, batch norm, SiLU, pointwise conv.
final class SharpAIGraniteSpeechPlusConvModule: Module {

    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "up_conv") var upConv: Conv1d
    @ModuleInfo(key: "depth_conv") var depthConv: SharpAIGraniteSpeechPlusDepthwiseConv
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "down_conv") var downConv: Conv1d

    init(_ config: SharpAIGraniteSpeechPlusEncoderConfiguration) {
        let hidden = config.hiddenDim
        let inner = hidden * config.convExpansionFactor
        self._norm.wrappedValue = LayerNorm(dimensions: hidden)
        self._upConv.wrappedValue = Conv1d(
            inputChannels: hidden, outputChannels: inner * 2, kernelSize: 1)
        self._depthConv.wrappedValue = SharpAIGraniteSpeechPlusDepthwiseConv(
            channels: inner, kernelSize: config.convKernelSize)
        self._batchNorm.wrappedValue = BatchNorm(featureCount: inner)
        self._downConv.wrappedValue = Conv1d(inputChannels: inner, outputChannels: hidden, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = upConv(norm(x))
        // GLU over channels: the first half gates the second half.
        let halves = h.split(parts: 2, axis: -1)
        h = halves[0] * MLX.sigmoid(halves[1])
        h = silu(batchNorm(depthConv(h)))
        return downConv(h)
    }
}

/// One conformer block: half-step feed-forward, attention, conv, half-step feed-forward, norm.
final class SharpAIGraniteSpeechPlusConformerBlock: Module {

    @ModuleInfo(key: "ff1") var ff1: SharpAIGraniteSpeechPlusFeedForward
    @ModuleInfo(key: "attn") var attn: SharpAIGraniteSpeechPlusBlockAttention
    @ModuleInfo(key: "conv") var conv: SharpAIGraniteSpeechPlusConvModule
    @ModuleInfo(key: "ff2") var ff2: SharpAIGraniteSpeechPlusFeedForward
    @ModuleInfo(key: "post_norm") var postNorm: LayerNorm

    init(_ config: SharpAIGraniteSpeechPlusEncoderConfiguration) {
        self._ff1.wrappedValue = SharpAIGraniteSpeechPlusFeedForward(
            hidden: config.hiddenDim, multiplier: config.feedforwardMult)
        self._attn.wrappedValue = SharpAIGraniteSpeechPlusBlockAttention(config)
        self._conv.wrappedValue = SharpAIGraniteSpeechPlusConvModule(config)
        self._ff2.wrappedValue = SharpAIGraniteSpeechPlusFeedForward(
            hidden: config.hiddenDim, multiplier: config.feedforwardMult)
        self._postNorm.wrappedValue = LayerNorm(dimensions: config.hiddenDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, attentionDists: MLXArray) -> MLXArray {
        var h = ff1(x) * 0.5 + x
        h = attn(h, attentionDists: attentionDists) + h
        h = conv(h) + h
        h = ff2(h) * 0.5 + h
        return postNorm(h)
    }
}

/// The granite_speech_plus CTC encoder (`GraniteSpeechPlusCTCEncoder`).
/// The mid-layer self-conditioning and the optional concatenation of earlier layers follow HF.
public final class SharpAIGraniteSpeechPlusEncoder: Module {

    let numLayers: Int
    let contextSize: Int
    let maxPosEmb: Int
    /// 1-based block indices whose outputs are concatenated. 0 is the input projection.
    let catLayers: Set<Int>

    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "layers") var layers: [SharpAIGraniteSpeechPlusConformerBlock]
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "out_mid") var outMid: Linear

    public init(_ config: SharpAIGraniteSpeechPlusEncoderConfiguration) {
        self.numLayers = config.numLayers
        self.contextSize = config.contextSize
        self.maxPosEmb = config.maxPosEmb
        self.catLayers = Set(config.catHiddenLayers ?? [])
        self._inputLinear.wrappedValue = Linear(config.inputDim, config.hiddenDim, bias: true)
        self._layers.wrappedValue = (0 ..< config.numLayers).map { _ in
            SharpAIGraniteSpeechPlusConformerBlock(config)
        }
        self._out.wrappedValue = Linear(config.hiddenDim, config.outputDim, bias: true)
        self._outMid.wrappedValue = Linear(config.outputDim, config.hiddenDim, bias: true)
        super.init()
    }

    /// (batch, time, input_dim) -> (batch, time, hidden * (1 + count(cat)))
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var h = inputLinear(features)
        let dists = Self.relativeDistances(contextSize: contextSize, maxPosEmb: maxPosEmb)

        var exported: [MLXArray] = []
        if catLayers.contains(0) {
            exported.append(h)
        }
        for (index, layer) in layers.enumerated() {
            let idx = index + 1
            h = layer(h, attentionDists: dists)
            // HF adds the mid-layer term in place, so an exported mid layer already includes it.
            if idx == numLayers / 2 {
                h = h + outMid(MLX.softmax(out(h), axis: -1))
            }
            if catLayers.contains(idx) {
                exported.append(h)
            }
        }
        if !exported.isEmpty {
            h = concatenated(exported + [h], axis: -1)
        }
        return h
    }

    /// Clamped relative distances `clamp(i - j, -context, context) + max_pos_emb`, as HF builds them.
    static func relativeDistances(contextSize: Int, maxPosEmb: Int) -> MLXArray {
        var values = [Int32]()
        values.reserveCapacity(contextSize * contextSize)
        for i in 0 ..< contextSize {
            for j in 0 ..< contextSize {
                let distance = max(-contextSize, min(contextSize, i - j))
                values.append(Int32(distance + maxPosEmb))
            }
        }
        return MLXArray(values, [contextSize, contextSize])
    }
}
