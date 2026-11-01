// SharpAI fork overlay: StepVL Perception Encoder tower. Not part of upstream mlx-swift-lm.
//
// Port of `PerceptionEncoder` in vLLM's `step_vl.py`, which copies perception_models `pe.py`.
// Input is NHWC. Output is `[batch, tokens, width * 4]` after two stride-2 3x3 convs.
// Module keys match the checkpoint after `SharpAIStepRoboticsSanitize`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum SharpAIStepRoboticsActivation {
    /// Activation for a `hidden_act` name, or nil when the name is not supported.
    public static func function(for name: String) -> ((MLXArray) -> MLXArray)? {
        switch name {
        case "gelu": return { gelu($0) }
        case "gelu_pytorch_tanh", "gelu_new": return { geluApproximate($0) }
        case "gelu_fast": return { geluFastApproximate($0) }
        case "quick_gelu": return { $0 * sigmoid(MLXArray(Float(1.702)) * $0) }
        case "silu": return { silu($0) }
        default: return nil
        }
    }
}

/// 2D rotary embedding of the Perception Encoder. Frequencies come from the patch grid.
enum SharpAIStepRoboticsRope2D {
    /// vLLM leaves the Perception Encoder theta at its default.
    static let theta: Double = 10_000

    /// `cos` and `sin` of shape `[1, 1, prefixTokens + gridH * gridW, headDim]`.
    /// The channel layout is `[w freqs, h freqs]`, and each frequency repeats for an adjacent pair.
    /// `gridOffset` is 1 when a CLS token is used, as in vLLM's table.
    static func cosSin(
        headDim: Int, gridH: Int, gridW: Int, prefixTokens: Int, gridOffset: Int
    ) -> (MLXArray, MLXArray) {
        let quarter = headDim / 4
        let invFreq: [Float] = (0 ..< quarter).map { k in
            Float(1 / pow(theta, Double(2 * k) / Double(headDim / 2)))
        }

        var values = [Float]()
        values.reserveCapacity((prefixTokens + gridH * gridW) * headDim)
        values += [Float](repeating: 0, count: prefixTokens * headDim)
        for h in 0 ..< gridH {
            for w in 0 ..< gridW {
                let rowPos = Float(h + gridOffset)
                let colPos = Float(w + gridOffset)
                for k in 0 ..< quarter {
                    let f = colPos * invFreq[k]
                    values.append(f)
                    values.append(f)
                }
                for k in 0 ..< quarter {
                    let f = rowPos * invFreq[k]
                    values.append(f)
                    values.append(f)
                }
            }
        }
        let angles = MLXArray(values, [1, 1, prefixTokens + gridH * gridW, headDim])
        return (MLX.cos(angles), MLX.sin(angles))
    }

    /// Rotates adjacent channel pairs: `(x0, x1) -> (-x1, x0)`, the `rotate_half` of vLLM.
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let shape = x.shape
        let channels = shape[shape.count - 1]
        let pairs = x.reshaped(Array(shape.dropLast()) + [channels / 2, 2])
        let even = pairs[.ellipsis, 0]
        let odd = pairs[.ellipsis, 1]
        let rotated = stacked([-odd, even], axis: -1).reshaped(shape)
        return x * cos.asType(x.dtype) + rotated * sin.asType(x.dtype)
    }
}

/// Multi-head self-attention with 2D RoPE. Module path `attn.qkv_proj`, `attn.out_proj`.
final class SharpAIStepRoboticsVisionAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let useClsToken: Bool

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SharpAIStepRoboticsConfiguration.VisionConfiguration) {
        self.heads = config.heads
        self.headDim = config.width / config.heads
        self.scale = pow(Float(config.width / config.heads), -0.5)
        self.useClsToken = config.useClsToken
        self._qkvProj.wrappedValue = Linear(config.width, 3 * config.width, bias: true)
        self._outProj.wrappedValue = Linear(config.width, config.width, bias: true)
    }

    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let qkv = qkvProj(x)
        let width = heads * headDim

        var q = qkv[.ellipsis, 0 ..< width].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var k = qkv[.ellipsis, width ..< 2 * width].reshaped(B, L, heads, headDim)
            .transposed(0, 2, 1, 3)
        let v = qkv[.ellipsis, (2 * width)...].reshaped(B, L, heads, headDim)
            .transposed(0, 2, 1, 3)

        let (cos, sin) = SharpAIStepRoboticsRope2D.cosSin(
            headDim: headDim, gridH: gridH, gridW: gridW,
            prefixTokens: useClsToken ? 1 : 0, gridOffset: useClsToken ? 1 : 0)
        q = SharpAIStepRoboticsRope2D.apply(q, cos: cos, sin: sin)
        k = SharpAIStepRoboticsRope2D.apply(k, cos: cos, sin: sin)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Per-channel scale on a residual branch. Module path `ls_1.gamma` or `ls_2.gamma`.
final class SharpAIStepRoboticsLayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dimensions: Int, initValue: Float) {
        self._gamma.wrappedValue = MLXArray.ones([dimensions]) * initValue
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

/// Two-layer MLP with GELU-family activation. Module path `mlp.fc1`, `mlp.fc2`.
final class SharpAIStepRoboticsVisionMLP: Module {
    let activation: (MLXArray) -> MLXArray

    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(width: Int, hiddenWidth: Int, activation: @escaping (MLXArray) -> MLXArray) {
        self.activation = activation
        self._fc1.wrappedValue = Linear(width, hiddenWidth, bias: true)
        self._fc2.wrappedValue = Linear(hiddenWidth, width, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(activation(fc1(x)))
    }
}

/// Pre-norm residual block. Module path `resblocks.N.*`.
final class SharpAIStepRoboticsVisionBlock: Module {
    @ModuleInfo var attn: SharpAIStepRoboticsVisionAttention
    @ModuleInfo(key: "ls_1") var ls1: SharpAIStepRoboticsLayerScale?
    @ModuleInfo(key: "ls_2") var ls2: SharpAIStepRoboticsLayerScale?
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo var mlp: SharpAIStepRoboticsVisionMLP

    init(
        _ config: SharpAIStepRoboticsConfiguration.VisionConfiguration,
        activation: @escaping (MLXArray) -> MLXArray
    ) {
        let width = config.width
        self._attn.wrappedValue = SharpAIStepRoboticsVisionAttention(config)
        if let initValue = config.layerScaleInit {
            self._ls1.wrappedValue = SharpAIStepRoboticsLayerScale(
                dimensions: width, initValue: initValue)
            self._ls2.wrappedValue = SharpAIStepRoboticsLayerScale(
                dimensions: width, initValue: initValue)
        }
        self._ln1.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        self._ln2.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        self._mlp.wrappedValue = SharpAIStepRoboticsVisionMLP(
            width: width, hiddenWidth: Int(Float(width) * config.mlpRatio), activation: activation)
    }

    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        var h = x
        var attnOut = attn(ln1(h), gridH: gridH, gridW: gridW)
        if let ls1 { attnOut = ls1(attnOut) }
        h = h + attnOut

        var mlpOut = mlp(ln2(h))
        if let ls2 { mlpOut = ls2(mlpOut) }
        return h + mlpOut
    }
}

/// Stack of residual blocks. Module path `transformer.resblocks.N`.
final class SharpAIStepRoboticsVisionTransformer: Module {
    @ModuleInfo var resblocks: [SharpAIStepRoboticsVisionBlock]

    init(_ config: SharpAIStepRoboticsConfiguration.VisionConfiguration,
        activation: @escaping (MLXArray) -> MLXArray
    ) {
        self._resblocks.wrappedValue = (0 ..< config.layers).map { _ in
            SharpAIStepRoboticsVisionBlock(config, activation: activation)
        }
    }

    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        var h = x
        for block in resblocks {
            h = block(h, gridH: gridH, gridW: gridW)
        }
        return h
    }
}

/// The Perception Encoder. Output is projected by the wrapper's `vit_large_projector`.
public final class SharpAIStepRoboticsVisionEncoder: Module {
    let imageSize: Int
    let patchSize: Int
    let width: Int
    let useClsToken: Bool

    @ModuleInfo var conv1: Conv2d
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm?
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm?
    @ModuleInfo var transformer: SharpAIStepRoboticsVisionTransformer
    @ModuleInfo(key: "vit_downsampler1") var downsampler1: Conv2d
    @ModuleInfo(key: "vit_downsampler2") var downsampler2: Conv2d
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray?
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray?

    init(_ config: SharpAIStepRoboticsConfiguration.VisionConfiguration) throws {
        guard let activation = SharpAIStepRoboticsActivation.function(for: config.hiddenAct)
        else {
            throw SharpAIStepRoboticsError.invalidConfiguration(
                "unsupported vision hidden_act '\(config.hiddenAct)'")
        }
        self.imageSize = config.imageSize
        self.patchSize = config.patchSize
        self.width = config.width
        self.useClsToken = config.useClsToken

        let grid = config.imageSize / config.patchSize
        self._conv1.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: config.width,
            kernelSize: IntOrPair(config.patchSize), stride: IntOrPair(config.patchSize),
            bias: false)
        self._lnPre.wrappedValue =
            config.useLnPre ? LayerNorm(dimensions: config.width, eps: 1e-5) : nil
        self._lnPost.wrappedValue =
            config.useLnPost ? LayerNorm(dimensions: config.width, eps: 1e-5) : nil
        self._transformer.wrappedValue = SharpAIStepRoboticsVisionTransformer(
            config, activation: activation)
        // UNVERIFIED: the downsampler convs take a bias, the Conv2d default. vLLM's Conv2dLayer default is assumed.
        self._downsampler1.wrappedValue = Conv2d(
            inputChannels: config.width, outputChannels: config.width * 2,
            kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1))
        self._downsampler2.wrappedValue = Conv2d(
            inputChannels: config.width * 2, outputChannels: config.width * 4,
            kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1))

        self._classEmbedding.wrappedValue =
            config.useClsToken ? MLXArray.zeros([config.width]) : nil
        self._positionalEmbedding.wrappedValue =
            config.useAbsPosEmb
            ? MLXArray.zeros([(config.useClsToken ? 1 : 0) + grid * grid, config.width]) : nil
    }

    /// NHWC pixels `[batch, imageSize, imageSize, 3]` to `[batch, tokens, width * 4]`.
    /// UNVERIFIED: the Perception Encoder's own pixel normalization is not in the sources here.
    public func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
        guard pixels.dim(1) == imageSize, pixels.dim(2) == imageSize else {
            throw SharpAIStepRoboticsError.imageSizeMismatch(
                expected: imageSize, height: pixels.dim(1), width: pixels.dim(2))
        }
        let grid = imageSize / patchSize
        let batch = pixels.dim(0)

        var x = conv1(pixels).reshaped(batch, grid * grid, width)
        if let classEmbedding {
            let cls = broadcast(
                classEmbedding.reshaped(1, 1, width), to: [batch, 1, width])
            x = concatenated([cls, x], axis: 1)
        }
        if let positionalEmbedding {
            x = x + positionalEmbedding.reshaped(1, -1, width)
        }
        if let lnPre { x = lnPre(x) }
        x = transformer(x, gridH: grid, gridW: grid)
        if let lnPost { x = lnPost(x) }
        if useClsToken {
            x = x[0..., 1..., 0...]
        }

        var spatial = x.reshaped(batch, grid, grid, width)
        spatial = downsampler1(spatial)
        spatial = downsampler2(spatial)
        return spatial.reshaped(batch, -1, width * 4)
    }
}
