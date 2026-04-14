import Foundation
import MLX
import MLXNN

// MARK: - Configurations

public struct Gemma4AudioConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let attentionChunkSize: Int
    public let convKernelSize: Int
    public let subsamplingConvChannels: [Int]
    public let useClippedLinears: Bool
    public let rmsNormEps: Float
    public let outputProjDims: Int

    public init(
        modelType: String = "gemma4_audio",
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 8,
        attentionChunkSize: Int = 12,
        convKernelSize: Int = 5,
        subsamplingConvChannels: [Int] = [128, 32],
        useClippedLinears: Bool = true,
        rmsNormEps: Float = 1e-6,
        outputProjDims: Int = 1536
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.attentionChunkSize = attentionChunkSize
        self.convKernelSize = convKernelSize
        self.subsamplingConvChannels = subsamplingConvChannels
        self.useClippedLinears = useClippedLinears
        self.rmsNormEps = rmsNormEps
        self.outputProjDims = outputProjDims
    }
}

// MARK: - Core Components

/// Standard Swish activation function: \nx * sigmoid(x)
private class Swish: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * sigmoid(x)
    }
}

/// Gated Linear Unit (GLU)
private class GLU: Module {
    let dim: Int
    init(dim: Int = -1) {
        self.dim = dim
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(x, parts: 2, axis: dim)
        return parts[0] * sigmoid(parts[1])
    }
}

/// A wrapper for Linears that supports HF quantized mapping
private class ClippedLinear: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    
    init(_ inputChannels: Int, _ outputChannels: Int, bias: Bool = false) {
        self._linear.wrappedValue = Linear(inputChannels, outputChannels, bias: bias)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear(x)
    }
}

// MARK: - Subsample Projection

private class SubsampleConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo var activation: Swish

    init(inChannels: Int, outChannels: Int, eps: Float) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: [3, 3],
            stride: [2, 2],
            padding: [1, 1], // 'SAME' equivalent for stride=2 kernel=3
            bias: false
        )
        self._norm.wrappedValue = RMSNorm(dimensions: outChannels, eps: eps)
        self.activation = Swish()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        out = activation(out)
        out = norm(out)
        return out
    }
}

private class SubsampleConvProjection: Module {
    @ModuleInfo(key: "layer0") var layer0: SubsampleConvLayer
    @ModuleInfo(key: "layer1") var layer1: SubsampleConvLayer
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear
    
    init(channels: [Int] = [128, 32], hiddenSize: Int, eps: Float) {
        self._layer0.wrappedValue = SubsampleConvLayer(inChannels: 1, outChannels: channels[0], eps: eps)
        self._layer1.wrappedValue = SubsampleConvLayer(inChannels: channels[0], outChannels: channels[1], eps: eps)
        
        // Gemma 4 uses 128 Mel Bins, double stranded: 128 / 4 = 32. 32 * channels[1] = 1024
        let flattendDimensions = channels[1] * (128 / 4)
        self._inputProjLinear.wrappedValue = Linear(flattendDimensions, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x.reshaped(x.dim(0), x.dim(1), x.dim(2), 1) // Shape: [B, L, 128, 1]
        
        hidden = layer0(hidden)
        hidden = layer1(hidden)
        
        // Output from layer 1: [B, L/4, 128/4, outChannels=32]
        let (B, L_new, F_new, C_new) = (hidden.dim(0), hidden.dim(1), hidden.dim(2), hidden.dim(3))
        
        hidden = hidden.reshaped([B, L_new, F_new * C_new]) // Flatten features
        
        return inputProjLinear(hidden)
    }
}

// MARK: - Conformer Components

/// Macaron FFN
private class MacaronFFN: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: ClippedLinear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: ClippedLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: RMSNorm
    @ModuleInfo var activation: Swish

    init(hiddenSize: Int, eps: Float) {
        let expansion = hiddenSize * 4
        self._preLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: eps)
        self._ffwLayer1.wrappedValue = ClippedLinear(hiddenSize, expansion, bias: false)
        self._ffwLayer2.wrappedValue = ClippedLinear(expansion, hiddenSize, bias: false)
        self._postLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: eps)
        self.activation = Swish()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = preLayerNorm(x)
        hidden = ffwLayer1(hidden)
        hidden = activation(hidden)
        hidden = ffwLayer2(hidden)
        return postLayerNorm(hidden) * 0.5
    }
}

/// Conformer Convolution Module
private class ConformerConvModule: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: ClippedLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: RMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: ClippedLinear
    
    @ModuleInfo var glu: GLU
    @ModuleInfo var activation: Swish

    init(hiddenSize: Int, kernelSize: Int, eps: Float) {
        self._preLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: eps)
        self._linearStart.wrappedValue = ClippedLinear(hiddenSize, hiddenSize * 2, bias: false)
        self.glu = GLU(dim: -1)
        
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: kernelSize,
            stride: 1,
            padding: (kernelSize - 1) / 2,
            groups: hiddenSize,
            bias: false
        )
        
        self._convNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: eps)
        self.activation = Swish()
        self._linearEnd.wrappedValue = ClippedLinear(hiddenSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = preLayerNorm(x)
        
        hidden = linearStart(hidden)
        hidden = glu(hidden)
        
        hidden = depthwiseConv(hidden)
        hidden = convNorm(hidden)
        hidden = activation(hidden)
        
        hidden = linearEnd(hidden)
        return hidden
    }
}

/// Conformer Self-Attention block
private class ConformerAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: ClippedLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippedLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippedLinear
    @ModuleInfo(key: "post") var post: ClippedLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear?
    
    let numHeads: Int
    let scale: Float
    
    init(hiddenSize: Int, numHeads: Int, eps: Float) {
        self.numHeads = numHeads
        self.scale = Float(hiddenSize / numHeads).squareRoot()
        
        self._qProj.wrappedValue = ClippedLinear(hiddenSize, hiddenSize, bias: false)
        self._kProj.wrappedValue = ClippedLinear(hiddenSize, hiddenSize, bias: false)
        self._vProj.wrappedValue = ClippedLinear(hiddenSize, hiddenSize, bias: false)
        self._post.wrappedValue = ClippedLinear(hiddenSize, hiddenSize, bias: false)
        self._relativeKProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)
        
        let (B, L, _) = (q.dim(0), q.dim(1), q.dim(2))
        
        q = q.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        k = k.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        v = v.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        
        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0 / scale,
            mask: attentionMask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        
        return post(attn)
    }
}

/// A complete Conformer block layer
private class ConformerBlock: Module {
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: RMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: RMSNorm
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm

    @ModuleInfo(key: "feed_forward1") var ffn1: MacaronFFN
    @ModuleInfo(key: "self_attn") var selfAttention: ConformerAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: ConformerConvModule
    @ModuleInfo(key: "feed_forward2") var ffn2: MacaronFFN

    init(config: Gemma4AudioConfiguration) {
        let eps = config.rmsNormEps
        self._normPreAttn.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        self._normPostAttn.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)
        self._normOut.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: eps)

        self._ffn1.wrappedValue = MacaronFFN(hiddenSize: config.hiddenSize, eps: eps)
        self._selfAttention.wrappedValue = ConformerAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            eps: eps
        )
        self._lconv1d.wrappedValue = ConformerConvModule(
            hiddenSize: config.hiddenSize,
            kernelSize: config.convKernelSize,
            eps: eps
        )
        self._ffn2.wrappedValue = MacaronFFN(hiddenSize: config.hiddenSize, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        
        // FFN 1
        hidden = hidden + ffn1(hidden)
        
        // Self Attention with pre-norm
        let attnIn = normPreAttn(hidden)
        hidden = hidden + selfAttention(attnIn)
        
        // Conv Module with post-attn norm
        let convIn = normPostAttn(hidden)
        hidden = hidden + lconv1d(convIn)
        
        // FFN 2
        hidden = hidden + ffn2(hidden)
        
        return normOut(hidden)
    }
}

// MARK: - Audio Model Wrapper

public class Gemma4AudioModel: Module {
    @ModuleInfo(key: "subsample_conv_projection") fileprivate var subsampleConvProjection: SubsampleConvProjection
    @ModuleInfo(key: "layers") fileprivate var layers: [ConformerBlock]
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    public let config: Gemma4AudioConfiguration

    public init(config: Gemma4AudioConfiguration) {
        self.config = config
        self._subsampleConvProjection.wrappedValue = SubsampleConvProjection(
            channels: config.subsamplingConvChannels,
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps
        )
        
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            ConformerBlock(config: config)
        }
        
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.outputProjDims, bias: true)
    }

    public func callAsFunction(_ melSpectrogram: MLXArray) -> MLXArray {
        // MelSpectrogram input shape: [batch, seq_len, 80]
        var hidden = subsampleConvProjection(melSpectrogram)
        
        for layer in layers {
            hidden = layer(hidden)
        }
        
        return outputProj(hidden)
    }
    
    open func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights
    }
}
