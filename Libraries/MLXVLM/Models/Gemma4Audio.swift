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
    
    // New fields
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let attentionInvalidLogitsValue: Float
    public let residualWeight: Float
    public let gradientClipping: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case attentionChunkSize = "attention_chunk_size"
        case convKernelSize = "conv_kernel_size"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case useClippedLinears = "use_clipped_linears"
        case rmsNormEps = "rms_norm_eps"
        case outputProjDims = "output_proj_dims"
        
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case residualWeight = "residual_weight"
        case gradientClipping = "gradient_clipping"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_audio"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.attentionChunkSize = try container.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 12
        self.convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 5
        self.subsamplingConvChannels = try container.decodeIfPresent([Int].self, forKey: .subsamplingConvChannels) ?? [128, 32]
        self.useClippedLinears = try container.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? true
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.outputProjDims = try container.decodeIfPresent(Int.self, forKey: .outputProjDims) ?? 1536
        
        self.attentionContextLeft = try container.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? 13
        self.attentionContextRight = try container.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? 0
        self.attentionLogitCap = try container.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? 50.0
        self.attentionInvalidLogitsValue = try container.decodeIfPresent(Float.self, forKey: .attentionInvalidLogitsValue) ?? -1e9
        self.residualWeight = try container.decodeIfPresent(Float.self, forKey: .residualWeight) ?? 0.5
        self.gradientClipping = try container.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? 1e10
    }
}

// MARK: - Core Components

/// Standard Swish activation function: \nx * sigmoid(x)
private class Swish: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * sigmoid(x)
    }
}

class AudioRMSNorm: Module {
    var weight: MLXArray
    let eps: Float
    
    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat32 = x.asType(.float32)
        let norm = rsqrt(xFloat32.square().mean(axes: [-1], keepDims: true) + eps)
        return ((xFloat32 * norm) * weight.asType(.float32)).asType(x.dtype)
    }
}

class AudioLayerNorm: Module {
    var weight: MLXArray
    var bias: MLXArray?
    let eps: Float
    
    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.bias = nil  // Not all variants include bias; loaded if present in checkpoint
        self.eps = eps
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat32 = x.asType(.float32)
        let mean = xFloat32.mean(axes: [-1], keepDims: true)
        let variance = xFloat32.variance(axes: [-1], keepDims: true)
        let norm = (xFloat32 - mean) * rsqrt(variance + eps)
        let scaled = norm * weight.asType(.float32)
        if let b = bias {
            return (scaled + b.asType(.float32)).asType(x.dtype)
        }
        return scaled.asType(x.dtype)
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
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?
    
    init(_ inputChannels: Int, _ outputChannels: Int, bias: Bool = false) {
        self._linear.wrappedValue = Linear(inputChannels, outputChannels, bias: bias)
        self._inputMin.wrappedValue = MLXArray(-Float.infinity)
        self._inputMax.wrappedValue = MLXArray(Float.infinity)
        self._outputMin.wrappedValue = MLXArray(-Float.infinity)
        self._outputMax.wrappedValue = MLXArray(Float.infinity)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Apply activation clipping: these bounds are calibrated quantization ranges
        // stored in the 8-bit PTQ checkpoint. Without clipping, activations blow out of
        // the calibrated distribution across all 12 conformer blocks → wrong speech features.
        var inp = x
        if let lo = inputMin, let hi = inputMax {
            inp = MLX.clip(inp, min: lo.asType(inp.dtype), max: hi.asType(inp.dtype))
        }
        var out = linear(inp)
        if let lo = outputMin, let hi = outputMax {
            out = MLX.clip(out, min: lo.asType(out.dtype), max: hi.asType(out.dtype))
        }
        return out
    }
}

// MARK: - Subsample Projection

private class SubsampleConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: AudioLayerNorm

    init(inChannels: Int, outChannels: Int, eps: Float) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: [3, 3],
            stride: [2, 2],
            padding: [0, 0],
            bias: false
        )
        self._norm.wrappedValue = AudioLayerNorm(dimensions: outChannels, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var out = x

        // Match HF semantics: mask == true marks valid audio frames.
        let expandedMask = mask.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
        out = MLX.where(expandedMask, out, MLXArray.zeros(like: out))

        out = MLX.padded(out, widths: [[0, 0], [1, 1], [1, 1], [0, 0]])

        out = conv(out)

        // Downsample mask by time stride (2)
        let tOut = out.dim(1)
        var outMask = mask[0..., .stride(by: 2)]
        outMask = outMask[0..., ..<tOut]

        out = norm(out)
        out = MLX.maximum(out, MLXArray(0, dtype: out.dtype)) // ReLU — matches Python nn.relu

        return (out, outMask)
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

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var hidden = x.reshaped(x.dim(0), x.dim(1), x.dim(2), 1) // Shape: [B, L, 128, 1]
        var currentMask = mask
        
        let out0 = layer0(hidden, mask: currentMask)
        hidden = out0.0
        currentMask = out0.1
        
        let out1 = layer1(hidden, mask: currentMask)
        hidden = out1.0
        currentMask = out1.1
        
        // Output from layer 1: [B, L/4, 128/4, outChannels=32]
        let (B, L_new, F_new, C_new) = (hidden.dim(0), hidden.dim(1), hidden.dim(2), hidden.dim(3))
        
        hidden = hidden.reshaped(B, L_new, F_new * C_new) // Flatten features
        
        return (inputProjLinear(hidden), currentMask)
    }
}

// MARK: - Conformer Components

/// Macaron FFN
private class MacaronFFN: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: ClippedLinear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: ClippedLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: AudioRMSNorm
    
    let gradientClipping: Float
    let residualWeight: Float

    init(hiddenSize: Int, eps: Float, gradientClipping: Float, residualWeight: Float) {
        let expansion = hiddenSize * 4
        self.gradientClipping = gradientClipping
        self.residualWeight = residualWeight
        self._preLayerNorm.wrappedValue = AudioRMSNorm(dimensions: hiddenSize, eps: eps)
        self._ffwLayer1.wrappedValue = ClippedLinear(hiddenSize, expansion, bias: false)
        self._ffwLayer2.wrappedValue = ClippedLinear(expansion, hiddenSize, bias: false)
        self._postLayerNorm.wrappedValue = AudioRMSNorm(dimensions: hiddenSize, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var hidden = MLX.clip(x, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        hidden = preLayerNorm(hidden)
        hidden = ffwLayer1(hidden)
        hidden = MLXNN.silu(hidden)
        hidden = ffwLayer2(hidden)
        hidden = MLX.clip(hidden, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        hidden = postLayerNorm(hidden)
        return residual + hidden * residualWeight
    }
}

/// Conformer Light Convolution Module
private class ConformerLightConv1d: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: ClippedLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: AudioRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: ClippedLinear
    
    let causalPadding: Int
    let gradientClipping: Float

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        self.causalPadding = config.convKernelSize - 1

        self._preLayerNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._linearStart.wrappedValue = ClippedLinear(config.hiddenSize, config.hiddenSize * 2, bias: false)

        self._depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            stride: 1,
            padding: 0,
            groups: config.hiddenSize,
            bias: false
        )

        self._convNorm.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._linearEnd.wrappedValue = ClippedLinear(config.hiddenSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        
        var hidden = preLayerNorm(x)
        hidden = linearStart(hidden)
        
        let parts = split(hidden, parts: 2, axis: -1)
        hidden = parts[0] * sigmoid(parts[1])
        
        // Causal padding: [B, L, C]
        hidden = MLX.padded(hidden, widths: [[0, 0], [causalPadding, 0], [0, 0]])
        
        hidden = depthwiseConv1d(hidden)
        
        hidden = MLX.clip(hidden, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        hidden = convNorm(hidden)
        hidden = MLXNN.silu(hidden)
        hidden = linearEnd(hidden)
        
        return hidden + residual
    }
}

/// Sinusoidal relative position embedding for chunked attention
private struct AudioRelativePositionEmbedding {
    let numHeads: Int
    let channels: Int
    let headDim: Int
    let maxBackward: Int
    let maxForward: Int

    var invTimescales: MLXArray

    init(config: Gemma4AudioConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.channels = config.hiddenSize
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.maxBackward = max(0, config.attentionContextLeft - 1)
        self.maxForward = config.attentionContextRight

        let minTimescale: Float = 1.0
        let maxTimescale: Float = 10000.0
        let numTimescales = config.hiddenSize / 2
        let logTimescaleIncrement = log(maxTimescale / minTimescale) / Float(max(numTimescales - 1, 1))
        
        let invT = minTimescale * MLX.exp(MLXArray((0..<numTimescales).map { Float($0) }) * -logTimescaleIncrement)
        self.invTimescales = invT.reshaped(1, 1, numTimescales)
    }

    func getTimingSignal(position: MLXArray, dtype: DType) -> MLXArray {
        let posFloat = position.asType(.float32).expandedDimensions(axis: -1)
        let scaledTime = posFloat * invTimescales
        let signal = concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: -1)
        return signal.asType(dtype)
    }

    func relativeShift(termBd: MLXArray, batchSize: Int, numHeads: Int, numBlocks: Int, blockSize: Int, contextSize: Int, maxSpanPlus1: Int) -> MLXArray {
        let padAmount = (contextSize + 1) - maxSpanPlus1
        let termPadded = MLX.padded(termBd, widths: [[0, 0], [0, 0], [0, 0], [0, 0], [0, padAmount]])
        var shifted = termPadded.reshaped(batchSize, numHeads, numBlocks, blockSize * (contextSize + 1))
        shifted = shifted[0..., 0..., 0..., ..<(blockSize * contextSize)]
        return shifted.reshaped(batchSize, numHeads, numBlocks, blockSize, contextSize)
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, posProj: Linear) -> MLXArray {
        let B = queries.dim(0)
        let U = queries.dim(1)
        let W = queries.dim(2)
        let N = queries.dim(3)
        let H = queries.dim(4)
        let C = keys.dim(2)

        let posIndicesRange = stride(from: maxBackward, through: -maxForward, by: -1).map { Int32($0) }
        let posIndices = MLXArray(posIndicesRange).expandedDimensions(axis: 0)
        let maxSpanPlus1 = posIndices.dim(1)

        var sinEmb = getTimingSignal(position: posIndices, dtype: queries.dtype)
        sinEmb = posProj(sinEmb.asType(posProj.weight.dtype))
        sinEmb = sinEmb.reshaped(maxSpanPlus1, numHeads, headDim)
        sinEmb = sinEmb.asType(queries.dtype)

        let queriesP = queries.transposed(0, 3, 1, 2, 4)
        let keysP = keys.transposed(0, 3, 1, 4, 2)
        let termAc = matmul(queriesP, keysP)

        let sinEmbT = sinEmb.transposed(1, 2, 0)
        let qReshaped = queriesP.reshaped(B, N, U * W, H)
        let termBd = matmul(qReshaped, sinEmbT).reshaped(B, N, U, W, maxSpanPlus1)

        let termBdShifted = relativeShift(termBd: termBd, batchSize: B, numHeads: N, numBlocks: U, blockSize: W, contextSize: C, maxSpanPlus1: maxSpanPlus1)

        return termAc + termBdShifted
    }
}

/// Chunked local attention with relative position embeddings and logit softcapping.
private class AudioAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: ClippedLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippedLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippedLinear
    @ModuleInfo(key: "post") var post: ClippedLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray
    
    let numHeads: Int
    let hiddenSize: Int
    let headDim: Int
    let chunkSize: Int
    let maxFutureHorizon: Int
    let maxPastHorizon: Int
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    
    let qScale: Float
    let kScale: Float
    let relPos: AudioRelativePositionEmbedding
    
    init(config: Gemma4AudioConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.hiddenSize = config.hiddenSize
        self.headDim = config.hiddenSize / config.numAttentionHeads
        
        self.chunkSize = config.attentionChunkSize
        self.maxFutureHorizon = config.attentionContextRight
        self.maxPastHorizon = max(0, config.attentionContextLeft - 1)
        self.contextSize = self.chunkSize + self.maxPastHorizon + self.maxFutureHorizon
        self.invalidLogitsValue = config.attentionInvalidLogitsValue
        self.softcap = config.attentionLogitCap
        
        self._qProj.wrappedValue = ClippedLinear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = ClippedLinear(hiddenSize, numHeads * headDim, bias: false)
        self._vProj.wrappedValue = ClippedLinear(hiddenSize, numHeads * headDim, bias: false)
        self._post.wrappedValue = ClippedLinear(hiddenSize, hiddenSize, bias: false)
        self._relativeKProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._perDimScale.wrappedValue = MLXArray.zeros([headDim])
        
        self.qScale = Float(pow(Double(headDim), -0.5)) / log(2.0)
        self.kScale = log(1.0 + exp(1.0)) / log(2.0)
        
        self.relPos = AudioRelativePositionEmbedding(config: config)
    }

    func padDim1(_ x: MLXArray, padLeft: Int, padRight: Int) -> MLXArray {
        var pads = Array(repeating: IntOrPair([0, 0]), count: x.ndim)
        pads[1] = IntOrPair([padLeft, padRight])
        return MLX.padded(x, widths: pads)
    }

    func convertToBlock(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let rest = Array(x.shape[2...])
        let numBlocks = (T + chunkSize - 1) / chunkSize
        let padLen = numBlocks * chunkSize - T
        var expanded = x
        if padLen > 0 {
            expanded = padDim1(x, padLeft: 0, padRight: padLen)
        }
        var newShape = [B, numBlocks, chunkSize]
        newShape.append(contentsOf: rest)
        return expanded.reshaped(newShape)
    }

    func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let padded = padDim1(x, padLeft: padLeft, padRight: padRight)
        let tPadded = padded.dim(1)
        let numBlocks = (tPadded - contextSize) / chunkSize + 1
        
        let starts = MLXArray(stride(from: 0, to: numBlocks * chunkSize, by: chunkSize).map { Int32($0) })
        let offsets = MLXArray((0..<contextSize).map { Int32($0) })
        let indices = starts.expandedDimensions(axis: 1) + offsets.expandedDimensions(axis: 0)
        
        return padded[0..., indices]
    }

    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        let B = hiddenStates.dim(0)
        let T = hiddenStates.dim(1)
        let qkvShape = [B, T, numHeads, headDim]

        var q = qProj(hiddenStates).asType(.float32).reshaped(qkvShape)
        var k = kProj(hiddenStates).asType(.float32).reshaped(qkvShape)
        let v = vProj(hiddenStates).asType(.float32).reshaped(qkvShape)

        let safePerDimScale = MLXNN.softplus(perDimScale)
        q = q * (MLXArray(qScale) * safePerDimScale)
        k = k * MLXArray(kScale)

        let queryBlocks = convertToBlock(q)
        let keyBlocks = extractBlockContext(k)
        let valueBlocks = extractBlockContext(v)
        let U = queryBlocks.dim(1)

        let extractedValid = extractBlockContext(mask)
        
        let cond1 = extractedValid.reshaped(extractedValid.dim(0), 1, extractedValid.dim(1), 1, extractedValid.dim(2))
        let cond2 = causalValidMask.reshaped(1, 1, 1, causalValidMask.dim(0), causalValidMask.dim(1))
        let condition = logicalAnd(cond1, cond2)

        var logits = relPos(queries: queryBlocks, keys: keyBlocks, posProj: relativeKProj)
        
        logits = MLX.tanh(logits / MLXArray(softcap)) * MLXArray(softcap)
        logits = MLX.where(condition, logits, MLXArray(invalidLogitsValue))

        let probs = MLX.softmax(logits, axis: -1)
        
        // einsum("bnuwc,bucnh->buwnh", probs, value_blocks)
        // Manual implementation:
        // probs: [B, N, U, W, C]  -> [B, U, N, W, C]
        // valueBlocks: [B, U, C, N, H] -> [B, U, N, C, H]
        let probsT = probs.transposed(0, 2, 1, 3, 4)
        let vT = valueBlocks.transposed(0, 1, 3, 2, 4)
        var context = matmul(probsT, vT)
        context = context.transposed(0, 1, 3, 2, 4)
        
        context = context.reshaped(B, U * chunkSize, numHeads, headDim)
        context = context[0..., ..<T]

        let BOut = context.dim(0)
        let TOut = context.dim(1)
        context = context.reshaped(BOut, TOut, numHeads * headDim)
        return post(context)
    }
}

/// A complete Conformer block layer
private class ConformerBlock: Module {
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: AudioRMSNorm

    @ModuleInfo(key: "feed_forward1") var ffn1: MacaronFFN
    @ModuleInfo(key: "self_attn") var selfAttention: AudioAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: ConformerLightConv1d
    @ModuleInfo(key: "feed_forward2") var ffn2: MacaronFFN

    let gradientClipping: Float

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        let eps = config.rmsNormEps
        self._normPreAttn.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize, eps: eps)
        self._normPostAttn.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize, eps: eps)
        self._normOut.wrappedValue = AudioRMSNorm(dimensions: config.hiddenSize, eps: eps)

        self._ffn1.wrappedValue = MacaronFFN(hiddenSize: config.hiddenSize, eps: eps, gradientClipping: config.gradientClipping, residualWeight: config.residualWeight)
        self._selfAttention.wrappedValue = AudioAttention(config: config)
        self._lconv1d.wrappedValue = ConformerLightConv1d(config: config)
        self._ffn2.wrappedValue = MacaronFFN(hiddenSize: config.hiddenSize, eps: eps, gradientClipping: config.gradientClipping, residualWeight: config.residualWeight)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        var hidden = ffn1(x)

        // Residual taken AFTER ffn1 (matches Python: res = x where x = ffn1(x))
        let residual = hidden
        let attnIn = normPreAttn(hidden)
        hidden = selfAttention(attnIn, mask: mask, causalValidMask: causalValidMask)
        hidden = MLX.clip(hidden, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        hidden = residual + normPostAttn(hidden)

        // Zero invalid frames before lconv1d — required by both HF (modeling_gemma3n.py:897-900)
        // and mlx-vlm Python (audio.py:451-452). Prevents padding from bleeding into valid
        // frame outputs via the convolution receptive field.
        // mask = True=VALID (current convention), so multiply directly to zero invalid frames.
        let validityMask = mask.expandedDimensions(axis: -1).asType(hidden.dtype)
        hidden = hidden * validityMask
        hidden = lconv1d(hidden)
        hidden = ffn2(hidden)
        hidden = MLX.clip(hidden, min: MLXArray(-gradientClipping), max: MLXArray(gradientClipping))
        
        return normOut(hidden)
    }
}

// MARK: - Audio Model Wrapper

public class Gemma4AudioModel: Module {
    @ModuleInfo(key: "subsample_conv_projection") fileprivate var subsampleConvProjection: SubsampleConvProjection
    @ModuleInfo(key: "layers") fileprivate var layers: [ConformerBlock]
    @ModuleInfo(key: "output_proj") var outputProj: Linear?

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
        
        if config.outputProjDims != config.hiddenSize {
            self._outputProj.wrappedValue = Linear(config.hiddenSize, config.outputProjDims, bias: true)
        }
    }

    private func buildCausalValidMask() -> MLXArray {
        let chunkSize = config.attentionChunkSize
        let maxFutureHorizon = config.attentionContextRight
        let maxPastHorizon = max(0, config.attentionContextLeft - 1)
        let contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        let upperDiagonal = maxPastHorizon + maxFutureHorizon

        let onesLower = MLXArray.ones([contextSize, chunkSize])
        let lowerCausal = MLX.tril(onesLower).transposed()

        let onesUpper = MLXArray.ones([chunkSize, contextSize])
        let upperCausal = MLX.tril(onesUpper, k: upperDiagonal)

        let mask = (lowerCausal * upperCausal).asType(.bool)
        return mask
    }

    func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var (audioEncodings, currentMask) = subsampleConvProjection(audioMel, mask: mask)
        let causalValidMask = buildCausalValidMask()

        for block in layers {
            audioEncodings = block(audioEncodings, mask: currentMask, causalValidMask: causalValidMask)
        }

        if let outputProj = outputProj {
            audioEncodings = outputProj(audioEncodings)
        }

        // Strip out padding mismatches
        if currentMask.dim(1) != audioEncodings.dim(1) {
            let targetLen = audioEncodings.dim(1)
            currentMask = currentMask[0..., ..<targetLen]
        }

        let validOut = MLX.where(
            currentMask.expandedDimensions(axis: -1),
            audioEncodings,
            MLXArray.zeros(like: audioEncodings)
        )
        return (validOut, currentMask)
    }

    open func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights
    }
}
