// SharpAI fork overlay: qianfan_ocr vision tower (InternViT layout). Not part of upstream mlx-swift-lm.
// Pixels are channel-last [batch, H, W, C]. Only the native image_size is supported: the
// reference bicubically resizes the position table for other sizes, and this overlay does not.
// UNVERIFIED: no reference numerics were run. Layout follows the HF modeling file.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIQianfanOcrVisionAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "projection_layer") var projectionLayer: Linear

    init(_ config: SharpAIQianfanOcrVisionConfiguration) {
        let headDim = config.hiddenSize / config.attentionHeads
        self.heads = config.attentionHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _projectionLayer.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        return projectionLayer(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAIQianfanOcrVisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: SharpAIQianfanOcrVisionConfiguration) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXNN.gelu(fc1(x)))
    }
}

/// Pre-norm block with per-channel layer scale (`ls1`, `ls2` in the checkpoint).
final class SharpAIQianfanOcrVisionLayer: Module {
    @ModuleInfo(key: "attention") var attention: SharpAIQianfanOcrVisionAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIQianfanOcrVisionMLP
    @ModuleInfo(key: "layernorm_before") var layernormBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var layernormAfter: LayerNorm
    @ParameterInfo(key: "lambda_1") var lambda1: MLXArray
    @ParameterInfo(key: "lambda_2") var lambda2: MLXArray

    init(_ config: SharpAIQianfanOcrVisionConfiguration) {
        _attention.wrappedValue = SharpAIQianfanOcrVisionAttention(config)
        _mlp.wrappedValue = SharpAIQianfanOcrVisionMLP(config)
        _layernormBefore.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _layernormAfter.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _lambda1.wrappedValue = MLXArray.ones([config.hiddenSize])
        _lambda2.wrappedValue = MLXArray.ones([config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + lambda1 * attention(layernormBefore(x))
        h = h + lambda2 * mlp(layernormAfter(h))
        return h
    }
}

/// Conv patch embedding. The conv stores channel-last weights in MLX.
final class SharpAIQianfanOcrPatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ config: SharpAIQianfanOcrVisionConfiguration) {
        _projection.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: true)
    }

    /// `[batch, H, W, C]` to `[batch, H / patch, W / patch, hidden]`.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        projection(pixels)
    }
}

/// CLS token, patch tokens, and learned absolute position embeddings.
final class SharpAIQianfanOcrVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: SharpAIQianfanOcrPatchEmbeddings
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    init(_ config: SharpAIQianfanOcrVisionConfiguration) {
        let grid = config.imageSize / config.patchSize
        _patchEmbeddings.wrappedValue = SharpAIQianfanOcrPatchEmbeddings(config)
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, grid * grid + 1, config.hiddenSize])
    }

    /// `[batch, H, W, C]` to `[batch, 1 + patches, hidden]`, CLS token first.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(pixels)
        let batch = patches.dim(0)
        let tokens = patches.reshaped(batch, -1, patches.dim(-1))
        let cls = repeated(clsToken, count: batch, axis: 0)
        return concatenated([cls, tokens], axis: 1) + positionEmbeddings
    }
}

/// InternViT encoder. The reference uses mean pooling, so there is no final LayerNorm.
public final class SharpAIQianfanOcrVisionTower: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SharpAIQianfanOcrVisionEmbeddings
    let layers: [SharpAIQianfanOcrVisionLayer]

    public init(_ config: SharpAIQianfanOcrVisionConfiguration) {
        _embeddings.wrappedValue = SharpAIQianfanOcrVisionEmbeddings(config)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIQianfanOcrVisionLayer(config)
        }
    }

    /// Returns `[batch, 1 + patches, hidden]`. The caller drops the CLS row.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var h = embeddings(pixels)
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}
