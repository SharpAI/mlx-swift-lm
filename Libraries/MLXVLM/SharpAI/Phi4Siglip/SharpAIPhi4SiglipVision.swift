// SharpAI fork overlay: SigLIP2 NaFlex vision tower for phi4-siglip. Not part of upstream.
//
// Input is flat patch rows `[N, C * p * p]` with the grid (height, width) in patches. The patch
// projection is a Linear, as in HF Siglip2. The learned position table is resized to the grid
// with bilinear weights (align_corners false). One image is run at a time, so no padding mask.
// UNVERIFIED: antialias is not applied when the grid is smaller than the table, and the output
// uses post_layernorm on the last layer.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Dense bilinear weights `[output, input]` for `align_corners=false` resampling.
private func sharpAIPhi4SiglipInterpolation(output: Int, input: Int) -> MLXArray {
    var weights = [Float](repeating: 0, count: output * input)
    let ratio = Float(input) / Float(output)
    for i in 0 ..< output {
        let source = min(max((Float(i) + 0.5) * ratio - 0.5, 0), Float(input - 1))
        let low = Int(source.rounded(.down))
        let high = min(low + 1, input - 1)
        let fraction = source - Float(low)
        weights[i * input + low] += 1 - fraction
        weights[i * input + high] += fraction
    }
    return MLXArray(weights, [output, input])
}

final class SharpAIPhi4SiglipVisionAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        let headDim = config.hiddenSize / config.attentionHeads
        self.heads = config.attentionHeads
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return outProj(out)
    }
}

final class SharpAIPhi4SiglipVisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let tanhApproximation: Bool

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        tanhApproximation = config.hiddenAct == "gelu_pytorch_tanh"
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = fc1(x)
        return fc2(tanhApproximation ? geluApproximate(hidden) : MLXNN.gelu(hidden))
    }
}

final class SharpAIPhi4SiglipVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIPhi4SiglipVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    let mlp: SharpAIPhi4SiglipVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        _attention.wrappedValue = SharpAIPhi4SiglipVisionAttention(config)
        _layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = SharpAIPhi4SiglipVisionMLP(config)
        _layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(layerNorm1(x))
        h = h + mlp(layerNorm2(h))
        return h
    }
}

final class SharpAIPhi4SiglipVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    let gridSide: Int

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        let patchDimension = config.channels * config.patchSize * config.patchSize
        self.gridSide = Int(Double(config.positionCount).squareRoot().rounded())
        _patchEmbedding.wrappedValue = Linear(patchDimension, config.hiddenSize, bias: true)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.positionCount, dimensions: config.hiddenSize)
    }

    /// Position table `[gridHeight * gridWidth, hidden]` resized from the square table.
    func positions(gridHeight: Int, gridWidth: Int) -> MLXArray {
        let table = positionEmbedding.weight
            .reshaped(gridSide, gridSide, -1)
            .transposed(2, 0, 1)
        let rows = sharpAIPhi4SiglipInterpolation(output: gridHeight, input: gridSide)
            .asType(table.dtype)
        let cols = sharpAIPhi4SiglipInterpolation(output: gridWidth, input: gridSide)
            .asType(table.dtype)
        let resized = matmul(matmul(rows, table), cols.transposed(1, 0))
        return resized.transposed(1, 2, 0).reshaped(gridHeight * gridWidth, -1)
    }

    /// `patches` is `[1, N, C * p * p]`. Returns `[1, N, hidden]`.
    func callAsFunction(_ patches: MLXArray, gridHeight: Int, gridWidth: Int) -> MLXArray {
        let embedded = patchEmbedding(patches.asType(patchEmbedding.weight.dtype))
        let position = positions(gridHeight: gridHeight, gridWidth: gridWidth)
        return embedded + position.expandedDimensions(axis: 0).asType(embedded.dtype)
    }
}

final class SharpAIPhi4SiglipVisionEncoder: Module {
    let layers: [SharpAIPhi4SiglipVisionEncoderLayer]

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIPhi4SiglipVisionEncoderLayer(config)
        }
    }
}

/// HF `Siglip2VisionTransformer`: embeddings, encoder, post_layernorm. No pooling head.
final class SharpAIPhi4SiglipVisionTransformer: Module {
    let embeddings: SharpAIPhi4SiglipVisionEmbeddings
    let encoder: SharpAIPhi4SiglipVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        self.embeddings = SharpAIPhi4SiglipVisionEmbeddings(config)
        self.encoder = SharpAIPhi4SiglipVisionEncoder(config)
        _postLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    /// `patches` is `[N, C * p * p]`. Returns `[N, hidden]`.
    func callAsFunction(_ patches: MLXArray, gridHeight: Int, gridWidth: Int) -> MLXArray {
        var h = embeddings(
            patches.expandedDimensions(axis: 0), gridHeight: gridHeight, gridWidth: gridWidth)
        for layer in encoder.layers {
            h = layer(h)
        }
        return postLayerNorm(h)[0]
    }
}

/// Wrapper matching HF `Siglip2VisionModel`, so keys nest as `vision_model.*`.
final class SharpAIPhi4SiglipVisionModel: Module {
    @ModuleInfo(key: "vision_model") var visionModel: SharpAIPhi4SiglipVisionTransformer

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        _visionModel.wrappedValue = SharpAIPhi4SiglipVisionTransformer(config)
    }
}

/// LLaVA-style tower holder, so keys nest as `vision_tower.vision_tower.vision_model.*`.
final class SharpAIPhi4SiglipVisionTower: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: SharpAIPhi4SiglipVisionModel

    init(_ config: SharpAIPhi4SiglipVisionConfiguration) {
        _visionTower.wrappedValue = SharpAIPhi4SiglipVisionModel(config)
    }

    /// Returns `[N, hidden]` for one image's patch rows.
    func callAsFunction(_ patches: MLXArray, gridHeight: Int, gridWidth: Int) -> MLXArray {
        visionTower.visionModel(patches, gridHeight: gridHeight, gridWidth: gridWidth)
    }
}
