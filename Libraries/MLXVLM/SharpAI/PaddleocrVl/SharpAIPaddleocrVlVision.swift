// SharpAI fork overlay: PaddleOCR-VL vision tower and projector. Not part of upstream mlx-swift-lm.
//
// SigLIP-style encoder with absolute position embeddings, resized bilinearly to each image grid.
// Images run one at a time, so attention needs no packing mask. Attention rotates q and k with a
// 2D (row, column) rotary embedding, as in the mlx-vlm port. The projector merges 2x2 patch
// tokens and maps them to the text width.
// Pixel input: (patches, P, P, 3), channels last, in row-major patch order for each image.
// The fixture lists temporal_patch_size 2, but this file uses a 2D patch conv for still images.
// UNVERIFIED: the 2D conv and the projector's pre-norm placement and GELU variant. Check them
// against HF output.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// 2D vision rotary embedding tables. Each axis (row, column) gets headDim / 4 frequencies with
/// base 10000, so the pair covers headDim / 2; cos and sin are tiled to headDim (half-split
/// rotation). Returns cos and sin shaped (1, 1, height * width, headDim), patches in row-major order.
func sharpAIPaddleocrVlVisionRotary(height: Int, width: Int, headDim: Int) -> (
    cos: MLXArray, sin: MLXArray
) {
    let axisDim = headDim / 2
    let count = axisDim / 2
    let invFreq = (0 ..< count).map { index in
        Float(1) / pow(Float(10000), Float(2 * index) / Float(axisDim))
    }
    var angles = [Float]()
    angles.reserveCapacity(height * width * 2 * count)
    for row in 0 ..< height {
        for column in 0 ..< width {
            for freq in invFreq { angles.append(Float(row) * freq) }
            for freq in invFreq { angles.append(Float(column) * freq) }
        }
    }
    let freqs = MLXArray(angles, [height * width, 2 * count])
    let tiled = concatenated([freqs, freqs], axis: -1)
        .reshaped(1, 1, height * width, 2 * axisDim)
    return (cos(tiled), sin(tiled))
}

func sharpAIPaddleocrVlRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let first = x[.ellipsis, 0 ..< half]
    let second = x[.ellipsis, half...]
    return concatenated([-second, first], axis: -1)
}

final class SharpAIPaddleocrVlVisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(hiddenSize / numHeads), -0.5)
        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    /// Bidirectional attention over (batch, length, hidden). `cos` and `sin` are the tables from
    /// `sharpAIPaddleocrVlVisionRotary`.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = numHeads * headDim
        var q = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let qCos = cos.asType(q.dtype)
        let qSin = sin.asType(q.dtype)
        q = q * qCos + sharpAIPaddleocrVlRotateHalf(q) * qSin
        k = k * qCos + sharpAIPaddleocrVlRotateHalf(k) * qSin
        let v = vProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, hidden)
        return outProj(attended)
    }
}

final class SharpAIPaddleocrVlVisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(_ config: SharpAIPaddleocrVlVisionConfiguration) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    /// `gelu_pytorch_tanh`, the tanh approximation.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

final class SharpAIPaddleocrVlVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIPaddleocrVlVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: SharpAIPaddleocrVlVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIPaddleocrVlVisionConfiguration) {
        self._selfAttn.wrappedValue = SharpAIPaddleocrVlVisionAttention(
            hiddenSize: config.hiddenSize, numHeads: config.attentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = SharpAIPaddleocrVlVisionMLP(config)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x), cos: cos, sin: sin)
        return h + mlp(layerNorm2(h))
    }
}

final class SharpAIPaddleocrVlVisionEncoder: Module {
    var layers: [SharpAIPaddleocrVlVisionEncoderLayer]

    init(_ config: SharpAIPaddleocrVlVisionConfiguration) {
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIPaddleocrVlVisionEncoderLayer(config)
        }
        super.init()
    }
}

final class SharpAIPaddleocrVlVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    let numPatchesPerSide: Int

    init(_ config: SharpAIPaddleocrVlVisionConfiguration) {
        self.numPatchesPerSide = config.patchesPerSide
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize)
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPatchesPerSide * numPatchesPerSide,
            dimensions: config.hiddenSize
        )
        super.init()
    }

    /// Weights (outputSize, inputSize) for bilinear resizing, align_corners = false.
    static func bilinearWeights(outputSize: Int, inputSize: Int) -> MLXArray {
        var weights = [Float](repeating: 0, count: outputSize * inputSize)
        let scale = Float(inputSize) / Float(outputSize)
        for index in 0 ..< outputSize {
            let source = min(
                max((Float(index) + 0.5) * scale - 0.5, 0), Float(inputSize - 1))
            let low = Int(source)
            let high = min(low + 1, inputSize - 1)
            let fraction = source - Float(low)
            weights[index * inputSize + low] += 1 - fraction
            weights[index * inputSize + high] += fraction
        }
        return MLXArray(weights, [outputSize, inputSize])
    }

    /// Position embeddings resized to a height x width patch grid: (height * width, hidden).
    func interpolatedPositions(height: Int, width: Int) -> MLXArray {
        let n = numPatchesPerSide
        let hidden = positionEmbedding.weight.dim(-1)
        let grid = positionEmbedding.weight.reshaped(n, n, hidden)
        if height == n && width == n {
            return grid.reshaped(n * n, hidden)
        }

        let rows = SharpAIPaddleocrVlVisionEmbeddings.bilinearWeights(
            outputSize: height, inputSize: n
        ).asType(grid.dtype)
        let cols = SharpAIPaddleocrVlVisionEmbeddings.bilinearWeights(
            outputSize: width, inputSize: n
        ).asType(grid.dtype)

        // Resize rows, then columns.
        let byRow = matmul(rows, grid.reshaped(n, n * hidden)).reshaped(height, n, hidden)
        let byColumn = matmul(cols, byRow.transposed(1, 0, 2).reshaped(n, height * hidden))
            .reshaped(width, height, hidden)
            .transposed(1, 0, 2)
        return byColumn.reshaped(height * width, hidden)
    }

    /// `pixels` is (height * width, P, P, C), channels last. Returns (height * width, hidden).
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let dtype = patchEmbedding.weight.dtype
        let patches = patchEmbedding(pixels.asType(dtype)).reshaped(height * width, -1)
        return patches + interpolatedPositions(height: height, width: width).asType(patches.dtype)
    }
}

final class SharpAIPaddleocrVlVisionTower: Module {
    @ModuleInfo var embeddings: SharpAIPaddleocrVlVisionEmbeddings
    @ModuleInfo var encoder: SharpAIPaddleocrVlVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

    init(_ config: SharpAIPaddleocrVlVisionConfiguration) {
        self.embeddings = SharpAIPaddleocrVlVisionEmbeddings(config)
        self.encoder = SharpAIPaddleocrVlVisionEncoder(config)
        self._postLayernorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    /// One image: `pixels` is (height * width, P, P, C). Returns (height * width, hidden).
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        var hidden = embeddings(pixels, height: height, width: width).expandedDimensions(axis: 0)
        let headDim = encoder.layers.first?.selfAttn.headDim ?? 0
        let rotary = sharpAIPaddleocrVlVisionRotary(height: height, width: width, headDim: headDim)
        for layer in encoder.layers {
            hidden = layer(hidden, cos: rotary.cos, sin: rotary.sin)
        }
        return postLayernorm(hidden)[0]
    }
}

/// Merges each 2x2 block of patch tokens into one token, then maps it to the text width.
final class SharpAIPaddleocrVlProjector: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let mergeSize: Int

    init(_ config: SharpAIPaddleocrVlVisionConfiguration, textHiddenSize: Int) {
        let mergeSize = config.spatialMergeSize
        let merged = config.hiddenSize * mergeSize * mergeSize
        self.mergeSize = mergeSize
        self._preNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._linear1.wrappedValue = Linear(merged, merged, bias: true)
        self._linear2.wrappedValue = Linear(merged, textHiddenSize, bias: true)
        super.init()
    }

    /// `x` is (height * width, hidden) in row-major order. Returns
    /// ((height / m) * (width / m), textHidden) in row-major order of the merged grid.
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let hidden = x.dim(-1)
        let m = mergeSize
        let grouped = preNorm(x)
            .reshaped(height / m, m, width / m, m, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped((height / m) * (width / m), m * m * hidden)
        return linear2(gelu(linear1(grouped)))
    }
}
