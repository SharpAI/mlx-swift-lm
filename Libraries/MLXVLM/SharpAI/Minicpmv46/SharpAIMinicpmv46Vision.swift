// SharpAI fork overlay: MiniCPM-V 4.6 vision tower, window merger and MLP merger.
// Not part of upstream mlx-swift-lm.
//
// Every slice is encoded on its own, so no padding mask is needed.
// UNVERIFIED: the bucket rule for NaViT position ids follows Idefics2 (integer comparison
// instead of torch float bucketize). Check it against HF output before trusting logits.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIMinicpmv46VisionAttention: Module {
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

    /// Bidirectional attention over `(batch, length, hidden)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = numHeads * headDim
        let q = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, hidden)
        return outProj(attended)
    }
}

final class SharpAIMinicpmv46VisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(_ config: SharpAIMinicpmv46VisionConfiguration) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    /// `gelu_pytorch_tanh`, the tanh approximation.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

final class SharpAIMinicpmv46VisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinicpmv46VisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: SharpAIMinicpmv46VisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIMinicpmv46VisionConfiguration) {
        self._selfAttn.wrappedValue = SharpAIMinicpmv46VisionAttention(
            hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = SharpAIMinicpmv46VisionMLP(config)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

final class SharpAIMinicpmv46VisionEncoder: Module {
    var layers: [SharpAIMinicpmv46VisionEncoderLayer]

    init(_ config: SharpAIMinicpmv46VisionConfiguration, layerCount: Int) {
        self.layers = (0 ..< layerCount).map { _ in
            SharpAIMinicpmv46VisionEncoderLayer(config)
        }
        super.init()
    }
}

final class SharpAIMinicpmv46VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    let numPatchesPerSide: Int

    init(_ config: SharpAIMinicpmv46VisionConfiguration) {
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize)
        )
        self.numPatchesPerSide = config.imageSize / config.patchSize
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPatchesPerSide * numPatchesPerSide,
            dimensions: config.hiddenSize
        )
        super.init()
    }

    /// Row-major bucket ids for an `height x width` patch grid.
    ///
    /// Each patch maps to bucket `(bh, bw)` on the pretrained `n x n` position grid.
    /// `bh` counts the boundaries `k / n` (k in 1..<n) that are at most `i / height`.
    func positionIds(height: Int, width: Int) -> [Int] {
        let n = numPatchesPerSide
        func bucket(_ index: Int, _ total: Int) -> Int {
            var count = 0
            for k in 1 ..< n where k * total <= index * n {
                count += 1
            }
            return count
        }
        var ids = [Int]()
        ids.reserveCapacity(height * width)
        for i in 0 ..< height {
            let bh = bucket(i, height)
            for j in 0 ..< width {
                ids.append(bh * n + bucket(j, width))
            }
        }
        return ids
    }

    /// `pixels` is `(1, H, W, 3)` channels-last. Returns `(H * W, hidden)`.
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let patches = patchEmbedding(pixels).reshaped(height * width, -1)
        let ids = MLXArray(positionIds(height: height, width: width).map { Int32($0) })
        return patches + positionEmbedding(ids)
    }
}

final class SharpAIMinicpmv46VisionTower: Module {
    @ModuleInfo var embeddings: SharpAIMinicpmv46VisionEmbeddings
    @ModuleInfo var encoder: SharpAIMinicpmv46VisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

    init(_ config: SharpAIMinicpmv46VisionConfiguration, dropLastLayer: Bool) {
        let layerCount =
            dropLastLayer
            ? max(config.numHiddenLayers - 1, 0) : config.numHiddenLayers
        self.embeddings = SharpAIMinicpmv46VisionEmbeddings(config)
        self.encoder = SharpAIMinicpmv46VisionEncoder(config, layerCount: layerCount)
        self._postLayernorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }
}

/// Window attention over 2x2 windows, then a residual MLP that halves the grid.
/// Input and output are `(H * W, hidden)` and `((H / 2) * (W / 2), hidden)`.
/// UNVERIFIED: tensor names, widths and the window-mean residual are guesses. No reference here.
final class SharpAIMinicpmv46WindowMerger: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinicpmv46VisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: SharpAIMinicpmv46VisionConfiguration) {
        let hidden = config.hiddenSize
        self._selfAttn.wrappedValue = SharpAIMinicpmv46VisionAttention(
            hiddenSize: hidden, numHeads: config.numAttentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: hidden, eps: config.layerNormEps)
        self._preNorm.wrappedValue = LayerNorm(dimensions: hidden * 4, eps: config.layerNormEps)
        self._linear1.wrappedValue = Linear(hidden * 4, config.intermediateSize * 4, bias: true)
        self._linear2.wrappedValue = Linear(config.intermediateSize * 4, hidden, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let hidden = x.dim(-1)
        let halfHeight = height / 2
        let halfWidth = width / 2

        // Window attention over 2x2 windows.
        let windows = layerNorm1(x)
            .reshaped(halfHeight, 2, halfWidth, 2, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(halfHeight * halfWidth, 4, hidden)
        let attended = selfAttn(windows)
            .reshaped(halfHeight, halfWidth, 2, 2, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(height * width, hidden)
        let attendedResidual = x + attended

        // MLP downsample of each 2x2 window, with the window mean as residual.
        let grouped = attendedResidual
            .reshaped(halfHeight, 2, halfWidth, 2, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(halfHeight * halfWidth, 4, hidden)
        let residual = grouped.mean(axis: 1)
        var merged = grouped.reshaped(halfHeight * halfWidth, 4 * hidden)
        merged = preNorm(merged)
        merged = linear1(merged)
        merged = geluApproximate(merged)
        merged = linear2(merged)
        return merged + residual
    }
}

/// One `DownsampleMLP` block. Input is `(N, 4 * hidden)`, output is `(N, llmDim)`.
final class SharpAIMinicpmv46DownsampleMLP: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(hiddenSize: Int, llmDim: Int) {
        let inner = hiddenSize * 4
        self._preNorm.wrappedValue = LayerNorm(dimensions: inner, eps: 1e-6)
        self._linear1.wrappedValue = Linear(inner, inner, bias: true)
        self._linear2.wrappedValue = Linear(inner, llmDim, bias: true)
        super.init()
    }

    /// Exact GELU. UNVERIFIED: the checkpoint activation is not in the fixture.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(preNorm(x))))
    }
}

/// MLP merger with one stage (`times = 1`). Halves the grid with a 2x2 pixel shuffle.
/// UNVERIFIED: `merger.mlp.0` key names and the `hidden * 4` norm width are guesses.
final class SharpAIMinicpmv46Merger: Module {
    var mlp: [SharpAIMinicpmv46DownsampleMLP]

    init(hiddenSize: Int, llmDim: Int) {
        self.mlp = [SharpAIMinicpmv46DownsampleMLP(hiddenSize: hiddenSize, llmDim: llmDim)]
        super.init()
    }

    /// `x` is `(height * width, hidden)`. Returns `((height / 2) * (width / 2), llmDim)`.
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let hidden = x.dim(-1)
        let grouped = x
            .reshaped(height / 2, 2, width / 2, 2, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped((height / 2) * (width / 2), 4 * hidden)
        return mlp[0](grouped)
    }
}
