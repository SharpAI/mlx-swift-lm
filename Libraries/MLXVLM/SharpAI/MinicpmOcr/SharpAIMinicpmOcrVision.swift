// SharpAI fork overlay: minicpm_ocr vision tower, window merger and MLP merger.
// Not part of upstream mlx-swift-lm.
//
// Follows `MiniCPMOcrVisionModel`, `MiniCPMOcrViTWindowAttentionMerger` and `MiniCPMOcrMerger`
// in huggingface/transformers `modeling_minicpm_ocr.py`.
// Every slice is encoded on its own, so no cross-image attention mask is needed.
// UNVERIFIED: no checkpoint is on disk, so numerical parity with the reference is not checked.
// The NaViT bucket rule is an integer form of the reference `torch.bucketize` boundaries.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIMinicpmOcrVisionAttention: Module {
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

final class SharpAIMinicpmOcrVisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(_ config: SharpAIMinicpmOcrVisionConfiguration) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    /// `gelu_pytorch_tanh`, the tanh approximation.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

final class SharpAIMinicpmOcrVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinicpmOcrVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: SharpAIMinicpmOcrVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIMinicpmOcrVisionConfiguration) {
        self._selfAttn.wrappedValue = SharpAIMinicpmOcrVisionAttention(
            hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = SharpAIMinicpmOcrVisionMLP(config)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

final class SharpAIMinicpmOcrVisionEncoder: Module {
    var layers: [SharpAIMinicpmOcrVisionEncoderLayer]

    init(_ config: SharpAIMinicpmOcrVisionConfiguration) {
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            SharpAIMinicpmOcrVisionEncoderLayer(config)
        }
        super.init()
    }
}

final class SharpAIMinicpmOcrVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    let numPatchesPerSide: Int

    init(_ config: SharpAIMinicpmOcrVisionConfiguration) {
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
    /// `get_vision_nearest_position_ids` maps `i / height` to the count of boundaries `k / n`
    /// (k in 1..<n) that are at most that value. In integers: `k * height <= i * n`.
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

final class SharpAIMinicpmOcrVisionTower: Module {
    @ModuleInfo var embeddings: SharpAIMinicpmOcrVisionEmbeddings
    @ModuleInfo var encoder: SharpAIMinicpmOcrVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

    init(_ config: SharpAIMinicpmOcrVisionConfiguration) {
        self.embeddings = SharpAIMinicpmOcrVisionEmbeddings(config)
        self.encoder = SharpAIMinicpmOcrVisionEncoder(config)
        self._postLayernorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }
}

/// Window attention over 2x2 windows, then a residual MLP that halves the grid.
/// Input is `(H * W, hidden)` in row-major order. Output is `((H / 2) * (W / 2), hidden)`.
///
/// The reference reorders tokens into windows before attention. Here the windows are formed
/// directly from the row-major grid and reordered back, which gives the same token order.
/// The residual is the window mean of the post-attention states, as in the reference.
final class SharpAIMinicpmOcrWindowMerger: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinicpmOcrVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: SharpAIMinicpmOcrVisionConfiguration) {
        let hidden = config.hiddenSize
        self._selfAttn.wrappedValue = SharpAIMinicpmOcrVisionAttention(
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

        // Window attention over 2x2 windows (4 tokens each).
        let windows = layerNorm1(x)
            .reshaped(halfHeight, 2, halfWidth, 2, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(halfHeight * halfWidth, 4, hidden)
        let attended = selfAttn(windows)
            .reshaped(halfHeight, halfWidth, 2, 2, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(height * width, hidden)
        let attendedResidual = x + attended

        // MLP over each 2x2 window (4 * hidden in), with the window mean as residual.
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

/// One `DownsampleMLP` block. Input is `(N, 4 * hidden)`, output is `(N, outDim)`.
/// The reference uses an exact `nn.GELU`.
final class SharpAIMinicpmOcrDownsampleMLP: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(hiddenSize: Int, outDim: Int) {
        let inner = hiddenSize * 4
        self._preNorm.wrappedValue = LayerNorm(dimensions: inner, eps: 1e-6)
        self._linear1.wrappedValue = Linear(inner, inner, bias: true)
        self._linear2.wrappedValue = Linear(inner, outDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(preNorm(x))))
    }
}

/// MLP merger (`MiniCPMOcrMerger`). Applies `merger_times` rounds of 2x2 pixel shuffle, each
/// followed by one `DownsampleMLP`. The last block projects into the LLM width.
final class SharpAIMinicpmOcrMerger: Module {
    var mlp: [SharpAIMinicpmOcrDownsampleMLP]
    let mergerTimes: Int

    init(hiddenSize: Int, llmDim: Int, mergerTimes: Int) {
        let times = max(mergerTimes, 1)
        var blocks = [SharpAIMinicpmOcrDownsampleMLP]()
        for _ in 0 ..< (times - 1) {
            blocks.append(SharpAIMinicpmOcrDownsampleMLP(hiddenSize: hiddenSize, outDim: hiddenSize))
        }
        blocks.append(SharpAIMinicpmOcrDownsampleMLP(hiddenSize: hiddenSize, outDim: llmDim))
        self.mlp = blocks
        self.mergerTimes = times
        super.init()
    }

    /// `x` is `(height * width, hidden)` row-major. Returns `(tokens, llmDim)` with
    /// `tokens = (height / 2^times) * (width / 2^times)`. Callers check divisibility.
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        var tokens = x
        var gridHeight = height
        var gridWidth = width
        for index in 0 ..< mergerTimes {
            let halfHeight = gridHeight / 2
            let halfWidth = gridWidth / 2
            let grouped = tokens
                .reshaped(halfHeight, 2, halfWidth, 2, tokens.dim(-1))
                .transposed(0, 2, 1, 3, 4)
                .reshaped(halfHeight * halfWidth, 4 * tokens.dim(-1))
            tokens = mlp[index](grouped)
            gridHeight = halfHeight
            gridWidth = halfWidth
        }
        return tokens
    }
}
