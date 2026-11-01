// SharpAI fork overlay: MiniCPM-V 4.7 vision tower, window-attention merger and MLP merger.
// Not part of upstream mlx-swift-lm.
//
// Reference: transformers modeling_minicpmv4_7.py (MiniCPMV4_7VisionModel,
// MiniCPMV4_7ViTWindowAttentionMerger, MiniCPMV4_7Merger, MiniCPMV4_7DownsampleMLP) and
// vision_utils.py (get_vision_window_index, get_vision_nearest_position_ids).
//
// Every slice is encoded on its own, so no padding mask is needed.
// UNVERIFIED: the NaViT bucket rule is integer arithmetic. Torch computes it in float32, so an
// exact tie (i / h == k / n) can fall on the other side of a bucket boundary. Check logits
// against the HF reference before trusting them.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIMinicpmv47VisionAttention: Module {
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

final class SharpAIMinicpmv47VisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(_ config: SharpAIMinicpmv47VisionConfiguration) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    /// `gelu_pytorch_tanh`, the tanh approximation.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

final class SharpAIMinicpmv47VisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinicpmv47VisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: SharpAIMinicpmv47VisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIMinicpmv47VisionConfiguration) {
        self._selfAttn.wrappedValue = SharpAIMinicpmv47VisionAttention(
            hiddenSize: config.hiddenSize, numHeads: config.numAttentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = SharpAIMinicpmv47VisionMLP(config)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

final class SharpAIMinicpmv47VisionEncoder: Module {
    var layers: [SharpAIMinicpmv47VisionEncoderLayer]

    init(_ config: SharpAIMinicpmv47VisionConfiguration) {
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            SharpAIMinicpmv47VisionEncoderLayer(config)
        }
        super.init()
    }
}

final class SharpAIMinicpmv47VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    let numPatchesPerSide: Int

    init(_ config: SharpAIMinicpmv47VisionConfiguration) {
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

    /// Row-major bucket ids for a `height x width` patch grid, as `get_vision_nearest_position_ids`.
    ///
    /// Each patch maps to bucket `(bh, bw)` on the pretrained `n x n` table. `bh` counts the
    /// boundaries `k / n` (k in 1..<n) that are at most `i / height`.
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

    /// `pixels` is `(1, H, W, 3)` channels-last. Returns `(H * W, hidden)` in raster order.
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let patches = patchEmbedding(pixels).reshaped(height * width, -1)
        let ids = MLXArray(positionIds(height: height, width: width).map { Int32($0) })
        return patches + positionEmbedding(ids)
    }
}

final class SharpAIMinicpmv47VisionTower: Module {
    @ModuleInfo var embeddings: SharpAIMinicpmv47VisionEmbeddings
    @ModuleInfo var encoder: SharpAIMinicpmv47VisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

    init(_ config: SharpAIMinicpmv47VisionConfiguration) {
        self.embeddings = SharpAIMinicpmv47VisionEmbeddings(config)
        self.encoder = SharpAIMinicpmv47VisionEncoder(config)
        self._postLayernorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }
}

/// Window attention over 2x2 patch windows, then an MLP that merges each window into one token.
/// Input is `(H * W, hidden)` in raster order. Output is `((H / 2) * (W / 2), hidden)` in
/// row-major window order, which is the raster order of the halved grid.
final class SharpAIMinicpmv47WindowMerger: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinicpmv47VisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: SharpAIMinicpmv47VisionConfiguration) {
        let hidden = config.hiddenSize
        // window_kernel_size is (2, 2) in the 1B config, so one window holds 4 patches.
        self._selfAttn.wrappedValue = SharpAIMinicpmv47VisionAttention(
            hiddenSize: hidden, numHeads: config.numAttentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: hidden, eps: config.layerNormEps)
        self._preNorm.wrappedValue = LayerNorm(dimensions: hidden * 4, eps: config.layerNormEps)
        self._linear1.wrappedValue = Linear(hidden * 4, config.intermediateSize * 4, bias: true)
        self._linear2.wrappedValue = Linear(config.intermediateSize * 4, hidden, bias: true)
        super.init()
    }

    /// Raster index of each patch in window order: windows row-major, patches row-major inside a
    /// window. Mirrors `get_vision_window_index` with a 2x2 window and no spatial merge.
    static func windowOrder(height: Int, width: Int) -> [Int32] {
        var order = [Int32]()
        order.reserveCapacity(height * width)
        for windowRow in 0 ..< height / 2 {
            for windowCol in 0 ..< width / 2 {
                for row in 0 ..< 2 {
                    for col in 0 ..< 2 {
                        let r = 2 * windowRow + row
                        let c = 2 * windowCol + col
                        order.append(Int32(r * width + c))
                    }
                }
            }
        }
        return order
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let hidden = x.dim(-1)
        let windowCount = (height / 2) * (width / 2)
        let order = MLXArray(Self.windowOrder(height: height, width: width))

        // Attention inside each 2x2 window. The residual is the un-normalized input, as in HF.
        let normed = layerNorm1(x).take(order, axis: 0).reshaped(windowCount, 4, hidden)
        let attended = selfAttn(normed)
        let residual = x.take(order, axis: 0).reshaped(windowCount, 4, hidden)
        let windowed = residual + attended

        // MLP over each window, with the window mean as a residual.
        let patchResidual = windowed.mean(axis: 1)
        var merged = preNorm(windowed.reshaped(windowCount, 4 * hidden))
        merged = linear1(merged)
        merged = geluApproximate(merged)
        merged = linear2(merged)
        return merged + patchResidual
    }
}

/// One `DownsampleMLP` block. Input is `(N, 4 * hidden)`, output is `(N, outDim)`.
final class SharpAIMinicpmv47DownsampleMLP: Module {
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

    /// Exact GELU, which is `nn.GELU()` in the HF reference.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(preNorm(x))))
    }
}

/// MLP merger. Each round merges 2x2 neighbours and projects. The last round projects to the
/// LLM width. With `merger_times = 1` there is one round.
final class SharpAIMinicpmv47Merger: Module {
    var mlp: [SharpAIMinicpmv47DownsampleMLP]
    let times: Int

    init(hiddenSize: Int, llmDim: Int, times: Int) {
        self.times = times
        var blocks = (0 ..< max(times - 1, 0)).map { _ in
            SharpAIMinicpmv47DownsampleMLP(hiddenSize: hiddenSize, outDim: hiddenSize)
        }
        blocks.append(SharpAIMinicpmv47DownsampleMLP(hiddenSize: hiddenSize, outDim: llmDim))
        self.mlp = blocks
        super.init()
    }

    /// Groups `(height * width, C)` into 2x2 blocks: `((height / 2) * (width / 2), 4 * C)`.
    static func group(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let hidden = x.dim(-1)
        return x
            .reshaped(height / 2, 2, width / 2, 2, hidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped((height / 2) * (width / 2), 4 * hidden)
    }

    /// `x` is `(height * width, hidden)` on the grid the caller passes. Returns
    /// `(tokens, llmDim)`.
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) throws -> MLXArray {
        guard height % 2 == 0, width % 2 == 0 else {
            throw SharpAIMinicpmv47Error.invalidImageGrid(height: height, width: width, mergeFactor: 2)
        }
        var hidden = mlp[0](Self.group(x, height: height, width: width))
        var gridHeight = height / 2
        var gridWidth = width / 2
        for index in 1 ..< times {
            guard gridHeight % 2 == 0, gridWidth % 2 == 0 else {
                throw SharpAIMinicpmv47Error.invalidImageGrid(
                    height: gridHeight, width: gridWidth, mergeFactor: 2)
            }
            hidden = mlp[index](Self.group(hidden, height: gridHeight, width: gridWidth))
            gridHeight /= 2
            gridWidth /= 2
        }
        return hidden
    }
}
