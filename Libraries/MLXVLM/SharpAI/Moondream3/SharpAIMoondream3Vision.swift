// SharpAI fork overlay: moondream3 vision encoder, crop reconstruction and projector.
// Not part of upstream mlx-swift-lm.
//
// Port of vLLM's model_executor/models/moondream3.py (vision path). Patches are flattened in
// PyTorch order (channel, row, col) so `patch_emb` weights load unchanged.
// UNVERIFIED: vLLM is the only source here. Its pooling uses enc_n_layers as the output side,
// and this port keeps that rule.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Two-layer MLP with tanh-approximate GELU. Used for the vision blocks and the projector.
final class SharpAIMoondream3GeluMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        _fc1.wrappedValue = Linear(inputDim, hiddenDim, bias: true)
        _fc2.wrappedValue = Linear(hiddenDim, outputDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

final class SharpAIMoondream3VisionAttention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    init(_ config: SharpAIMoondream3VisionConfiguration) {
        numHeads = config.encHeads
        headDim = config.encDim / config.encHeads
        _qkv.wrappedValue = Linear(config.encDim, 3 * config.encDim, bias: true)
        _proj.wrappedValue = Linear(config.encDim, config.encDim, bias: true)
        super.init()
    }

    /// Bidirectional self-attention over `[B, P, enc]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let parts = qkv(x).split(parts: 3, axis: -1)
        let q = parts[0].reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = parts[1].reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: pow(Float(headDim), -0.5), mask: .none)
        return proj(out.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAIMoondream3VisionBlock: Module {
    @ModuleInfo var ln1: LayerNorm
    @ModuleInfo var attn: SharpAIMoondream3VisionAttention
    @ModuleInfo var ln2: LayerNorm
    @ModuleInfo var mlp: SharpAIMoondream3GeluMLP

    init(_ config: SharpAIMoondream3VisionConfiguration) {
        _ln1.wrappedValue = LayerNorm(dimensions: config.encDim, eps: 1e-5)
        _attn.wrappedValue = SharpAIMoondream3VisionAttention(config)
        _ln2.wrappedValue = LayerNorm(dimensions: config.encDim, eps: 1e-5)
        _mlp.wrappedValue = SharpAIMoondream3GeluMLP(
            inputDim: config.encDim, hiddenDim: config.encFFDim, outputDim: config.encDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attn(ln1(x))
        h = h + mlp(ln2(h))
        return h
    }
}

/// SigLIP-style encoder plus the `proj_mlp` projector. Keys sit under `vision.*`.
final class SharpAIMoondream3Vision: Module {
    @ModuleInfo(key: "patch_emb") var patchEmb: Linear
    @ParameterInfo(key: "pos_emb") var posEmb: MLXArray
    let blocks: [SharpAIMoondream3VisionBlock]
    @ModuleInfo(key: "post_ln") var postLn: LayerNorm
    @ModuleInfo(key: "proj_mlp") var projMLP: SharpAIMoondream3GeluMLP

    let patchSize: Int
    let gridSize: Int
    let encDim: Int
    let overlapMargin: Int

    init(_ config: SharpAIMoondream3VisionConfiguration, textDim: Int) {
        patchSize = config.patchSize
        gridSize = config.gridSize
        encDim = config.encDim
        overlapMargin = config.overlapMargin

        _patchEmb.wrappedValue = Linear(
            config.patchSize * config.patchSize * 3, config.encDim, bias: true)
        _posEmb.wrappedValue = MLXArray.zeros([1, config.gridSize * config.gridSize, config.encDim])
        self.blocks = (0 ..< config.encLayers).map { _ in
            SharpAIMoondream3VisionBlock(config)
        }
        _postLn.wrappedValue = LayerNorm(dimensions: config.encDim, eps: 1e-5)
        _projMLP.wrappedValue = SharpAIMoondream3GeluMLP(
            inputDim: config.encDim * 2, hiddenDim: config.projInnerDim, outputDim: textDim)
        super.init()
    }

    /// `[N, C, H, W]` to `[N, P, C * p * p]`, channel-major within a patch.
    static func patchify(_ pixels: MLXArray, patchSize p: Int) -> MLXArray {
        let (n, c, h, w) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let channelsLast = pixels.transposed(0, 2, 3, 1)
        // [N, i, u, j, v, C] to [N, i, j, C, u, v]
        return channelsLast
            .reshaped(n, h / p, p, w / p, p, c)
            .transposed(0, 1, 3, 5, 2, 4)
            .reshaped(n, (h / p) * (w / p), c * p * p)
    }

    /// Encodes `[N, C, S, S]` crops to `[N, P, enc]`.
    func encode(_ pixels: MLXArray) -> MLXArray {
        let dtype = patchEmb.weight.dtype
        var h = patchEmb(Self.patchify(pixels.asType(dtype), patchSize: patchSize))
        h = h + posEmb
        for block in blocks {
            h = block(h)
        }
        return postLn(h)
    }

    /// Projected rows `[P, text_dim]` for one image. Multi-crop features are merged first.
    func imageRows(_ pixels: MLXArray, tiling: (rows: Int, cols: Int)?) throws -> MLXArray {
        let crops = pixels.dim(0)
        let features = encode(pixels)
        let global = features[0]

        let recon: MLXArray
        if crops > 1 {
            guard let tiling else {
                throw SharpAIMoondream3Error.missingTiling
            }
            guard tiling.rows * tiling.cols == crops - 1 else {
                throw SharpAIMoondream3Error.tilingMismatch(
                    crops: crops - 1, tiles: tiling.rows * tiling.cols)
            }
            let local = features[1...].reshaped(crops - 1, gridSize, gridSize, encDim)
            recon = Self.reconstruct(local, tiling: tiling, overlapMargin: overlapMargin)
        } else {
            recon = global.reshaped(gridSize, gridSize, encDim)
        }

        let pooled = Self.adaptiveAveragePool(recon, outputSize: gridSize)
        let merged = MLX.concatenated(
            [global, pooled.reshaped(gridSize * gridSize, encDim)], axis: -1)
        return projMLP(merged)
    }

    /// Stitches overlapping local feature grids back into one `[H, W, C]` grid.
    /// Port of vLLM's `reconstruct_from_crops`, with patch size 1.
    static func reconstruct(
        _ crops: MLXArray, tiling: (rows: Int, cols: Int), overlapMargin m: Int
    ) -> MLXArray {
        let cropH = crops.dim(1)
        let cropW = crops.dim(2)
        var stacked: [MLXArray] = []

        for r in 0 ..< tiling.rows {
            let y0 = r == 0 ? 0 : m
            let y1 = r == tiling.rows - 1 ? cropH : cropH - m
            var segments: [MLXArray] = []
            for c in 0 ..< tiling.cols {
                let x0 = c == 0 ? 0 : m
                let x1 = c == tiling.cols - 1 ? cropW : cropW - m
                let crop = crops[r * tiling.cols + c]
                segments.append(crop[y0 ..< y1, x0 ..< x1, 0...])
            }
            stacked.append(MLX.concatenated(segments, axis: 1))
        }
        return MLX.concatenated(stacked, axis: 0)
    }

    /// Averages `[H, W, C]` down to `[outputSize, outputSize, C]` with PyTorch's bin edges.
    static func adaptiveAveragePool(_ x: MLXArray, outputSize: Int) -> MLXArray {
        let rows = poolMatrix(inputSize: x.dim(0), outputSize: outputSize).asType(x.dtype)
        let cols = poolMatrix(inputSize: x.dim(1), outputSize: outputSize).asType(x.dtype)
        let channelsFirst = x.transposed(2, 0, 1)
        let pooledRows = MLX.matmul(rows, channelsFirst)
        let pooled = MLX.matmul(pooledRows, cols.transposed(1, 0))
        return pooled.transposed(1, 2, 0)
    }

    static func poolMatrix(inputSize: Int, outputSize: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: outputSize * inputSize)
        for o in 0 ..< outputSize {
            let start = (o * inputSize) / outputSize
            let end = ((o + 1) * inputSize + outputSize - 1) / outputSize
            let weight = 1 / Float(end - start)
            for i in start ..< end {
                values[o * inputSize + i] = weight
            }
        }
        return MLXArray(values).reshaped(outputSize, inputSize)
    }
}
