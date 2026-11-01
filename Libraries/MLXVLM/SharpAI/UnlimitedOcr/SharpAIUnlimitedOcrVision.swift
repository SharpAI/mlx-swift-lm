// SharpAI fork overlay: unlimited_ocr vision encoders (SAM ViT-B, CLIP-L) and projector.
// Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/baidu/Unlimited-OCR/raw/main/deepencoder.py
//   build_sam_vit_b / ImageEncoderViT / Block / Attention (rel-pos, window partition) / PatchEmbed,
//   build_clip_l / VitModel / CLIPVisionEmbeddings / NoTPTransformer (quick_gelu, pre-LN),
//   MlpProjector (projector_type linear).
// Key layout verified against the tensor names in model.safetensors.index.json (model.sam_model.*,
// model.vision_model.*, model.projector.layers.*).
//
// MLX layout is NHWC. The global view is 1024 x 1024; SAM yields a 16 x 16 grid, which is exactly
// the CLIP-L position grid (224 / 14), so no position interpolation is needed. Other grid sizes
// (the 640 px crop path) need antialiased bicubic interpolation of the absolute position
// embeddings, which is not ported; those inputs throw SharpAIUnlimitedOcrError.unsupportedInputSize.
// Relative-position tables are linearly resized when their length differs, as get_rel_pos does.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAIUnlimitedOcrVisionSizes {
    static let samPatchSize = 16
    static let samGrid = 64  // 1024 / 16, the pos_embed grid
    static let samEmbedDim = 768
    static let samDepth = 12
    static let samHeads = 12
    static let samWindow = 14
    static let samGlobalIndexes: Set<Int> = [2, 5, 8, 11]
    static let samOutChannels = 256
    static let clipWidth = 1024
    static let clipLayers = 24
    static let clipHeads = 16
    static let clipFFN = 4096
    static let clipPatchSize = 14
    static let clipGrid = 16  // 224 / 14
    static let projectorInputDim = 2048  // CLIP-L 1024 + SAM 1024
}

// MARK: - Helpers

/// Linear interpolation weights (outSize x inSize) matching torch F.interpolate(mode="linear",
/// align_corners=False): source = scale * (dst + 0.5) - 0.5, clamped at 0.
func sharpAIUnlimitedOcrLinearResizeWeights(inSize: Int, outSize: Int) -> MLXArray {
    var w = [Float](repeating: 0, count: outSize * inSize)
    let scale = Float(inSize) / Float(outSize)
    for o in 0 ..< outSize {
        let src = max((Float(o) + 0.5) * scale - 0.5, 0)
        let i0 = min(Int(src), inSize - 1)
        let i1 = min(i0 + 1, inSize - 1)
        let lambda = src - Float(i0)
        w[o * inSize + i0] += 1 - lambda
        w[o * inSize + i1] += lambda
    }
    return MLXArray(w, [outSize, inSize])
}

/// Partitions (B, H, W, C) into non-overlapping windows, zero-padding H and W to multiples of
/// `windowSize`. Returns (B * numWindows, ws, ws, C) and the padded size.
func sharpAIUnlimitedOcrWindowPartition(_ x: MLXArray, _ windowSize: Int) -> (
    windows: MLXArray, padded: (Int, Int)
) {
    let B = x.dim(0)
    let H = x.dim(1)
    let W = x.dim(2)
    let C = x.dim(3)
    let padH = (windowSize - H % windowSize) % windowSize
    let padW = (windowSize - W % windowSize) % windowSize
    var y = x
    if padH > 0 || padW > 0 {
        y = padded(y, widths: [0, .init((0, padH)), .init((0, padW)), 0])
    }
    let Hp = H + padH
    let Wp = W + padW
    let windows =
        y
        .reshaped(B, Hp / windowSize, windowSize, Wp / windowSize, windowSize, C)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(-1, windowSize, windowSize, C)
    return (windows, (Hp, Wp))
}

/// Inverse of `sharpAIUnlimitedOcrWindowPartition`, cropping the padding back to (H, W).
func sharpAIUnlimitedOcrWindowUnpartition(
    _ windows: MLXArray, _ windowSize: Int, padded: (Int, Int), original: (Int, Int)
) -> MLXArray {
    let (Hp, Wp) = padded
    let (H, W) = original
    let C = windows.dim(3)
    let B = windows.dim(0) / ((Hp / windowSize) * (Wp / windowSize))
    var y =
        windows
        .reshaped(B, Hp / windowSize, Wp / windowSize, windowSize, windowSize, C)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(B, Hp, Wp, C)
    if Hp > H || Wp > W {
        y = y[0..., 0 ..< H, 0 ..< W, 0...]
    }
    return y
}

func sharpAIUnlimitedOcrQuickGelu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x * 1.702)
}

// MARK: - SAM ViT-B

final class SharpAIUnlimitedOcrSAMAttention: Module {
    let numHeads: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray

    init(dim: Int, numHeads: Int, inputSize: Int) {
        self.numHeads = numHeads
        let headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        _proj.wrappedValue = Linear(dim, dim, bias: true)
        _relPosH.wrappedValue = zeros([2 * inputSize - 1, headDim])
        _relPosW.wrappedValue = zeros([2 * inputSize - 1, headDim])
    }

    /// get_rel_pos: a (qSize, kSize, C) table indexed by the relative coordinate of each pair.
    private func relativeTable(_ relPos: MLXArray, qSize: Int, kSize: Int) -> MLXArray {
        let maxRel = 2 * max(qSize, kSize) - 1
        var table = relPos
        if relPos.dim(0) != maxRel {
            let weights = sharpAIUnlimitedOcrLinearResizeWeights(
                inSize: relPos.dim(0), outSize: maxRel)
            table = weights.asType(relPos.dtype).matmul(relPos)
        }
        let qScale = max(Float(kSize) / Float(qSize), 1.0)
        let kScale = max(Float(qSize) / Float(kSize), 1.0)
        var indices = [Int32]()
        indices.reserveCapacity(qSize * kSize)
        for i in 0 ..< qSize {
            for j in 0 ..< kSize {
                let rel = Float(i) * qScale - Float(j) * kScale + Float(kSize - 1) * kScale
                indices.append(Int32(rel))
            }
        }
        return take(table, MLXArray(indices), axis: 0).reshaped(qSize, kSize, -1)
    }

    /// add_decomposed_rel_pos: an additive (B*heads, H*W, H*W) attention bias.
    private func relativeBias(q: MLXArray, height: Int, width: Int) -> MLXArray {
        let bh = q.dim(0)
        let rQ = q.reshaped(bh, height, width, -1)
        let rH = relativeTable(relPosH, qSize: height, kSize: height)
        let rW = relativeTable(relPosW, qSize: width, kSize: width)

        // relH[b, h, w, k] = sum_c rQ[b, h, w, c] * rH[h, k, c]
        let relH = matmul(
            rQ.transposed(1, 0, 2, 3).reshaped(height, bh * width, -1),
            rH.transposed(0, 2, 1)
        )
        .reshaped(height, bh, width, height)
        .transposed(1, 0, 2, 3)

        // relW[b, h, w, l] = sum_c rQ[b, h, w, c] * rW[w, l, c]
        let relW = matmul(
            rQ.transposed(2, 0, 1, 3).reshaped(width, bh * height, -1),
            rW.transposed(0, 2, 1)
        )
        .reshaped(width, bh, height, width)
        .transposed(1, 2, 0, 3)

        let bias =
            relH.reshaped(bh, height, width, height, 1)
            + relW.reshaped(bh, height, width, 1, width)
        return bias.reshaped(bh, height * width, height * width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)
        let C = x.dim(3)
        let headDim = C / numHeads

        let qkvOut = qkv(x).reshaped(B, H * W, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        let bias = relativeBias(q: q.reshaped(B * numHeads, H * W, headDim), height: H, width: W)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: .array(bias.reshaped(B, numHeads, H * W, H * W).asType(q.dtype)))
        return proj(out.transposed(0, 2, 1, 3).reshaped(B, H, W, C))
    }
}

final class SharpAIUnlimitedOcrSAMMLP: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(dim: Int, hiddenDim: Int) {
        _lin1.wrappedValue = Linear(dim, hiddenDim, bias: true)
        _lin2.wrappedValue = Linear(hiddenDim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(gelu(lin1(x)))
    }
}

final class SharpAIUnlimitedOcrSAMBlock: Module {
    let windowSize: Int

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SharpAIUnlimitedOcrSAMAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIUnlimitedOcrSAMMLP

    /// windowSize 0 is global attention over the whole grid (`gridSize`).
    init(dim: Int, numHeads: Int, windowSize: Int, gridSize: Int) {
        self.windowSize = windowSize
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _attn.wrappedValue = SharpAIUnlimitedOcrSAMAttention(
            dim: dim, numHeads: numHeads, inputSize: windowSize > 0 ? windowSize : gridSize)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _mlp.wrappedValue = SharpAIUnlimitedOcrSAMMLP(dim: dim, hiddenDim: dim * 4)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var h = norm1(x)
        let H = h.dim(1)
        let W = h.dim(2)
        var padded = (H, W)
        if windowSize > 0 {
            let partitioned = sharpAIUnlimitedOcrWindowPartition(h, windowSize)
            h = partitioned.windows
            padded = partitioned.padded
        }
        h = attn(h)
        if windowSize > 0 {
            h = sharpAIUnlimitedOcrWindowUnpartition(
                h, windowSize, padded: padded, original: (H, W))
        }
        let r = shortcut + h
        return r + mlp(norm2(r))
    }
}

final class SharpAIUnlimitedOcrSAMPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(embedDim: Int) {
        _proj.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: embedDim,
            kernelSize: .init(SharpAIUnlimitedOcrVisionSizes.samPatchSize),
            stride: .init(SharpAIUnlimitedOcrVisionSizes.samPatchSize), bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(x)
    }
}

/// Neck, indexed as in the checkpoint: 0 = 1x1 conv, 1 = LayerNorm2d, 2 = 3x3 conv, 3 = LayerNorm2d.
final class SharpAIUnlimitedOcrSAMNeck: Module {
    @ModuleInfo(key: "0") var conv0: Conv2d
    @ModuleInfo(key: "1") var norm1: LayerNorm
    @ModuleInfo(key: "2") var conv2: Conv2d
    @ModuleInfo(key: "3") var norm3: LayerNorm

    init(inputDim: Int, outChannels: Int) {
        let out = outChannels
        _conv0.wrappedValue = Conv2d(
            inputChannels: inputDim, outputChannels: out,
            kernelSize: .init(1), bias: false)
        _norm1.wrappedValue = LayerNorm(dimensions: out, eps: 1e-6)
        _conv2.wrappedValue = Conv2d(
            inputChannels: out, outputChannels: out, kernelSize: .init(3), padding: .init(1),
            bias: false)
        _norm3.wrappedValue = LayerNorm(dimensions: out, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm3(conv2(norm1(conv0(x))))
    }
}

/// SAM ViT-B image encoder: (B, 1024, 1024, 3) -> (B, 16, 16, 1024).
final class SharpAIUnlimitedOcrSAMEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: SharpAIUnlimitedOcrSAMPatchEmbed
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SharpAIUnlimitedOcrSAMBlock]
    @ModuleInfo(key: "neck") var neck: SharpAIUnlimitedOcrSAMNeck
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    /// `gridSize` is the pos_embed grid (64 for the 1024 px checkpoint).
    init(gridSize: Int = SharpAIUnlimitedOcrVisionSizes.samGrid) {
        let grid = gridSize
        let embedDim = SharpAIUnlimitedOcrVisionSizes.samEmbedDim
        _patchEmbed.wrappedValue = SharpAIUnlimitedOcrSAMPatchEmbed(embedDim: embedDim)
        _posEmbed.wrappedValue = zeros([1, grid, grid, embedDim])
        let globalIndexes = SharpAIUnlimitedOcrVisionSizes.samGlobalIndexes
        let window = SharpAIUnlimitedOcrVisionSizes.samWindow
        _blocks.wrappedValue = (0 ..< SharpAIUnlimitedOcrVisionSizes.samDepth).map { i in
            let windowSize = globalIndexes.contains(i) ? 0 : window
            return SharpAIUnlimitedOcrSAMBlock(
                dim: embedDim, numHeads: SharpAIUnlimitedOcrVisionSizes.samHeads,
                windowSize: windowSize, gridSize: grid)
        }
        _neck.wrappedValue = SharpAIUnlimitedOcrSAMNeck(
            inputDim: embedDim, outChannels: SharpAIUnlimitedOcrVisionSizes.samOutChannels)
        _net2.wrappedValue = Conv2d(
            inputChannels: SharpAIUnlimitedOcrVisionSizes.samOutChannels, outputChannels: 512,
            kernelSize: .init(3), stride: .init(2), padding: .init(1), bias: false)
        _net3.wrappedValue = Conv2d(
            inputChannels: 512, outputChannels: 1024, kernelSize: .init(3),
            stride: .init(2), padding: .init(1), bias: false)
        super.init()
    }

    func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
        var x = patchEmbed(pixels)
        guard x.dim(1) == posEmbed.dim(1), x.dim(2) == posEmbed.dim(2) else {
            throw SharpAIUnlimitedOcrError.unsupportedInputSize(
                "SAM grid \(x.dim(1))x\(x.dim(2)), pos_embed grid \(posEmbed.dim(1))")
        }
        x = x + posEmbed
        for block in blocks {
            x = block(x)
        }
        x = neck(x)
        return net3(net2(x))
    }
}

// MARK: - CLIP-L

final class SharpAIUnlimitedOcrCLIPAttention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(width: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = width / numHeads
        _qkvProj.wrappedValue = Linear(width, width * 3, bias: true)
        _outProj.wrappedValue = Linear(width, width, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let qkv = qkvProj(x).reshaped(B, L, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let out = MLXFast.scaledDotProductAttention(
            queries: qkv[0], keys: qkv[1], values: qkv[2],
            scale: pow(Float(headDim), -0.5), mask: .none)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class SharpAIUnlimitedOcrCLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(width: Int, ffn: Int) {
        _fc1.wrappedValue = Linear(width, ffn, bias: true)
        _fc2.wrappedValue = Linear(ffn, width, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(sharpAIUnlimitedOcrQuickGelu(fc1(x)))
    }
}

final class SharpAIUnlimitedOcrCLIPLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIUnlimitedOcrCLIPAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIUnlimitedOcrCLIPMLP

    init(width: Int, heads: Int, ffn: Int) {
        _layerNorm1.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _selfAttn.wrappedValue = SharpAIUnlimitedOcrCLIPAttention(width: width, numHeads: heads)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _mlp.wrappedValue = SharpAIUnlimitedOcrCLIPMLP(width: width, ffn: ffn)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

final class SharpAIUnlimitedOcrCLIPTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [SharpAIUnlimitedOcrCLIPLayer]

    init(width: Int, depth: Int, heads: Int, ffn: Int) {
        _layers.wrappedValue = (0 ..< depth).map { _ in
            SharpAIUnlimitedOcrCLIPLayer(width: width, heads: heads, ffn: ffn)
        }
    }
}

final class SharpAIUnlimitedOcrCLIPEmbeddings: Module {
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    /// Present so the checkpoint loads. The forward pass takes SAM features in its place.
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(width: Int, patchSize: Int, numPositions: Int) {
        _classEmbedding.wrappedValue = zeros([width])
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: width, kernelSize: .init(patchSize),
            stride: .init(patchSize), bias: false)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPositions, dimensions: width)
    }
}

/// CLIP-L vision tower. Its patch embeddings are the SAM feature map (B, 16, 16, 1024), as in
/// VitModel.forward(x, patch_embeds=local_features_1). Returns (B, 1 + 256, 1024) tokens.
final class SharpAIUnlimitedOcrCLIPVision: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SharpAIUnlimitedOcrCLIPEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: SharpAIUnlimitedOcrCLIPTransformer

    let numPatches: Int

    init(gridSize: Int = SharpAIUnlimitedOcrVisionSizes.clipGrid) {
        let width = SharpAIUnlimitedOcrVisionSizes.clipWidth
        let grid = gridSize
        self.numPatches = grid * grid
        _embeddings.wrappedValue = SharpAIUnlimitedOcrCLIPEmbeddings(
            width: width, patchSize: SharpAIUnlimitedOcrVisionSizes.clipPatchSize,
            numPositions: grid * grid + 1)
        _preLayerNorm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _transformer.wrappedValue = SharpAIUnlimitedOcrCLIPTransformer(
            width: width, depth: SharpAIUnlimitedOcrVisionSizes.clipLayers,
            heads: SharpAIUnlimitedOcrVisionSizes.clipHeads,
            ffn: SharpAIUnlimitedOcrVisionSizes.clipFFN)
        super.init()
    }

    func callAsFunction(_ samFeatures: MLXArray) throws -> MLXArray {
        let B = samFeatures.dim(0)
        let h = samFeatures.dim(1)
        let w = samFeatures.dim(2)
        let C = samFeatures.dim(3)
        guard h * w == numPatches else {
            throw SharpAIUnlimitedOcrError.unsupportedInputSize(
                "CLIP position grid expects \(numPatches) patches, got \(h * w)")
        }
        let patches = samFeatures.reshaped(B, h * w, C)
        let cls = repeated(
            embeddings.classEmbedding.reshaped(1, 1, -1), count: B, axis: 0)
        var x = concatenated([cls, patches], axis: 1)
        let positionIds = MLXArray(0 ..< (numPatches + 1))[.newAxis, 0...]
        x = x + embeddings.positionEmbedding(positionIds)
        x = preLayerNorm(x)
        for layer in transformer.layers {
            x = layer(x)
        }
        return x
    }
}

// MARK: - Projector

/// MlpProjector with projector_type "linear": 2048 (CLIP-L 1024 + SAM 1024) to the text hidden size.
final class SharpAIUnlimitedOcrProjector: Module {
    @ModuleInfo(key: "layers") var layers: Linear

    init(outputDim: Int) {
        _layers.wrappedValue = Linear(
            SharpAIUnlimitedOcrVisionSizes.projectorInputDim, outputDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers(x)
    }
}
