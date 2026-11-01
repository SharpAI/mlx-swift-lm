// SharpAI fork overlay: NemotronH_Nano_VL_V2 vision tower and projector. Not part of upstream.
//
// Ported from the reference sources (read in full):
//  - nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 modeling.py: extract_feature, pixel_shuffle
//    (ps_version v2), mlp1 = RMSNorm -> Linear(no bias) -> SquaredReLU -> Linear(no bias).
//  - nvidia/C-RADIOv2-H radio_model.py: forward_features = patch_generator -> blocks -> norm,
//    norm is Identity for this checkpoint; features are y[:, num_skip:].
//  - nvidia/C-RADIOv2-H vit_patch_generator.py: Im2Patches order (c, yy, xx), patch embedder
//    without bias, eval-mode CPE position embedding (bilinear, align_corners=True, to the
//    square of max(H, W), then window select).
//  - nvidia/C-RADIOv2-H cls_token.py: num_registers = register_multiple - (num_tokens % register_multiple).
//  - nvidia/C-RADIOv2-H enable_cpe_support.py: cls and register tokens are prepended after the
//    position embedding is added.
//  - timm VisionTransformer as built by vit_huge_patch16_224: pre-norm blocks, LayerNorm eps 1e-6,
//    GELU MLP, fused qkv with bias.
//
// Input pixel values are NCHW and already mean/std normalized (norm_mean / norm_std in config.json).
// The reference moves that conditioner out of the tower, so it is not applied here.
// UNVERIFIED: the checkpoint loader. The overlay sanitizer still drops vision keys, so these
// modules run with whatever weights the caller sets, not with the released checkpoint.

import Foundation
import MLX
import MLXNN

/// Shape of the RADIO ViT and the pixel-shuffle projector input.
public struct SharpAINemotronhNanoVlV2VisionConfiguration: Sendable {
    public let patchSize: Int
    public let embedDim: Int
    public let depth: Int
    public let numHeads: Int
    public let mlpHiddenSize: Int
    /// Side of the square learned position grid, in pixels (`cpe_max_size`).
    public let cpeMaxSize: Int
    /// `num_cls_tokens`: one per distillation teacher when `cls_token_per_teacher` is set.
    public let numClsTokens: Int
    /// `register_multiple` from the vision args. Nil means no register tokens.
    public let registerMultiple: Int?
    public let layerNormEps: Float

    /// vit_huge_patch16_224 as used by C-RADIOv2-H (radio_v2.5-h) in the released VL checkpoint.
    public static let radioV25H = SharpAINemotronhNanoVlV2VisionConfiguration(
        patchSize: 16, embedDim: 1280, depth: 32, numHeads: 16, mlpHiddenSize: 5120,
        cpeMaxSize: 2048, numClsTokens: 4, registerMultiple: 16, layerNormEps: 1e-6)

    public init(
        patchSize: Int, embedDim: Int, depth: Int, numHeads: Int, mlpHiddenSize: Int,
        cpeMaxSize: Int, numClsTokens: Int, registerMultiple: Int?, layerNormEps: Float
    ) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.depth = depth
        self.numHeads = numHeads
        self.mlpHiddenSize = mlpHiddenSize
        self.cpeMaxSize = cpeMaxSize
        self.numClsTokens = numClsTokens
        self.registerMultiple = registerMultiple
        self.layerNormEps = layerNormEps
    }

    /// Number of leading tokens that carry no spatial feature (cls plus registers).
    public var numSkipTokens: Int {
        var registers = 0
        if let registerMultiple {
            registers = registerMultiple - (numClsTokens % registerMultiple)
        }
        return numClsTokens + registers
    }
}

// MARK: - Patch generator

final class SharpAINemotronhNanoVlV2ClsToken: Module {
    @ParameterInfo(key: "token") var token: MLXArray

    init(tokens: Int, dim: Int) {
        self._token.wrappedValue = MLXArray.zeros([tokens, dim])
        super.init()
    }
}

final class SharpAINemotronhNanoVlV2PatchGenerator: Module {
    let patchSize: Int
    let numRows: Int
    let numCols: Int
    let numSkip: Int
    let embedDim: Int

    @ModuleInfo(key: "embedder") var embedder: Linear
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "cls_token") var clsToken: SharpAINemotronhNanoVlV2ClsToken

    init(_ config: SharpAINemotronhNanoVlV2VisionConfiguration) {
        patchSize = config.patchSize
        embedDim = config.embedDim
        let gridSide = (config.cpeMaxSize + config.patchSize - 1) / config.patchSize
        numRows = gridSide
        numCols = gridSide
        numSkip = config.numSkipTokens

        self._embedder.wrappedValue = Linear(
            3 * config.patchSize * config.patchSize, config.embedDim, bias: false)
        self._posEmbed.wrappedValue = MLXArray.zeros([1, gridSide * gridSide, config.embedDim])
        self._clsToken.wrappedValue = SharpAINemotronhNanoVlV2ClsToken(
            tokens: config.numSkipTokens, dim: config.embedDim)
        super.init()
    }

    /// NCHW pixels to (N, numSkip + H/p * W/p, C) tokens.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let n = pixels.dim(0)
        let channels = pixels.dim(1)
        let gridH = pixels.dim(2) / patchSize
        let gridW = pixels.dim(3) / patchSize

        // Im2Patches: 'b c (py yy) (px xx) -> b (py px) (c yy xx)'.
        var patches = pixels.reshaped(n, channels, gridH, patchSize, gridW, patchSize)
            .transposed(0, 2, 4, 1, 3, 5)
            .reshaped(n, gridH * gridW, channels * patchSize * patchSize)
        patches = embedder(patches) + positionEmbedding(gridH: gridH, gridW: gridW)

        let cls = broadcast(
            clsToken.token.expandedDimensions(axis: 0), to: [n, numSkip, embedDim])
        return concatenated([cls, patches], axis: 1)
    }

    /// Eval-mode CPE: bilinear (align_corners) to a square of max(H, W), then crop to H x W.
    private func positionEmbedding(gridH: Int, gridW: Int) -> MLXArray {
        var grid = posEmbed.reshaped(1, numRows, numCols, embedDim)
        let side = max(gridH, gridW)
        grid = resizeBilinearAlignCorners(grid, height: side, width: side)
        if gridH < side || gridW < side {
            grid = grid[0..., 0 ..< gridH, 0 ..< gridW, 0...]
        }
        return grid.reshaped(1, gridH * gridW, embedDim)
    }
}

/// Bilinear resize of an NHWC tensor with align_corners=True, as torch F.interpolate.
func resizeBilinearAlignCorners(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
    resizeAxisAlignCorners(resizeAxisAlignCorners(x, axis: 1, size: height), axis: 2, size: width)
}

private func resizeAxisAlignCorners(_ x: MLXArray, axis: Int, size: Int) -> MLXArray {
    let inSize = x.dim(axis)
    if inSize == size {
        return x
    }

    var low = [Int32]()
    var high = [Int32]()
    var fraction = [Float]()
    for out in 0 ..< size {
        let src: Float = size == 1 ? 0 : Float(out) * Float(inSize - 1) / Float(size - 1)
        let base = min(Int(src), inSize - 1)
        low.append(Int32(base))
        high.append(Int32(min(base + 1, inSize - 1)))
        fraction.append(src - Float(base))
    }

    var weightShape = [Int](repeating: 1, count: x.ndim)
    weightShape[axis] = size
    let weight = MLXArray(fraction, weightShape).asType(x.dtype)
    let a = x.take(MLXArray(low), axis: axis)
    let b = x.take(MLXArray(high), axis: axis)
    return a * (MLXArray(Float(1)) - weight) + b * weight
}

// MARK: - Pre-norm ViT block (timm Block with no layer scale and no drop path)

final class SharpAINemotronhNanoVlV2Attention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(dim / numHeads), -0.5)
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._proj.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let parts = qkv(x).reshaped(batch, length, 3, numHeads, headDim).split(parts: 3, axis: 2)
        let q = parts[0].reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = parts[1].reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        return proj(output.transposed(0, 2, 1, 3).reshaped(batch, length, numHeads * headDim))
    }
}

final class SharpAINemotronhNanoVlV2MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dim: Int, hidden: Int) {
        self._fc1.wrappedValue = Linear(dim, hidden, bias: true)
        self._fc2.wrappedValue = Linear(hidden, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

final class SharpAINemotronhNanoVlV2Block: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SharpAINemotronhNanoVlV2Attention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAINemotronhNanoVlV2MLP

    init(_ config: SharpAINemotronhNanoVlV2VisionConfiguration) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: config.layerNormEps)
        self._attn.wrappedValue = SharpAINemotronhNanoVlV2Attention(
            dim: config.embedDim, numHeads: config.numHeads)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: config.layerNormEps)
        self._mlp.wrappedValue = SharpAINemotronhNanoVlV2MLP(
            dim: config.embedDim, hidden: config.mlpHiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x + attn(norm1(x))
        hidden = hidden + mlp(norm2(hidden))
        return hidden
    }
}

final class SharpAINemotronhNanoVlV2ViT: Module {
    @ModuleInfo(key: "patch_generator") var patchGenerator: SharpAINemotronhNanoVlV2PatchGenerator
    @ModuleInfo(key: "blocks") var blocks: [SharpAINemotronhNanoVlV2Block]

    init(_ config: SharpAINemotronhNanoVlV2VisionConfiguration) {
        self._patchGenerator.wrappedValue = SharpAINemotronhNanoVlV2PatchGenerator(config)
        self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
            SharpAINemotronhNanoVlV2Block(config)
        }
        super.init()
    }

    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var hidden = patchGenerator(pixels)
        for block in blocks {
            hidden = block(hidden)
        }
        // The RADIO final norm is Identity for this checkpoint (create_model_from_args).
        return hidden
    }
}

final class SharpAINemotronhNanoVlV2RADIOModel: Module {
    @ModuleInfo(key: "model") var model: SharpAINemotronhNanoVlV2ViT

    init(_ config: SharpAINemotronhNanoVlV2VisionConfiguration) {
        self._model.wrappedValue = SharpAINemotronhNanoVlV2ViT(config)
        super.init()
    }
}

// MARK: - Vision model and projector

/// The `vision_model` subtree: RADIO ViT with the summary tokens removed.
public final class SharpAINemotronhNanoVlV2VisionModel: Module {
    public let configuration: SharpAINemotronhNanoVlV2VisionConfiguration

    @ModuleInfo(key: "radio_model") var radioModel: SharpAINemotronhNanoVlV2RADIOModel

    public init(_ configuration: SharpAINemotronhNanoVlV2VisionConfiguration) {
        self.configuration = configuration
        self._radioModel.wrappedValue = SharpAINemotronhNanoVlV2RADIOModel(configuration)
        super.init()
    }

    /// NCHW normalized pixels to (N, (H/p) * (W/p), embedDim) spatial features.
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let output = radioModel.model(pixels)
        return output[0..., configuration.numSkipTokens..., 0...]
    }
}

/// `mlp1`: RMSNorm, Linear (no bias), squared ReLU, Linear (no bias).
public final class SharpAINemotronhNanoVlV2Projector: Module {
    @ModuleInfo(key: "0") var norm: RMSNorm
    @ModuleInfo(key: "1") var fc1: Linear
    @ModuleInfo(key: "3") var fc2: Linear

    public init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        self._norm.wrappedValue = RMSNorm(dimensions: inputSize, eps: 1e-5)
        self._fc1.wrappedValue = Linear(inputSize, hiddenSize, bias: false)
        self._fc2.wrappedValue = Linear(hiddenSize, outputSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = MLX.maximum(fc1(norm(x)), MLXArray(Float(0)))
        return fc2(activated * activated)
    }
}

/// Pixel shuffle with ps_version "v2" and downsample_ratio 0.5.
/// NHWC (n, h, w, c) to (n, (h/2) * (w/2), 4c). Token order is row-major over the half grid,
/// and each token concatenates (row offset, column offset, channel), as the reference permutes do.
func sharpAINemotronhNanoVlV2PixelShuffle(_ x: MLXArray) -> MLXArray {
    let n = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)
    return x.reshaped(n, h / 2, 2, w / 2, 2, c)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(n, (h / 2) * (w / 2), 4 * c)
}
