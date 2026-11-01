// SharpAI fork overlay: frozen DINOv2 + SigLIP feature extractors for OpenVLA. Not part of upstream
// mlx-swift-lm.
//
// Module paths and parameter names follow timm (`vit_large_patch14_reg4_dinov2.lvd142m` and
// `vit_so400m_patch14_siglip_224`), which the OpenVLA checkpoint uses under
// `vision_backbone.featurizer` and `vision_backbone.fused_featurizer`.
// Each encoder runs all blocks but the last and returns the penultimate block output, as the
// OpenVLA reference does. UNVERIFIED: that choice is recalled from the reference code, not read
// from this repo.

import Foundation
import MLX
import MLXNN

/// Shape of one timm ViT used as a feature extractor.
struct SharpAIOpenvlaViTSpec: Sendable {
    let hiddenSize: Int
    let depth: Int
    let heads: Int
    let mlpHiddenSize: Int
    let patchSize: Int
    let imageSize: Int
    /// DINOv2 register tokens. Zero for SigLIP.
    let registerTokens: Int
    /// DINOv2 has a CLS token. SigLIP has none.
    let hasClassToken: Bool
    /// DINOv2 scales each residual branch by a learned LayerScale vector.
    let layerScale: Bool
    /// SigLIP's MLP uses the tanh GELU approximation. UNVERIFIED against timm.
    let approximateGelu: Bool

    var gridSize: Int { imageSize / patchSize }
    var patchCount: Int { gridSize * gridSize }
    var layerNormEps: Float { 1e-6 }
}

/// Both backbones in the fused OpenVLA vision stack.
struct SharpAIOpenvlaVisionSpec: Sendable {
    let dino: SharpAIOpenvlaViTSpec
    let siglip: SharpAIOpenvlaViTSpec

    /// Width of the fused features that feed the projector.
    var fusedHiddenSize: Int { dino.hiddenSize + siglip.hiddenSize }

    /// ViT-L/14 with 4 registers (DINOv2) and So400m/14 (SigLIP), both at 224 px.
    static let openvla7b = SharpAIOpenvlaVisionSpec(
        dino: SharpAIOpenvlaViTSpec(
            hiddenSize: 1024, depth: 24, heads: 16, mlpHiddenSize: 4096, patchSize: 14,
            imageSize: 224, registerTokens: 4, hasClassToken: true, layerScale: true,
            approximateGelu: false),
        siglip: SharpAIOpenvlaViTSpec(
            hiddenSize: 1152, depth: 27, heads: 16, mlpHiddenSize: 4304, patchSize: 14,
            imageSize: 224, registerTokens: 0, hasClassToken: false, layerScale: false,
            approximateGelu: true)
    )
}

/// Patch embedding. Module path `patch_embed.proj.*`. Input is NHWC.
final class SharpAIOpenvlaPatchEmbed: Module {
    @ModuleInfo var proj: Conv2d

    init(_ spec: SharpAIOpenvlaViTSpec) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: spec.hiddenSize,
            kernelSize: .init(spec.patchSize), stride: .init(spec.patchSize))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(x)
    }
}

/// Multi-head self-attention with a fused `qkv` projection. Module path `attn.*`.
final class SharpAIOpenvlaViTAttention: Module {
    let numHeads: Int
    let scale: Float

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    init(hiddenSize: Int, heads: Int) {
        precondition(hiddenSize % heads == 0, "hidden size must be divisible by heads")
        self.numHeads = heads
        self.scale = pow(Float(hiddenSize / heads), -0.5)
        self._qkv.wrappedValue = Linear(hiddenSize, 3 * hiddenSize, bias: true)
        self._proj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let parts = qkv(x).split(parts: 3, axis: -1)
        let queries = parts[0].reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let keys = parts[1].reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return proj(output)
    }
}

/// Two-layer MLP. Module path `mlp.fc1` and `mlp.fc2`.
final class SharpAIOpenvlaViTMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let approximateGelu: Bool

    init(hiddenSize: Int, mlpHiddenSize: Int, approximateGelu: Bool) {
        self.approximateGelu = approximateGelu
        self._fc1.wrappedValue = Linear(hiddenSize, mlpHiddenSize, bias: true)
        self._fc2.wrappedValue = Linear(mlpHiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = fc1(x)
        return fc2(approximateGelu ? geluApproximate(hidden) : gelu(hidden))
    }
}

/// Learned per-channel residual scale. Module path `ls1.gamma` or `ls2.gamma`.
final class SharpAIOpenvlaLayerScale: Module {
    @ParameterInfo var gamma: MLXArray

    init(_ dim: Int) {
        self._gamma.wrappedValue = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

/// Pre-norm ViT block. SigLIP blocks have no LayerScale.
final class SharpAIOpenvlaViTBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: SharpAIOpenvlaViTAttention
    @ModuleInfo var ls1: SharpAIOpenvlaLayerScale?
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: SharpAIOpenvlaViTMLP
    @ModuleInfo var ls2: SharpAIOpenvlaLayerScale?

    init(_ spec: SharpAIOpenvlaViTSpec) {
        self._norm1.wrappedValue = LayerNorm(dimensions: spec.hiddenSize, eps: spec.layerNormEps)
        self._attn.wrappedValue = SharpAIOpenvlaViTAttention(
            hiddenSize: spec.hiddenSize, heads: spec.heads)
        self._ls1.wrappedValue =
            spec.layerScale ? SharpAIOpenvlaLayerScale(spec.hiddenSize) : nil
        self._norm2.wrappedValue = LayerNorm(dimensions: spec.hiddenSize, eps: spec.layerNormEps)
        self._mlp.wrappedValue = SharpAIOpenvlaViTMLP(
            hiddenSize: spec.hiddenSize, mlpHiddenSize: spec.mlpHiddenSize,
            approximateGelu: spec.approximateGelu)
        self._ls2.wrappedValue =
            spec.layerScale ? SharpAIOpenvlaLayerScale(spec.hiddenSize) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var r = attn(norm1(x))
        if let ls1 {
            r = ls1(r)
        }
        let h = x + r

        r = mlp(norm2(h))
        if let ls2 {
            r = ls2(r)
        }
        return h + r
    }
}

/// DINOv2 ViT-L/14 with registers. Module path `featurizer.*`.
///
/// Output is the patch tokens of the penultimate block, without the CLS and register tokens.
/// timm's reg4 DINOv2 uses `no_embed_class`: `pos_embed` is [1, patches, hidden] with no CLS slot,
/// it is added to the patch tokens only, and then [CLS, registers] are prepended.
/// The checkpoint's `pos_embed` must already match the 224 px grid. There is no interpolation.
final class SharpAIOpenvlaDinoV2Featurizer: Module {
    let spec: SharpAIOpenvlaViTSpec

    @ModuleInfo(key: "patch_embed") var patchEmbed: SharpAIOpenvlaPatchEmbed
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "reg_token") var regToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    var blocks: [SharpAIOpenvlaViTBlock]

    init(_ spec: SharpAIOpenvlaViTSpec) {
        precondition(spec.hasClassToken, "DINOv2 featurizer needs a CLS token")
        self.spec = spec
        self._patchEmbed.wrappedValue = SharpAIOpenvlaPatchEmbed(spec)
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, spec.hiddenSize])
        self._regToken.wrappedValue = MLXArray.zeros([1, spec.registerTokens, spec.hiddenSize])
        self._posEmbed.wrappedValue = MLXArray.zeros([1, spec.patchCount, spec.hiddenSize])
        self.blocks = (0 ..< spec.depth).map { _ in SharpAIOpenvlaViTBlock(spec) }
    }

    /// NHWC pixels `[batch, H, W, 3]` to patch features `[batch, patches, hiddenSize]`.
    func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
        let batch = pixels.dim(0)
        guard posEmbed.dim(1) == spec.patchCount else {
            throw SharpAIOpenvlaError.visionPositionMismatch(
                expected: spec.patchCount, actual: posEmbed.dim(1))
        }

        let patches = patchEmbed(pixels).reshaped(batch, spec.patchCount, spec.hiddenSize)
        let cls = broadcast(clsToken, to: [batch, 1, spec.hiddenSize])
        let body = patches + posEmbed
        let registers = broadcast(regToken, to: [batch, spec.registerTokens, spec.hiddenSize])

        var h = concatenated([cls, registers, body], axis: 1)
        for block in blocks.dropLast() {
            h = block(h)
        }
        return h[0..., (1 + spec.registerTokens)..., 0...]
    }
}

/// SigLIP So400m/14 without the attention-pooling head. Module path `fused_featurizer.*`.
///
/// Output is every token of the penultimate block. There is no CLS token and no final norm.
final class SharpAIOpenvlaSigLIPFeaturizer: Module {
    let spec: SharpAIOpenvlaViTSpec

    @ModuleInfo(key: "patch_embed") var patchEmbed: SharpAIOpenvlaPatchEmbed
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    var blocks: [SharpAIOpenvlaViTBlock]

    init(_ spec: SharpAIOpenvlaViTSpec) {
        precondition(!spec.hasClassToken, "SigLIP featurizer has no CLS token")
        self.spec = spec
        self._patchEmbed.wrappedValue = SharpAIOpenvlaPatchEmbed(spec)
        self._posEmbed.wrappedValue = MLXArray.zeros([1, spec.patchCount, spec.hiddenSize])
        self.blocks = (0 ..< spec.depth).map { _ in SharpAIOpenvlaViTBlock(spec) }
    }

    /// NHWC pixels `[batch, H, W, 3]` to patch features `[batch, patches, hiddenSize]`.
    func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
        guard posEmbed.dim(1) == spec.patchCount else {
            throw SharpAIOpenvlaError.visionPositionMismatch(
                expected: spec.patchCount, actual: posEmbed.dim(1))
        }

        var h = patchEmbed(pixels).reshaped(pixels.dim(0), spec.patchCount, spec.hiddenSize)
        h = h + posEmbed
        for block in blocks.dropLast() {
            h = block(h)
        }
        return h
    }
}

/// Fused DINOv2 and SigLIP features, concatenated on the channel axis.
/// Module path `vision_backbone.*`.
final class SharpAIOpenvlaVisionBackbone: Module {
    let spec: SharpAIOpenvlaVisionSpec

    @ModuleInfo var featurizer: SharpAIOpenvlaDinoV2Featurizer
    @ModuleInfo(key: "fused_featurizer") var fusedFeaturizer: SharpAIOpenvlaSigLIPFeaturizer

    init(_ spec: SharpAIOpenvlaVisionSpec) {
        precondition(spec.dino.imageSize == spec.siglip.imageSize, "backbones need one image size")
        self.spec = spec
        self._featurizer.wrappedValue = SharpAIOpenvlaDinoV2Featurizer(spec.dino)
        self._fusedFeaturizer.wrappedValue = SharpAIOpenvlaSigLIPFeaturizer(spec.siglip)
    }

    /// NHWC pixels `[batch, H, W, 6]`. Channels 0..<3 feed DINOv2 and 3..<6 feed SigLIP.
    /// Each backbone expects its own normalization, applied by the caller.
    /// Returns `[batch, patches, dino hidden + siglip hidden]`.
    func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
        guard pixels.dim(-1) == 6 else {
            throw SharpAIOpenvlaError.imageChannelMismatch(pixels.dim(-1))
        }
        let size = spec.dino.imageSize
        guard pixels.dim(1) == size, pixels.dim(2) == size else {
            throw SharpAIOpenvlaError.imageSizeMismatch(
                expected: size, height: pixels.dim(1), width: pixels.dim(2))
        }

        let dino = try featurizer(pixels[0..., 0..., 0..., 0 ..< 3])
        let siglip = try fusedFeaturizer(pixels[0..., 0..., 0..., 3 ..< 6])
        return concatenated([dino, siglip], axis: -1)
    }
}
