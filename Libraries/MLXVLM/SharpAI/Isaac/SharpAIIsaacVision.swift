// SharpAI fork overlay: Siglip2 vision tower for isaac. Not part of upstream mlx-swift-lm.
//
// Takes flattened patches `[N, P, C * p * p]` (see SharpAIIsaacImageProcessor). Patch
// embedding is a Linear, not a Conv2d. The position table is a learned `[grid, grid, hidden]`
// parameter (checkpoint key `embeddings.position_embedding`, no `.weight` suffix), used as-is
// for the fixed config grid. Other grid sizes would need resizing, which is not implemented.
// Key layout matches Perceptron/Isaac-0.1 `model.vision_tower.*` (checked against the
// safetensors header). The pixel shuffle and projector live in SharpAIIsaacModel.swift.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIIsaacVisionAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SharpAIIsaacVisionConfiguration) {
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

final class SharpAIIsaacVisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let tanhApproximation: Bool

    init(_ config: SharpAIIsaacVisionConfiguration) {
        tanhApproximation = config.hiddenAct == "gelu_pytorch_tanh"
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = fc1(x)
        return fc2(tanhApproximation ? geluApproximate(hidden) : MLXNN.gelu(hidden))
    }
}

final class SharpAIIsaacVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIIsaacVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    let mlp: SharpAIIsaacVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIIsaacVisionConfiguration) {
        _attention.wrappedValue = SharpAIIsaacVisionAttention(config)
        _layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = SharpAIIsaacVisionMLP(config)
        _layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(layerNorm1(x))
        h = h + mlp(layerNorm2(h))
        return h
    }
}

final class SharpAIIsaacVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    /// `[grid, grid, hidden]`, row-major over patches, matching the patchify order.
    @ParameterInfo(key: "position_embedding") var positionEmbedding: MLXArray

    init(_ config: SharpAIIsaacVisionConfiguration) {
        _patchEmbedding.wrappedValue = Linear(
            config.patchInputSize, config.hiddenSize, bias: true)
        _positionEmbedding.wrappedValue = MLXArray.zeros(
            [config.gridSide, config.gridSide, config.hiddenSize])
    }

    /// `patches` is `[N, P, C * p * p]` with `P == grid * grid`. Returns `[N, P, hidden]`.
    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        let hidden = positionEmbedding.dim(-1)
        return patchEmbedding(patches) + positionEmbedding.reshaped(1, -1, hidden)
    }
}

final class SharpAIIsaacVisionEncoder: Module {
    let layers: [SharpAIIsaacVisionEncoderLayer]

    init(_ config: SharpAIIsaacVisionConfiguration) {
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIIsaacVisionEncoderLayer(config)
        }
    }
}

/// Siglip2 vision tower. Output is `[N, P, hidden]` after `post_layernorm`.
public final class SharpAIIsaacVisionModel: Module {
    let embeddings: SharpAIIsaacVisionEmbeddings
    let encoder: SharpAIIsaacVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    public init(_ config: SharpAIIsaacVisionConfiguration) {
        self.embeddings = SharpAIIsaacVisionEmbeddings(config)
        self.encoder = SharpAIIsaacVisionEncoder(config)
        _postLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    public func callAsFunction(_ patches: MLXArray) -> MLXArray {
        var h = embeddings(patches)
        for layer in encoder.layers {
            h = layer(h)
        }
        return postLayerNorm(h)
    }
}

/// Isaac pixel shuffle on a fixed square grid, the fixed-resolution case of
/// `pixel_shuffle_varlen` in the Isaac modeling code.
///
/// Input `[N, g * g, C]` becomes `[N, (g / s) * (g / s), C * s * s]`. Output token `(a, b)`
/// gathers input patches `(a * s + i, b * s + j)`; the sub-patch `(i, j)` fills channel block
/// `i * s + j`, so the gather order is fixed by the HF code, not by convention.
enum SharpAIIsaacPixelShuffle {
    static func shuffle(_ x: MLXArray, gridSide: Int, scale: Int) -> MLXArray {
        let n = x.dim(0)
        let c = x.dim(-1)
        let outSide = gridSide / scale
        return x.reshaped(n, outSide, scale, outSide, scale, c)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(n, outSide * outSide, scale * scale * c)
    }
}
