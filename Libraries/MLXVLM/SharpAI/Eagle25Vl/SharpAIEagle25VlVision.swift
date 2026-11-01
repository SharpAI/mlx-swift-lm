// SharpAI fork overlay: SigLIP vision tower for eagle_2_5_vl. Not part of upstream mlx-swift-lm.
//
// The PaliGemma and Idefics3 SigLIP blocks are file-private, so this file has its own copy.
// Follows vLLM's siglip.py: no CLS token, learned absolute positions, no pooling head, and
// `post_layernorm` only when every encoder layer is kept.
// UNVERIFIED: the `vision_model.vision_model.*` nesting comes from vLLM's module tree, not a
// checkpoint index.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIEagle25VlVisionAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: SharpAIEagle25VlVisionConfiguration) {
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

final class SharpAIEagle25VlVisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let tanhApproximation: Bool

    init(_ config: SharpAIEagle25VlVisionConfiguration) {
        tanhApproximation = config.hiddenAct == "gelu_pytorch_tanh"
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = fc1(x)
        return fc2(tanhApproximation ? geluApproximate(hidden) : MLXNN.gelu(hidden))
    }
}

final class SharpAIEagle25VlVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIEagle25VlVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    let mlp: SharpAIEagle25VlVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIEagle25VlVisionConfiguration) {
        _attention.wrappedValue = SharpAIEagle25VlVisionAttention(config)
        _layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = SharpAIEagle25VlVisionMLP(config)
        _layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(layerNorm1(x))
        h = h + mlp(layerNorm2(h))
        return h
    }
}

final class SharpAIEagle25VlVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    // Underscore prefix keeps this out of the module parameter tree, so `verify: .all` does not
    // expect a `position_ids` checkpoint key (matches Paligemma's `_positionIds`).
    let _positionIds: MLXArray

    init(_ config: SharpAIEagle25VlVisionConfiguration) {
        let grid = config.imageSize / config.patchSize
        let positions = grid * grid
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.channels, outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize), stride: .init(config.patchSize)
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: positions, dimensions: config.hiddenSize)
        self._positionIds = MLXArray((0 ..< positions).map { Int32($0) })
    }

    /// `pixels` is channels-last `[N, H, W, C]`. Returns `[N, P, hidden]`.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let patches = patchEmbedding(pixels)
        let flat = patches.reshaped(patches.dim(0), -1, patches.dim(-1))
        return flat + positionEmbedding(_positionIds)
    }
}

final class SharpAIEagle25VlVisionEncoder: Module {
    let layers: [SharpAIEagle25VlVisionEncoderLayer]

    init(_ config: SharpAIEagle25VlVisionConfiguration, layerCount: Int) {
        self.layers = (0 ..< layerCount).map { _ in
            SharpAIEagle25VlVisionEncoderLayer(config)
        }
    }
}

final class SharpAIEagle25VlVisionTransformer: Module {
    let embeddings: SharpAIEagle25VlVisionEmbeddings
    let encoder: SharpAIEagle25VlVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm?

    init(_ config: SharpAIEagle25VlVisionConfiguration, layerCount: Int, applyPostLayerNorm: Bool)
    {
        self.embeddings = SharpAIEagle25VlVisionEmbeddings(config)
        self.encoder = SharpAIEagle25VlVisionEncoder(config, layerCount: layerCount)
        if applyPostLayerNorm {
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }
    }

    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var h = embeddings(pixels)
        for layer in encoder.layers {
            h = layer(h)
        }
        if let postLayerNorm {
            h = postLayerNorm(h)
        }
        return h
    }
}

/// Wrapper matching vLLM's `SiglipVisionModel`, so keys nest as `vision_model.vision_model.*`.
final class SharpAIEagle25VlVisionModel: Module {
    @ModuleInfo(key: "vision_model") var visionModel: SharpAIEagle25VlVisionTransformer

    init(_ config: SharpAIEagle25VlVisionConfiguration, layerCount: Int, applyPostLayerNorm: Bool)
    {
        _visionModel.wrappedValue = SharpAIEagle25VlVisionTransformer(
            config, layerCount: layerCount, applyPostLayerNorm: applyPostLayerNorm)
    }

    /// `pixels` is channels-last `[N, H, W, C]`. Returns `[N, P, hidden]`.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        visionModel(pixels)
    }
}
