// SharpAI fork overlay: SigLIP vision tower and connector for the bagel model. Not part of
// upstream mlx-swift-lm.
//
// The SigLIP blocks follow the private ones in Paligemma.swift. Bagel's own ViT wiring
// (packed NaViT sequences, `vit_pos_embed`) is not modeled. Single image, NCHW input.
// UNVERIFIED: checkpoint key names for `vit_model` and `connector`, and whether Bagel applies
// `post_layernorm` before the connector.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIBagelVisionAttention: Module {
    let numHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "out_proj") var wo: Linear

    init(hiddenSize: Int, numHeads: Int) {
        let headDim = hiddenSize / numHeads
        self.numHeads = numHeads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _wk.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _wv.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _wo.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queries = wq(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

final class SharpAIBagelVisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self.fc1 = Linear(hiddenSize, intermediateSize, bias: true)
        self.fc2 = Linear(intermediateSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

final class SharpAIBagelVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIBagelVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: SharpAIBagelVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAIBagelConfiguration.VisionConfiguration) {
        _attention.wrappedValue = SharpAIBagelVisionAttention(
            hiddenSize: config.hiddenSize, numHeads: config.attentionHeads)
        _layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        mlp = SharpAIBagelVisionMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(layerNorm1(x))
        h = h + mlp(layerNorm2(h))
        return h
    }
}

final class SharpAIBagelVisionEncoder: Module {
    let layers: [SharpAIBagelVisionEncoderLayer]

    init(_ config: SharpAIBagelConfiguration.VisionConfiguration) {
        layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIBagelVisionEncoderLayer(config)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

final class SharpAIBagelVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    // UNVERIFIED: a learned `position_embedding` table is an assumption. Bagel may instead add a
    // 2D sin-cos `vit_pos_embed` after the connector. A real checkpoint may fail a strict load here.
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    /// Patches per side in the position table. Position id of patch (row, col) is
    /// `row * gridSize + col`.
    let gridSize: Int

    init(_ config: SharpAIBagelConfiguration.VisionConfiguration, gridSize: Int) {
        self.gridSize = gridSize
        // UNVERIFIED: Conv2d layout is an assumption. Bagel's NaViT patch embed may be a Linear
        // over flattened patches (2-D weight), which sanitize would not transpose.
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.channels, outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize), stride: .init(config.patchSize)
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: gridSize * gridSize, dimensions: config.hiddenSize)
    }

    /// `x` is NHWC (B, H, W, C). Returns (B, rows * cols, D).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let patches = patchEmbedding(x)
        let (B, rows, cols) = (patches.dim(0), patches.dim(1), patches.dim(2))

        var positionIds = [Int32]()
        positionIds.reserveCapacity(rows * cols)
        for row in 0 ..< rows {
            for col in 0 ..< cols {
                positionIds.append(Int32(row * gridSize + col))
            }
        }
        let ids = MLXArray(positionIds).reshaped(1, rows * cols)

        return patches.reshaped(B, rows * cols, -1) + positionEmbedding(ids)
    }
}

final class SharpAIBagelSigLIPVisionModel: Module {
    @ModuleInfo var embeddings: SharpAIBagelVisionEmbeddings
    @ModuleInfo var encoder: SharpAIBagelVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    init(_ config: SharpAIBagelConfiguration.VisionConfiguration, gridSize: Int) {
        embeddings = SharpAIBagelVisionEmbeddings(config, gridSize: gridSize)
        encoder = SharpAIBagelVisionEncoder(config)
        _postLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    /// `x` is NHWC (B, H, W, C). Returns (B, rows * cols, D).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        postLayerNorm(encoder(embeddings(x)))
    }
}

/// Wraps the SigLIP tower under the `vit_model` prefix of the bagel checkpoint.
final class SharpAIBagelViTModel: Module {
    @ModuleInfo(key: "vision_model") var visionModel: SharpAIBagelSigLIPVisionModel

    init(_ config: SharpAIBagelConfiguration.VisionConfiguration, gridSize: Int) {
        _visionModel.wrappedValue = SharpAIBagelSigLIPVisionModel(config, gridSize: gridSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        visionModel(x)
    }
}

/// Two-layer MLP from vision width to text width. UNVERIFIED: fc2 maps text width to text width.
/// UNVERIFIED: `fc1`/`fc2` key names are an assumption. Bagel may use `projector_1`/`projector_2`.
final class SharpAIBagelConnector: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(visionHidden: Int, textHidden: Int) {
        self.fc1 = Linear(visionHidden, textHidden, bias: true)
        self.fc2 = Linear(textHidden, textHidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}
