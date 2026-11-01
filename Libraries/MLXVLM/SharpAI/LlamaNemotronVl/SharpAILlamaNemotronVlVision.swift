// SharpAI fork overlay: SigLIP vision transformer for the LlamaNemotronVL embedder. Not part of upstream mlx-swift-lm.
//
// Layout follows the checkpoint: `vision_model.embeddings`, `vision_model.encoder`,
// `vision_model.post_layernorm`. No CLS token. Input is NHWC pixels.

import Foundation
import MLX
import MLXNN

enum SharpAILlamaNemotronVlVision {

    final class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "out_proj") var wo: Linear

        init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.scale = pow(Float(dims / numHeads), -0.5)
            self._wq.wrappedValue = Linear(dims, dims, bias: true)
            self._wk.wrappedValue = Linear(dims, dims, bias: true)
            self._wv.wrappedValue = Linear(dims, dims, bias: true)
            self._wo.wrappedValue = Linear(dims, dims, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            let queries = wq(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            let keys = wk(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    final class MLP: Module {
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

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: SharpAILlamaNemotronVlConfiguration.VisionConfiguration) {
            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.attentionHeads)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            self.mlp = MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var r = attention(layerNorm1(x))
            let h = x + r
            r = mlp(layerNorm2(h))
            return h + r
        }
    }

    final class Encoder: Module {
        var layers: [EncoderLayer]

        init(_ config: SharpAILlamaNemotronVlConfiguration.VisionConfiguration) {
            self.layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x
            for layer in layers {
                h = layer(h)
            }
            return h
        }
    }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        // Plain Int, not MLXArray: an MLXArray property would become a checkpoint parameter.
        let positionCount: Int

        init(_ config: SharpAILlamaNemotronVlConfiguration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels, outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize), stride: .init(config.patchSize)
            )
            let gridSide = config.imageSize / config.patchSize
            let count = gridSide * gridSide
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: count, dimensions: config.hiddenSize)
            self.positionCount = count
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patches = patchEmbedding(x)
            patches = patches.flattened(start: 1, end: 2)
            let positionIds = MLXArray(0 ..< positionCount)[.newAxis, 0...]
            return patches + positionEmbedding(positionIds)
        }
    }

    /// Module path `vision_model.*`. Returns `[batch, patches, hidden]` after `post_layernorm`.
    final class Transformer: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: SharpAILlamaNemotronVlConfiguration.VisionConfiguration) {
            self.embeddings = Embeddings(config)
            self.encoder = Encoder(config)
            self._postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            postLayerNorm(encoder(embeddings(x)))
        }
    }
}
