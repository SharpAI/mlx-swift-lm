// SharpAI fork overlay: SigLIP tower and pixel-shuffle connector for ModernVBERT.
// Not part of upstream mlx-swift-lm.
//
// Layout follows the SigLIP names used by the other SharpAI vision towers: `vision_model.*`.
// UNVERIFIED: the output is the post-layernorm last hidden state, and the connector is a
// bias-free linear layer under `connector.modality_projection.proj`.

import Foundation
import MLX
import MLXNN

enum SharpAIModernvbertVision {

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

        init(_ config: SharpAIModernvbertConfiguration.VisionConfiguration) {
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

        init(_ config: SharpAIModernvbertConfiguration.VisionConfiguration) {
            self.layers = (0 ..< config.hiddenLayers).map { _ in
                EncoderLayer(config)
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

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        // Plain Int, not MLXArray: a stored MLXArray would be read as a `position_ids` parameter.
        let positionCount: Int

        init(_ config: SharpAIModernvbertConfiguration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels, outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize), stride: .init(config.patchSize))
            let gridSide = config.imageSize / config.patchSize
            let count = gridSide * gridSide
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: count, dimensions: config.hiddenSize)
            self.positionCount = count
        }

        /// NHWC pixels in, `[batch, patches, hidden]` out.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let patches = patchEmbedding(x).flattened(start: 1, end: 2)
            let positionIds = MLXArray(0 ..< positionCount)[.newAxis, 0...]
            return patches + positionEmbedding(positionIds)
        }
    }

    final class Transformer: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: SharpAIModernvbertConfiguration.VisionConfiguration) {
            self.embeddings = Embeddings(config)
            self.encoder = Encoder(config)
            self._postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            postLayerNorm(encoder(embeddings(x)))
        }
    }

    final class Projection: Module {
        @ModuleInfo var proj: Linear

        init(inputSize: Int, outputSize: Int) {
            self._proj.wrappedValue = Linear(inputSize, outputSize, bias: false)
        }
    }

    /// Pixel shuffle, then a linear map to the text width. Module path `connector.modality_projection.*`.
    final class Connector: Module {
        @ModuleInfo(key: "modality_projection") var modalityProjection: Projection
        let factor: Int

        init(_ visionConfig: SharpAIModernvbertConfiguration.VisionConfiguration, factor: Int, textHidden: Int) {
            self.factor = factor
            self._modalityProjection.wrappedValue = Projection(
                inputSize: visionConfig.hiddenSize * factor * factor, outputSize: textHidden)
        }

        /// `[batch, patches, channels]` in, `[batch, patches / factor^2, channels * factor^2]` out.
        /// The patch grid must be square and divisible by `factor`.
        static func pixelShuffle(_ x: MLXArray, factor: Int) -> MLXArray {
            let (B, N, C) = (x.dim(0), x.dim(1), x.dim(2))
            let side = Int(Double(N).squareRoot())
            let reduced = side / factor

            var h = x.reshaped(B, side, side, C)
            h = h.reshaped(B, side, reduced, C * factor)
            h = h.transposed(0, 2, 1, 3)
            h = h.reshaped(B, reduced, reduced, C * factor * factor)
            h = h.transposed(0, 2, 1, 3)
            return h.reshaped(B, reduced * reduced, C * factor * factor)
        }

        func callAsFunction(_ features: MLXArray) -> MLXArray {
            modalityProjection.proj(Connector.pixelShuffle(features, factor: factor))
        }
    }
}
