// SharpAI fork overlay: SigLIP vision tower for ColPali. Not part of upstream mlx-swift-lm.
//
// Copied from the private PaliGemma vision code so the layout and key names match.
// The encoder keeps the full batch; the PaliGemma copy keeps only batch element 0.

import Foundation
import MLX
import MLXNN

enum SharpAIColpaliVision {

    final class VisionAttention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "out_proj") var wo: Linear

        init(dims: Int, numHeads: Int) {
            precondition(dims % numHeads == 0, "Dimensions must be divisible by numHeads")
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

    final class VisionMLP: Module {
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

    final class VisionEncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: VisionAttention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: VisionMLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: SharpAIColpaliConfiguration.VisionConfiguration) {
            self._attention.wrappedValue = VisionAttention(
                dims: config.hiddenSize, numHeads: config.attentionHeads)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            self.mlp = VisionMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
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

    final class VisionEncoder: Module {
        var layers: [VisionEncoderLayer]

        init(_ config: SharpAIColpaliConfiguration.VisionConfiguration) {
            self.layers = (0 ..< config.hiddenLayers).map { _ in
                VisionEncoderLayer(config)
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

    final class VisionEmbeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        // Underscore prefix keeps MLXNN from registering this as a parameter.
        let _positionIds: MLXArray

        init(_ config: SharpAIColpaliConfiguration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels, outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize), stride: .init(config.patchSize)
            )
            let gridSide = config.imageSize / config.patchSize
            let count = gridSide * gridSide
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: count, dimensions: config.hiddenSize)
            self._positionIds = MLXArray(0 ..< count)[.newAxis, 0...]
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patches = patchEmbedding(x)
            patches = patches.flattened(start: 1, end: 2)
            return patches + positionEmbedding(_positionIds)
        }
    }

    final class VisionTransformer: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: VisionEncoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: SharpAIColpaliConfiguration.VisionConfiguration) {
            self.embeddings = VisionEmbeddings(config)
            self.encoder = VisionEncoder(config)
            self._postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            postLayerNorm(encoder(embeddings(x)))
        }
    }

    /// Module path `vision_tower.vision_model.*`. Input is NHWC pixels.
    final class VisionTower: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionTransformer

        init(_ config: SharpAIColpaliConfiguration.VisionConfiguration) {
            self._visionModel.wrappedValue = VisionTransformer(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            visionModel(x)
        }

        /// Drops `position_ids` and moves the patch conv weight to MLX layout `[out, kH, kW, in]`.
        static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized = [String: MLXArray]()
            for (key, value) in weights {
                if key.contains("position_id") {
                    continue
                }
                if key.contains("patch_embedding.weight"), value.ndim == 4, !isMLXLayout(value) {
                    sanitized[key] = value.transposed(0, 2, 3, 1)
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }

        private static func isMLXLayout(_ array: MLXArray) -> Bool {
            let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }
    }
}
