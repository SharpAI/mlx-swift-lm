// SharpAI fork overlay: ModernBERT text encoder for ModernVBERT. Not part of upstream mlx-swift-lm.
//
// Bidirectional encoder with no KV cache. Layer 0 has no pre-attention norm. Every
// `global_attn_every_n_layers` layers attend globally. The rest use a sliding window of
// `local_attention / 2` tokens on each side. UNVERIFIED: the window rule and the
// GeGLU order (act on the first half, gate with the second) follow ModernBERT, not a
// checked reference run.

import Foundation
import MLX
import MLXNN

enum SharpAIModernvbertText {

    /// Token embedding table. The LayerNorm runs after the image merge, not inside the table.
    final class Embeddings: Module {
        @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
        @ModuleInfo var norm: LayerNorm

        init(_ config: SharpAIModernvbertConfiguration.TextConfiguration) {
            self._tokEmbeddings.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            self._norm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
        }
    }

    /// Fused QKV attention with half-split RoPE. Keys: `attn.Wqkv`, `attn.Wo`.
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float
        let ropeBase: Float

        @ModuleInfo(key: "Wqkv") var wqkv: Linear
        @ModuleInfo(key: "Wo") var wo: Linear

        init(_ config: SharpAIModernvbertConfiguration.TextConfiguration, isGlobal: Bool) {
            self.heads = config.attentionHeads
            self.headDim = config.headDim
            self.scale = pow(Float(config.headDim), -0.5)
            self.ropeBase = isGlobal ? config.globalRopeTheta : config.localRopeTheta
            self._wqkv.wrappedValue = Linear(
                config.hiddenSize, 3 * config.hiddenSize, bias: config.attentionBias)
            self._wo.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
        }

        /// `mask` is additive, broadcastable to `[batch, heads, seq, seq]`, or nil.
        func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let qkv = wqkv(x).reshaped(B, L, 3, heads, headDim)

            var queries = qkv[0..., 0..., 0].transposed(0, 2, 1, 3)
            var keys = qkv[0..., 0..., 1].transposed(0, 2, 1, 3)
            let values = qkv[0..., 0..., 2].transposed(0, 2, 1, 3)

            queries = MLXFast.RoPE(
                queries, dimensions: headDim, traditional: false, base: ropeBase, scale: 1,
                offset: 0)
            keys = MLXFast.RoPE(
                keys, dimensions: headDim, traditional: false, base: ropeBase, scale: 1,
                offset: 0)

            let output: MLXArray
            if let mask {
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale,
                    mask: .array(mask.asType(queries.dtype)))
            } else {
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: .none)
            }

            return wo(
                output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    /// GeGLU feed-forward. Keys: `mlp.Wi`, `mlp.Wo`.
    final class MLP: Module {
        @ModuleInfo(key: "Wi") var wi: Linear
        @ModuleInfo(key: "Wo") var wo: Linear

        init(_ config: SharpAIModernvbertConfiguration.TextConfiguration) {
            self._wi.wrappedValue = Linear(
                config.hiddenSize, 2 * config.intermediateSize, bias: config.mlpBias)
            self._wo.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = wi(x).split(parts: 2, axis: -1)
            return wo(gelu(parts[0]) * parts[1])
        }
    }

    final class EncoderLayer: Module {
        /// Nil on layer 0, where the input is not normalized again.
        @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm?
        @ModuleInfo var attn: Attention
        @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
        @ModuleInfo var mlp: MLP

        let isGlobal: Bool

        init(_ config: SharpAIModernvbertConfiguration.TextConfiguration, layerIndex: Int) {
            let global = layerIndex % config.globalAttnEveryNLayers == 0
            self.isGlobal = global
            if layerIndex == 0 {
                self._attnNorm.wrappedValue = nil
            } else {
                self._attnNorm.wrappedValue = LayerNorm(
                    dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
            }
            self.attn = Attention(config, isGlobal: global)
            self._mlpNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
            self.mlp = MLP(config)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
            let normed = attnNorm.map { $0(x) } ?? x
            var h = x + attn(normed, mask: mask)
            h = h + mlp(mlpNorm(h))
            return h
        }
    }

    /// ModernBERT body. Module path `text_model.*` in the checkpoint.
    final class TextModel: Module {
        @ModuleInfo var embeddings: Embeddings
        var layers: [EncoderLayer]
        @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

        let localWindow: Int

        init(_ config: SharpAIModernvbertConfiguration.TextConfiguration) {
            self.embeddings = Embeddings(config)
            self.layers = (0 ..< config.hiddenLayers).map { index in
                EncoderLayer(config, layerIndex: index)
            }
            self._finalNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias)
            self.localWindow = config.localAttention
        }

        /// Encodes token embeddings `[batch, seq, hidden]` taken before the embedding norm.
        /// `paddingMask` is Bool `[batch, seq]`, true for real tokens. Nil means no padding.
        func encode(_ tokenEmbeddings: MLXArray, paddingMask: MLXArray?) -> MLXArray {
            let (B, L) = (tokenEmbeddings.dim(0), tokenEmbeddings.dim(1))
            let pad = paddingMask.map { $0.reshaped(B, 1, 1, L) }

            let indices = MLXArray(0 ..< L)
            let distance = abs(
                indices.expandedDimensions(axis: 1) - indices.expandedDimensions(axis: 0))
            let band = (distance .<= MLXArray(localWindow / 2)).reshaped(1, 1, L, L)

            let globalMask: MLXArray? = pad.map { additiveMask($0) }
            let localMask = additiveMask(pad.map { band & $0 } ?? band)

            var h = embeddings.norm(tokenEmbeddings)
            for layer in layers {
                h = layer(h, mask: layer.isGlobal ? globalMask : localMask)
            }
            return finalNorm(h)
        }

        private func additiveMask(_ allowed: MLXArray) -> MLXArray {
            which(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        }
    }
}
