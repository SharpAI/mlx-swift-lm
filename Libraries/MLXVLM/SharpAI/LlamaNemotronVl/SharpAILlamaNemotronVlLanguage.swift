// SharpAI fork overlay: bidirectional Llama decoder for the LlamaNemotronVL embedder. Not part of upstream mlx-swift-lm.
//
// Attention has no causal mask, as in the embedder. No KV cache: one forward pass per input.
// UNVERIFIED: padding masks for batched inputs. Padded rows are not masked here.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAILlamaNemotronVlLanguage {

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPELayer

        init(_ args: SharpAILlamaNemotronVlConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            let headDim = args.headDim
            let heads = args.attentionHeads
            let kvHeads = args.kvHeads
            self.heads = heads
            self.kvHeads = kvHeads
            self.scale = pow(Float(headDim), -0.5)
            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self.rope = initializeRope(
                dims: headDim, base: args.ropeTheta, traditional: args.ropeTraditional,
                scalingConfig: args.ropeScaling, maxPositionEmbeddings: nil)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    final class MLP: Module {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ args: SharpAILlamaNemotronVlConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attention(inputLayerNorm(x))
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    /// Module path `language_model.*`: `embed_tokens`, `layers`, `norm`. No LM head.
    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        var layers: [Block]
        @ModuleInfo var norm: RMSNorm

        let kvHeads: [Int]

        init(_ args: SharpAILlamaNemotronVlConfiguration.TextConfiguration) {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            self.layers = (0 ..< args.hiddenLayers).map { _ in Block(args) }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        /// Hidden states `[batch, seq, hidden]` for input embeddings `[batch, seq, hidden]`.
        func callAsFunction(inputEmbeddings: MLXArray) -> MLXArray {
            var h = inputEmbeddings
            for layer in layers {
                h = layer(h)
            }
            return norm(h)
        }
    }
}
