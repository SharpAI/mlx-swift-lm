// SharpAI fork overlay: Gemma text decoder for ColPali. Not part of upstream mlx-swift-lm.
//
// Copied from the private PaliGemma language code. Returns final hidden states
// and has no LM head, because ColPali projects hidden states, not logits.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAIColpaliText {

    final class TextAttention: Module {
        let args: SharpAIColpaliConfiguration.TextConfiguration
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPE

        init(_ args: SharpAIColpaliConfiguration.TextConfiguration) {
            self.args = args
            let dim = args.hiddenSize
            let heads = args.attentionHeads
            let kvHeads = args.kvHeads
            let headDim = args.resolvedHeadDim

            self.scale = pow(Float(headDim), -0.5)
            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self.rope = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    final class TextMLP: Module {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(gelu(gate(x)) * up(x))
        }
    }

    final class TextBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: TextAttention
        @ModuleInfo var mlp: TextMLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm

        init(_ args: SharpAIColpaliConfiguration.TextConfiguration) {
            self._attention.wrappedValue = TextAttention(args)
            self.mlp = TextMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            return h + r
        }
    }

    /// Module path `language_model.model.*`.
    final class TextDecoder: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        var layers: [TextBlock]
        @ModuleInfo var norm: Gemma.RMSNorm

        let hiddenScale: Float

        init(_ args: SharpAIColpaliConfiguration.TextConfiguration) {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            self.hiddenScale = pow(Float(args.hiddenSize), 0.5)
            self.layers = (0 ..< args.hiddenLayers).map { _ in
                TextBlock(args)
            }
            self.norm = Gemma.RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        /// `bidirectional` drops the causal mask when there is no cache yet. It is set for
        /// image documents only. UNVERIFIED: ColPali's exact prefix mask is not checked
        /// against the reference implementation.
        func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]?, inputEmbedding: MLXArray?, bidirectional: Bool
        ) -> MLXArray {
            var h = inputEmbedding ?? embedTokens(inputs)
            h = h * hiddenScale

            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                (bidirectional && (cache?.first?.offset ?? 0) == 0)
                ? .none
                : createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    /// Module path `language_model.*`, with `model` as the decoder.
    final class TextModel: Module {
        @ModuleInfo var model: TextDecoder

        let kvHeads: [Int]

        init(_ args: SharpAIColpaliConfiguration.TextConfiguration) {
            self.model = TextDecoder(args)
            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }
    }
}
