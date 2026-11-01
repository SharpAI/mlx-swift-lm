// SharpAI fork overlay: EAGLE draft head for Cohere. Not part of upstream.
//
// Follows vLLM's cohere_eagle.py: fc(concat(embed(ids), hidden)) -> dense Cohere layers -> norm.
// Logits use the target's LM head (or this draft's embedding when untied). The draft has no
// LanguageModel conformance because it needs the target's hidden states as input.
// UNVERIFIED: the layer index offset in checkpoint keys (vLLM uses target layer count).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIEagleCohereDraftModel: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "fc") var fc: Linear

    let layers: [CohereTransformerBlock]
    let norm: LayerNorm
    let logitScale: Float

    public init(_ args: CohereConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        // vLLM's EAGLE fc has a bias term.
        self._fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: true)
        self.layers = (0 ..< args.hiddenLayers).map { _ in CohereTransformerBlock(args) }
        self.norm = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
        self.logitScale = args.logitScale
    }

    public func callAsFunction(
        _ inputs: MLXArray, hiddenStates: MLXArray, cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = fc(concatenated([embedTokens(inputs), hiddenStates], axis: -1))

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    /// Projects draft hidden states to logits. Pass the target's embedding as `head` when tied.
    public func logits(_ hidden: MLXArray, head: Embedding? = nil) -> MLXArray {
        let table = head ?? embedTokens
        return table.asLinear(hidden) * logitScale
    }
}
