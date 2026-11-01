// SharpAI fork overlay: k2_horizon model. Not part of upstream mlx-swift-lm.
//
// Reuses LlamaAttention and LlamaMLP unchanged. The layer norms and the final norm are
// grouped RMSNorm. Weight keys follow the Llama layout (model.layers.N.self_attn.*, mlp.*).
//
// UNVERIFIED: the Llama key layout and the norm placement are assumed. No checkpoint or
// reference implementation was available to check them.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIK2HorizonDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LlamaAttention
    @ModuleInfo(key: "mlp") var mlp: LlamaMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIK2HorizonGroupedRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm:
        SharpAIK2HorizonGroupedRMSNorm

    init(_ config: SharpAIK2HorizonConfiguration, llama: LlamaConfiguration) {
        self._attention.wrappedValue = LlamaAttention(llama)
        self._mlp.wrappedValue = LlamaMLP(llama)
        self._inputLayerNorm.wrappedValue = SharpAIK2HorizonGroupedRMSNorm(
            dimensions: config.hiddenSize, groups: config.rmsNormGroups, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = SharpAIK2HorizonGroupedRMSNorm(
            dimensions: config.hiddenSize, groups: config.rmsNormGroups, eps: config.rmsNormEps)
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

public class SharpAIK2HorizonModelInner: Module, LayerPartitionable {

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding

    let layers: [SharpAIK2HorizonDecoderLayer]
    let norm: SharpAIK2HorizonGroupedRMSNorm

    // LayerPartitionable
    public var gpuLayerCount: Int?
    public var totalLayerCount: Int { layers.count }

    init(_ config: SharpAIK2HorizonConfiguration, llama: LlamaConfiguration) {
        precondition(config.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIK2HorizonDecoderLayer(config, llama: llama)
        }
        self.norm = SharpAIK2HorizonGroupedRMSNorm(
            dimensions: config.hiddenSize, groups: config.rmsNormGroups, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount) {
                layer(h, mask: mask, cache: cache?[i])
            }
        }

        return norm(h)
    }
}

/// Text model for the `k2_horizon` model type.
public class SharpAIK2HorizonModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: SharpAIK2HorizonModelInner

    let configuration: SharpAIK2HorizonConfiguration

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ config: SharpAIK2HorizonConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIK2HorizonModelInner(config, llama: config.llama)
        self.configuration = config
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Drop precomputed rotary frequencies and the head when embeddings are tied.
        filterLMHeadWeights(
            from: weights.filter {
                !$0.key.contains("self_attn.rotary_emb.inv_freq")
            },
            tiedWordEmbeddings: configuration.tieWordEmbeddings)
    }
}

extension SharpAIK2HorizonModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
