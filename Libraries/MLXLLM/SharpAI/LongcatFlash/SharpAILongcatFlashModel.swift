// SharpAI fork overlay: longcat_flash (LongcatFlashNgramForCausalLM) model.
// Mirrors LongcatFlashNgramModel in mlx-lm longcat_flash_ngram.py. Not part of upstream.
//
// Cache layout, one flat [KVCache] array:
//   [0]        ArraysCache holding the last (emb_neighbor_num - 1) token ids
//   [1 + i]    CacheList(attention 0, attention 1) for decoder layer i

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAILongcatFlashDecoderLayer: Module {
    var mlp: SharpAILongcatFlashMoE
    var self_attn: [SharpAILongcatFlashAttention]
    var mlps: [SharpAILongcatFlashDenseMLP]
    var input_layernorm: [RMSNorm]
    var post_attention_layernorm: [RMSNorm]

    init(_ config: SharpAILongcatFlashConfiguration) {
        self.mlp = SharpAILongcatFlashMoE(config)
        self.self_attn = (0 ..< 2).map { _ in SharpAILongcatFlashAttention(config) }
        self.mlps = (0 ..< 2).map { _ in
            SharpAILongcatFlashDenseMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.ffnHiddenSize)
        }
        self.input_layernorm = (0 ..< 2).map { _ in
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self.post_attention_layernorm = (0 ..< 2).map { _ in
            RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
    }

    /// Two attention and dense-MLP sub-blocks. The MoE output from sub-block 0 is added at the end.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        var h = x
        var shortcut: MLXArray? = nil

        for i in 0 ..< 2 {
            let residual = h
            let attnOut = self_attn[i](
                input_layernorm[i](h), mask: mask, cache: cache?[i])
            h = residual + attnOut

            let preNorm = h
            let normed = post_attention_layernorm[i](h)
            if i == 0 {
                shortcut = mlp(normed)
            }
            h = preNorm + mlps[i](normed)

            if i == 1, let shortcut {
                h = h + shortcut
            }
        }
        return h
    }
}

class SharpAILongcatFlashModelInner: Module {
    let numLayers: Int

    @ModuleInfo(key: "ngram_embeddings") var ngramEmbeddings: SharpAILongcatFlashNgramEmbedding
    var layers: [SharpAILongcatFlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAILongcatFlashConfiguration) {
        self.numLayers = config.numLayers
        self._ngramEmbeddings.wrappedValue = SharpAILongcatFlashNgramEmbedding(config)
        self.layers = (0 ..< config.numLayers).map { _ in
            SharpAILongcatFlashDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Splits the flat cache into the n-gram history cache and the per-layer CacheLists.
    private func splitCaches(_ cache: [KVCache]?) -> (ArraysCache?, [CacheList?]) {
        guard let cache else {
            return (nil, Array(repeating: nil, count: layers.count))
        }
        precondition(
            cache.count == layers.count + 1,
            "longcat_flash expects 1 n-gram cache and \(layers.count) layer caches")
        guard let ngramCache = cache[0] as? ArraysCache else {
            fatalError("longcat_flash cache entry 0 must be an ArraysCache")
        }
        let layerCaches: [CacheList?] = cache.dropFirst().map { (entry: KVCache) -> CacheList? in
            guard let list = entry as? CacheList else {
                fatalError("longcat_flash layer caches must be CacheList")
            }
            return list
        }
        return (ngramCache, layerCaches)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let (ngramCache, layerCaches) = splitCaches(cache)

        var h = ngramEmbeddings(inputs, cache: ngramCache)

        // Same mask source as mlx-lm: the first attention cache of layer 0.
        let firstAttention: KVCache? = layerCaches.first.flatMap { $0 }?[0]
        let mask = createAttentionMask(h: h, cache: firstAttention, returnArray: true)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: layerCaches[i])
        }
        return norm(h)
    }
}

class SharpAILongcatFlashModel: Module, LLMModel {
    let config: SharpAILongcatFlashConfiguration
    var model: SharpAILongcatFlashModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAILongcatFlashConfiguration) {
        self.config = config
        self.model = SharpAILongcatFlashModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// One n-gram history cache, then one CacheList of two attention caches per layer.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        var caches: [KVCache] = [ArraysCache(size: 1)]
        for _ in 0 ..< config.numLayers {
            let first = try makeAttentionKVCache(parameters: parameters)
            let second = try makeAttentionKVCache(parameters: parameters)
            caches.append(CacheList(first, second))
        }
        return caches
    }

    /// Stacks per-expert weights into switch_mlp and renames the n-gram base embedding.
    /// kv_b_proj keeps its checkpoint name and is split at call time.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        if let embedding = newWeights.removeValue(forKey: "model.embed_tokens.weight") {
            newWeights["model.ngram_embeddings.word_embeddings.weight"] = embedding
        }

        let expertCount = config.nRoutedExperts
        for l in 0 ..< config.numLayers {
            let prefix = "model.layers.\(l)"
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let expertKeys = (0 ..< expertCount).map {
                        "\(prefix).mlp.experts.\($0).\(proj).\(key)"
                    }
                    guard expertKeys.allSatisfy({ newWeights[$0] != nil }) else {
                        continue
                    }
                    let experts = expertKeys.map { newWeights.removeValue(forKey: $0)! }
                    newWeights["\(prefix).mlp.switch_mlp.\(proj).\(key)"] = stacked(experts)
                }
            }
        }

        return newWeights.filter { !$0.key.hasPrefix("model.mtp") }
    }

    var loraLayers: [Module] {
        model.layers
    }
}
