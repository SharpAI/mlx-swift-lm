// SharpAI fork overlay: deepseek_v32 model with an MTP head. Not part of upstream mlx-swift-lm.
//
// The trunk matches DeepseekV32Model: same decoder layers, indexer, and CacheList layout.
// The difference is the MTP block at model.layers.<num_hidden_layers>. It is built only when
// `retainMTP` is set (default: MTPConfig.retainMTPWeights), so the default load matches upstream.
//
// UNVERIFIED: how callers align tokens with MTP depths. callMTP feeds the same token
// embeddings to every depth, so a caller that needs depth-shifted tokens must pass them.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class SharpAIDeepseekV32Inner: Module, LayerPartitionable, StreamableMoE {
    let numMainLayers: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV32DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public var gpuLayerCount: Int? = nil
    public var streamExperts: Bool = false
    public var totalLayerCount: Int { numMainLayers }

    init(_ args: DeepseekV32Configuration, includeMTP: Bool) {
        self.numMainLayers = args.numHiddenLayers
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map {
            DeepseekV32DecoderLayer(args, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if includeMTP {
            for depth in 0 ..< args.numNextnPredictLayers {
                layers.append(
                    SharpAIDeepseekV32MTPBlock(
                        mtpConfiguration: args, layerIdx: args.numHiddenLayers + depth))
            }
        }
    }

    /// Runs the main stack only. MTP blocks sit past `numMainLayers` and are not run here.
    func callAsFunction(_ x: MLXArray, cache: [CacheList]?) -> MLXArray {
        var h = embedTokens(x)
        let attentionMask = createAttentionMask(h: h, cache: cache?.first?[0])

        for i in 0 ..< numMainLayers {
            let layer = layers[i]
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts) {
                layer(h, mask: attentionMask, cache: cache?[i])
            }
        }

        return norm(h)
    }
}

public final class SharpAIDeepseekV32Model: Module, LLMModel, LoRAModel, MTPLanguageModel {
    public var kvHeads: [Int] { Array(repeating: args.numAttentionHeads, count: args.numHiddenLayers) }

    var args: DeepseekV32Configuration
    /// True when the MTP block is built and its weights are kept.
    let retainMTP: Bool
    public var model: SharpAIDeepseekV32Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: DeepseekV32Configuration, retainMTP: Bool = MTPConfig.retainMTPWeights) {
        let includeMTP = retainMTP && args.numNextnPredictLayers > 0
        self.args = args
        self.retainMTP = includeMTP
        self.model = SharpAIDeepseekV32Inner(args, includeMTP: includeMTP)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: Self.typedCaches(cache)))
    }

    /// One `CacheList(main, indexer)` per layer, same as DeepseekV32Model.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< args.numHiddenLayers).map { _ -> KVCache in
            makeCacheList(parameters: parameters)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIDeepseekV32Sanitizer.sanitize(
            weights, configuration: args, retainMTP: retainMTP)
    }

    public var loraLayers: [Module] {
        model.layers
    }

    // MARK: - MTP

    /// Returns `[main_logits, mtp_0_logits, …]`. Each depth runs one MTP block over the
    /// previous depth's hidden state and the token embeddings of `inputs`.
    public func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?)
        -> [MLXArray]
    {
        guard retainMTP else {
            return [callAsFunction(inputs, cache: cache)]
        }

        let hidden = model(inputs, cache: Self.typedCaches(cache))
        var results = [lmHead(hidden)]
        let embeddings = model.embedTokens(inputs)
        var previous = hidden

        for depth in 0 ..< args.numNextnPredictLayers {
            guard let block = model.layers[args.numHiddenLayers + depth]
                as? SharpAIDeepseekV32MTPBlock
            else {
                preconditionFailure("deepseek_v32 MTP block \(depth) is missing")
            }

            var blockCache: CacheList? = nil
            if let mtpCaches, depth < mtpCaches.count {
                blockCache = mtpCaches[depth].first as? CacheList
            }

            previous = block.predict(
                tokenEmbeddings: embeddings, hidden: previous, cache: blockCache)
            results.append(lmHead(block.sharedHead.norm(previous)))
        }

        return results
    }

    public func makeMTPCaches(parameters: GenerateParameters?) -> [[KVCache]] {
        let depths = retainMTP ? args.numNextnPredictLayers : 0
        return (0 ..< depths).map { _ -> [KVCache] in
            [makeCacheList(parameters: parameters)]
        }
    }

    // MARK: - Helpers

    private func makeCacheList(parameters: GenerateParameters?) -> CacheList {
        if let maxKVSize = parameters?.maxKVSize {
            return CacheList(
                RotatingKVCache(maxSize: maxKVSize, keep: 4),
                RotatingKVCache(maxSize: maxKVSize, keep: 4))
        }
        return CacheList(KVCacheSimple(), KVCacheSimple())
    }

    private static func typedCaches(_ cache: [KVCache]?) -> [CacheList]? {
        guard let cache else { return nil }
        guard let typed = cache as? [CacheList] else {
            fatalError(
                "SharpAIDeepseekV32Model requires CacheList-typed caches (from its own "
                    + "newCache(parameters:)); got \(type(of: cache))"
            )
        }
        return typed
    }
}
