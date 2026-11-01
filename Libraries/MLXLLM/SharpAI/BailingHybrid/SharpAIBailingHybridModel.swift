// SharpAI fork overlay: bailing_hybrid model. Not part of upstream mlx-swift-lm.
//
// Port of the Model and LanguageModel classes in mlx_lm/models/bailing_hybrid.py. Global layers
// (every layer_group_size-th, and the tail) use MLA with a KV cache. The other layers use the
// lightning recurrence with an ArraysCache that holds the float32 state. Checkpoint key names are kept
// as-is (model.word_embeddings, model.layers.N.attention.*), so the module tree matches the HF names.
// Expert stacking and kv_b_proj splitting from Model.sanitize are not done yet (see the overlay sanitizer).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIBailingHybridModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding

    let layers: [SharpAIBailingHybridDecoderLayer]
    let norm: RMSNorm
    /// Index of the first global (MLA) layer. Its cache supplies the position offset and the mask.
    let attentionLayerIndex: Int

    init(_ c: SharpAIBailingHybridConfiguration) {
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        let built = (0 ..< c.hiddenLayers).map {
            SharpAIBailingHybridDecoderLayer(c, layerIdx: $0)
        }
        layers = built
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        attentionLayerIndex = (0 ..< c.hiddenLayers).first { c.isGlobalLayer($0) } ?? 0
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = wordEmbeddings(inputs)
        let L = h.dim(1)
        let offset = cache?[attentionLayerIndex].offset ?? 0

        // Single-token steps need no mask. Multi-token steps get the causal mask over the full cache.
        let mask: MLXArray? = L > 1 ? createCausalMask(n: L, offset: offset) : nil

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], offset: offset)
        }
        return norm(h)
    }
}

public class SharpAIBailingHybridModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let model: SharpAIBailingHybridModelInner
    let configuration: SharpAIBailingHybridConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SharpAIBailingHybridConfiguration) throws {
        try args.validateForForward()
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = SharpAIBailingHybridModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.wordEmbeddings.asLinear(out)
    }

    /// MLA layers get the standard attention cache. Linear layers get an ArraysCache for the state.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        var caches = [KVCache]()
        for i in 0 ..< configuration.hiddenLayers {
            if configuration.isGlobalLayer(i) {
                caches.append(try makeAttentionKVCache(parameters: parameters))
            } else {
                caches.append(ArraysCache(size: 1))
            }
        }
        return caches
    }

    public func makeCache() -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { i -> KVCache in
            configuration.isGlobalLayer(i) ? KVCacheSimple() : ArraysCache(size: 1)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIBailingHybridSanitizer.sanitize(weights, configuration: configuration)
    }
}

extension SharpAIBailingHybridModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
