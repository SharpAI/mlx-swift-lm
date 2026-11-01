// SharpAI fork overlay: granite_swa model. Not part of upstream mlx-swift-lm.
//
// Port of transformers models/granite_swa/modeling_granite_swa.py (GraniteSWAForCausalLM):
// dense Granite with per-layer sliding windows, a per-head attention sink and a per-layer RoPE base.
// Written without compiling.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let slidingLayerType = "sliding_attention"

class SharpAIGraniteSwaAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray

    /// Nil on NoPE layers, where `layer_rope_theta` is 0.
    let rope: RoPELayer?

    init(_ configuration: SharpAIGraniteSwaConfiguration, layerIndex: Int) {
        let heads = configuration.attentionHeads
        let kvHeads = configuration.kvHeads
        let dim = configuration.hiddenSize
        let hd = dim / heads

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = hd
        self.scale = configuration.attentionMultiplier

        _qProj.wrappedValue = Linear(
            dim, heads * hd, bias: configuration.attentionBias)
        _kProj.wrappedValue = Linear(
            dim, kvHeads * hd, bias: configuration.attentionBias)
        _vProj.wrappedValue = Linear(
            dim, kvHeads * hd, bias: configuration.attentionBias)
        _oProj.wrappedValue = Linear(
            heads * hd, dim, bias: configuration.attentionBias)

        _sinks.wrappedValue = zeros([heads])

        let base = configuration.layerRopeTheta[layerIndex]
        if base == 0 {
            self.rope = nil
        } else {
            self.rope = initializeRope(
                dims: hd,
                base: base,
                traditional: false,
                scalingConfig: configuration.ropeScaling,
                maxPositionEmbeddings: configuration.maxPositionEmbeddings)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = attend(
            queries: queries, keys: keys, values: values, cache: cache, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    private func attend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        cache: KVCache?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let cache else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask, sinks: sinks)
        }

        var (cachedKeys, cachedValues) = (keys, values)
        if let quantizedCache = cache as? QuantizedKVCacheProtocol {
            (cachedKeys, cachedValues) = sharpAIDequantizedKVUpdate(
                quantizedCache, keys: keys, values: values, dtype: queries.dtype)
        } else {
            (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask, sinks: sinks)
    }
}

class SharpAIGraniteSwaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ configuration: SharpAIGraniteSwaConfiguration) {
        let dim = configuration.hiddenSize
        let hiddenDim = configuration.intermediateSize
        let mlpBias = configuration.mlpBias

        _gate.wrappedValue = Linear(dim, hiddenDim, bias: mlpBias)
        _down.wrappedValue = Linear(hiddenDim, dim, bias: mlpBias)
        _up.wrappedValue = Linear(dim, hiddenDim, bias: mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class SharpAIGraniteSwaLayer: Module {
    let residualMultiplier: Float

    @ModuleInfo(key: "self_attn") var attention: SharpAIGraniteSwaAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    let mlp: SharpAIGraniteSwaMLP

    init(_ configuration: SharpAIGraniteSwaConfiguration, layerIndex: Int) {
        self.residualMultiplier = configuration.residualMultiplier

        _attention.wrappedValue = SharpAIGraniteSwaAttention(
            configuration, layerIndex: layerIndex)
        self.mlp = SharpAIGraniteSwaMLP(configuration)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + attended * residualMultiplier
        let mlpOut = mlp(postAttentionLayerNorm(h))
        return h + mlpOut * residualMultiplier
    }
}

class SharpAIGraniteSwaModelInner: Module {
    let embeddingMultiplier: Float
    let layerTypes: [String]
    let slidingWindow: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIGraniteSwaLayer]
    let norm: RMSNorm

    init(_ configuration: SharpAIGraniteSwaConfiguration) {
        self.embeddingMultiplier = configuration.embeddingMultiplier
        self.layerTypes = configuration.layerTypes
        self.slidingWindow = configuration.slidingWindow

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        self.layers = (0 ..< configuration.hiddenLayers).map { index in
            SharpAIGraniteSwaLayer(configuration, layerIndex: index)
        }
        self.norm = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs) * embeddingMultiplier

        let caches: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)
        let seqLen = hidden.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if layerTypes[i] == slidingLayerType {
                if slidingMask == nil {
                    slidingMask = makeAttentionMask(
                        n: seqLen, cache: caches[i], windowSize: slidingWindow)
                }
                mask = slidingMask!
            } else {
                if fullMask == nil {
                    fullMask = makeAttentionMask(n: seqLen, cache: caches[i], windowSize: nil)
                }
                mask = fullMask!
            }
            hidden = layer(hidden, mask: mask, cache: caches[i])
        }

        return norm(hidden)
    }
}

public class SharpAIGraniteSwaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let logitsScaling: Float
    let configuration: SharpAIGraniteSwaConfiguration

    let model: SharpAIGraniteSwaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIGraniteSwaConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self.logitsScaling = configuration.logitsScaling

        self.model = SharpAIGraniteSwaModelInner(configuration)

        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out / logitsScaling
    }

    /// Sliding layers get a rotating window cache. Full layers get the ordinary cache.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layerType == slidingLayerType)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIGraniteSwaSanitizer.sanitize(weights, configuration: configuration)
    }
}

extension SharpAIGraniteSwaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
