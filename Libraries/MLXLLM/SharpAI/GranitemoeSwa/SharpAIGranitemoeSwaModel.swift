// SharpAI fork overlay: granitemoe_swa model. Not part of upstream mlx-swift-lm.
//
// Port of vLLM granitemoeshared.py (GraniteMoeSWAForCausalLM) with granite.py's
// per-layer sliding window, attention sinks and per-layer RoPE base.
// Written without compiling. Attention sinks take the MLXFast SDPA `sinks:` path.
// UNVERIFIED: the sliding-window size convention (window includes the current token) follows
// the repo's makeAttentionMask. Confirm it against vLLM's per-layer window on a real run.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let slidingLayerType = "sliding_attention"

class SharpAIGranitemoeSwaAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let hasSinks: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray

    /// Nil on NoPE layers, where `layer_rope_theta` is 0.
    let rope: RoPELayer?

    init(_ configuration: SharpAIGranitemoeSwaConfiguration, layerIndex: Int) {
        let heads = configuration.attentionHeads
        let kvHeads = configuration.kvHeads
        let dim = configuration.hiddenSize
        let hd = dim / heads

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = hd
        // UNVERIFIED: vLLM falls back to head_dim**-1 when attention_multiplier is None.
        self.scale = configuration.attentionMultiplier ?? (1.0 / Float(hd))
        self.hasSinks = configuration.usesAttentionSinks

        // vLLM: attention projections carry no bias.
        _qProj.wrappedValue = Linear(dim, heads * hd, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * hd, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * hd, bias: false)
        _oProj.wrappedValue = Linear(heads * hd, dim, bias: false)

        _sinks.wrappedValue = zeros([heads])

        let base = configuration.layerRopeTheta?[layerIndex] ?? configuration.ropeTheta
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
        guard hasSinks else {
            return attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask)
        }

        guard let cache else {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask, sinks: sinks)
        }

        precondition(
            !(cache is QuantizedKVCacheProtocol),
            "granitemoe_swa attention sinks do not support quantized KV caches")
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask, sinks: sinks)
    }

    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        // Layers without sinks do not load a `sinks` tensor. The default stays unused.
        if parameter == "sinks", !hasSinks {
            return
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }
}

/// Top-k router. Softmax runs over the selected logits only, as vLLM's renormalized routing does.
class SharpAIGranitemoeSwaRouter: Module {
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "layer") var layer: Linear

    init(inputSize: Int, numExperts: Int, topK: Int) {
        self.numExperts = numExperts
        self.topK = topK
        _layer.wrappedValue = Linear(inputSize, numExperts, bias: false)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (MLXArray, MLXArray) {
        let logits = layer(hiddenStates)
        let indices = MLX.argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let topKLogits = MLX.takeAlong(logits, indices, axis: -1)
        let gates = MLX.softmax(topKLogits, axis: -1, precise: true)
        return (indices, gates)
    }
}

class SharpAIGranitemoeSwaMoE: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    let router: SharpAIGranitemoeSwaRouter

    init(_ configuration: SharpAIGranitemoeSwaConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.intermediateSize,
            numExperts: configuration.numLocalExperts,
            bias: false)
        self.router = SharpAIGranitemoeSwaRouter(
            inputSize: configuration.hiddenSize,
            numExperts: configuration.numLocalExperts,
            topK: configuration.numExpertsPerToken)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, gates) = router(x)
        let expertOutputs = switchMLP(x, indices)
        return weightedExpertSum(expertOutputs, gates)
    }
}

/// Dense shared MLP. `input_linear` holds gate and up halves, as in vLLM's SiluAndMul.
class SharpAIGranitemoeSwaSharedMLP: Module {
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "output_linear") var outputLinear: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _inputLinear.wrappedValue = Linear(hiddenSize, intermediateSize * 2, bias: false)
        _outputLinear.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = inputLinear(x).split(parts: 2, axis: -1)
        return outputLinear(silu(parts[0]) * parts[1])
    }
}

class SharpAIGranitemoeSwaLayer: Module {
    let residualMultiplier: Float

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIGranitemoeSwaAttention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoE: SharpAIGranitemoeSwaMoE
    @ModuleInfo(key: "shared_mlp") var sharedMLP: SharpAIGranitemoeSwaSharedMLP?

    init(_ configuration: SharpAIGranitemoeSwaConfiguration, layerIndex: Int) {
        self.residualMultiplier = configuration.residualMultiplier

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _selfAttn.wrappedValue = SharpAIGranitemoeSwaAttention(
            configuration, layerIndex: layerIndex)
        _blockSparseMoE.wrappedValue = SharpAIGranitemoeSwaMoE(configuration)

        if let shared = configuration.sharedIntermediateSize, shared > 0 {
            _sharedMLP.wrappedValue = SharpAIGranitemoeSwaSharedMLP(
                hiddenSize: configuration.hiddenSize, intermediateSize: shared)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let hidden = x + attended * residualMultiplier

        let normed = postAttentionLayerNorm(hidden)
        var mlpOutput = blockSparseMoE(normed)
        if let sharedMLP {
            mlpOutput = mlpOutput + sharedMLP(normed)
        }
        return hidden + mlpOutput * residualMultiplier
    }
}

class SharpAIGranitemoeSwaModelInner: Module {
    let embeddingMultiplier: Float
    let layerTypes: [String]
    let slidingWindow: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIGranitemoeSwaLayer]
    let norm: RMSNorm

    init(_ configuration: SharpAIGranitemoeSwaConfiguration) {
        self.embeddingMultiplier = configuration.embeddingMultiplier
        self.layerTypes = configuration.resolvedLayerTypes
        self.slidingWindow = configuration.slidingWindow

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        self.layers = (0 ..< configuration.hiddenLayers).map { index in
            SharpAIGranitemoeSwaLayer(configuration, layerIndex: index)
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

public class SharpAIGranitemoeSwaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let logitsScaling: Float
    let configuration: SharpAIGranitemoeSwaConfiguration

    let model: SharpAIGranitemoeSwaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIGranitemoeSwaConfiguration) {
        self.configuration = configuration
        self.modelType = configuration.modelType
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self.logitsScaling = configuration.logitsScaling

        self.model = SharpAIGranitemoeSwaModelInner(configuration)

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
        try configuration.resolvedLayerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layerType == slidingLayerType)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIGranitemoeSwaSanitizer.sanitize(weights, configuration: configuration)
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
