// SharpAI fork overlay: domynedge text model. Not part of upstream mlx-swift-lm.
//
// Written without compiling. Layout covers: per-layer full or sliding-window attention,
// per-layer head size, RoPE skipped on "nope" layers, and two pre-norms per block.
// UNVERIFIED: the two-norm pre-norm layout (input_layernorm, post_attention_layernorm) is
// copied from Llama. No domynedge source on this machine confirms it. Check it against the
// Hub modeling code and a real checkpoint index.
// UNVERIFIED: RMSNorm is the plain form (weight, not 1 + weight). Same check applies.
// UNVERIFIED: attention scale is 1/sqrt(head_dim). No QK norm is assumed.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIDomynedgeAttention: Module {
    let headDim: Int
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float
    let rope: RoPELayer?

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    init(_ config: SharpAIDomynedgeConfiguration, spec: SharpAIDomynedgeLayerSpec) {
        self.headDim = spec.headDim
        self.attentionHeads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.scale = pow(Float(spec.headDim), -0.5)

        if spec.usesRope {
            self.rope = initializeRope(
                dims: spec.headDim, base: config.ropeTheta, traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
        } else {
            self.rope = nil
        }

        self._queryProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * spec.headDim, bias: config.attentionBias)
        self._keyProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * spec.headDim, bias: config.attentionBias)
        self._valueProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * spec.headDim, bias: config.attentionBias)
        self._outputProj.wrappedValue = Linear(
            config.attentionHeads * spec.headDim, config.hiddenSize, bias: config.attentionBias)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = queryProj(x).reshaped(B, L, attentionHeads, headDim).transposed(0, 2, 1, 3)
        var keys = keyProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = valueProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
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
        return outputProj(output)
    }
}

class SharpAIDomynedgeMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: SharpAIDomynedgeConfiguration) {
        self._gate.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class SharpAIDomynedgeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIDomynedgeAttention
    @ModuleInfo var mlp: SharpAIDomynedgeMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIDomynedgeConfiguration, spec: SharpAIDomynedgeLayerSpec) {
        self._selfAttn.wrappedValue = SharpAIDomynedgeAttention(config, spec: spec)
        self._mlp.wrappedValue = SharpAIDomynedgeMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAIDomynedgeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIDomynedgeDecoderLayer]
    let layerSpecs: [SharpAIDomynedgeLayerSpec]
    let norm: RMSNorm

    init(_ config: SharpAIDomynedgeConfiguration) {
        let specs = config.layerSpecs()
        self.layerSpecs = specs
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = specs.map { SharpAIDomynedgeDecoderLayer(config, spec: $0) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        // Layers with the same window share a mask. All caches advance in lockstep,
        // so one mask per window size is valid for every layer that uses it.
        var masks: [Int: MLXFast.ScaledDotProductAttentionMaskMode] = [:]
        for (i, layer) in layers.enumerated() {
            let window = layerSpecs[i].slidingWindow
            let key = window ?? 0
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if let cached = masks[key] {
                mask = cached
            } else {
                mask = createAttentionMask(h: h, cache: cache?[i], windowSize: window)
                masks[key] = mask
            }
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class SharpAIDomynedgeModel: Module, LLMModel {
    public let vocabularySize: Int
    public let configuration: SharpAIDomynedgeConfiguration
    @ModuleInfo var model: SharpAIDomynedgeModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIDomynedgeConfiguration) {
        self.vocabularySize = configuration.vocabularySize
        self.configuration = configuration
        self.model = SharpAIDomynedgeModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIDomynedgeSanitizer.sanitize(weights, configuration: configuration)
    }

    /// One cache per layer: a rotating window cache on sliding layers, a full cache otherwise.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layerSpecs.map { spec -> KVCache in
            if let window = spec.slidingWindow {
                return try makeSlidingWindowKVCache(parameters: parameters, window: window)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
