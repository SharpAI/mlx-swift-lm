// SharpAI fork overlay: olmo_hybrid model. Not part of upstream mlx-swift-lm.
//
// Port of OlmoHybridForCausalLM in transformers' modeling_olmo_hybrid.py.
// Linear-attention layers use pre-norm (input_layernorm, post_attention_layernorm). Full-attention layers
// have no input norm and apply post_attention_layernorm and post_feedforward_layernorm to the sub-block
// outputs. Attention applies RMSNorm over the full q and k projections, before the head split.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Full attention

final class SharpAIOlmoHybridAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAIOlmoHybridConfiguration) {
        nHeads = config.attentionHeads
        nKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)

        let dims = config.headDim
        rope = config.ropeTheta.map { (base: Float) -> RoPELayer in
            RoPE(dimensions: dims, traditional: false, base: base)
        }

        let hidden = config.hiddenSize
        _qProj.wrappedValue = Linear(
            hidden, config.attentionHeads * config.headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            hidden, config.kvHeads * config.headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            hidden, config.kvHeads * config.headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, hidden, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(
            dimensions: config.attentionHeads * config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(
            dimensions: config.kvHeads * config.headDim, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x))
        var keys = kNorm(kProj(x))
        let values = vProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        let valueHeads = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        // NoPE when the checkpoint has no rope_parameters.
        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: valueHeads,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MLP

final class SharpAIOlmoHybridMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: SharpAIOlmoHybridConfiguration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer

/// One layer of either kind. Only the attention or the linear block is set, as in Qwen3Next.
final class SharpAIOlmoHybridDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIOlmoHybridAttention?
    @ModuleInfo(key: "linear_attn") var linearAttn: SharpAIOlmoHybridGatedDeltaNet?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm?
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm?
    @ModuleInfo(key: "mlp") var mlp: SharpAIOlmoHybridMLP

    init(_ config: SharpAIOlmoHybridConfiguration, isLinear: Bool) {
        self.isLinear = isLinear

        _selfAttn.wrappedValue =
            isLinear ? nil : SharpAIOlmoHybridAttention(config)
        _linearAttn.wrappedValue =
            isLinear ? SharpAIOlmoHybridGatedDeltaNet(config) : nil
        _inputLayerNorm.wrappedValue =
            isLinear ? RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) : nil
        _postFeedforwardLayerNorm.wrappedValue =
            isLinear ? nil : RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = SharpAIOlmoHybridMLP(config)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        if let linearAttn, let inputLayerNorm {
            let h = x + linearAttn(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
            return h + mlp(postAttentionLayerNorm(h))
        }

        guard let selfAttn, let postFeedforwardLayerNorm else {
            fatalError("olmo_hybrid layer has neither an attention nor a linear block")
        }
        let h = x + postAttentionLayerNorm(selfAttn(x, mask: attentionMask, cache: cache))
        return h + postFeedforwardLayerNorm(mlp(h))
    }
}

// MARK: - Model

final class SharpAIOlmoHybridModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIOlmoHybridDecoderLayer]
    let norm: RMSNorm
    let fullAttentionIndex: Int
    let linearAttentionIndex: Int

    init(_ config: SharpAIOlmoHybridConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        layers = config.layerTypes.map { layerType in
            SharpAIOlmoHybridDecoderLayer(config, isLinear: layerType == "linear_attention")
        }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        linearAttentionIndex = config.layerTypes.firstIndex(of: "linear_attention") ?? 0

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let attentionMask = createAttentionMask(h: h, cache: cache?[fullAttentionIndex])
        let ssmMask = createSSMMask(
            h: h, cache: cache?[linearAttentionIndex] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class SharpAIOlmoHybridModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let model: SharpAIOlmoHybridModelInner
    let configuration: SharpAIOlmoHybridConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SharpAIOlmoHybridConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = SharpAIOlmoHybridModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
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
        return out
    }

    /// Linear layers get a MambaCache. Full-attention layers get the standard attention cache.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        var caches = [KVCache]()
        for layerType in configuration.layerTypes {
            if layerType == "linear_attention" {
                caches.append(MambaCache())
            } else {
                caches.append(try makeAttentionKVCache(parameters: parameters))
            }
        }
        return caches
    }

    public func makeCache() -> [KVCache] {
        var caches = [KVCache]()
        for layerType in configuration.layerTypes {
            if layerType == "linear_attention" {
                caches.append(MambaCache())
            } else {
                caches.append(KVCacheSimple())
            }
        }
        return caches
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIOlmoHybridSanitize.sanitize(
            weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)
    }
}

extension SharpAIOlmoHybridModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
