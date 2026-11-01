// SharpAI fork overlay: plamo3 forward pass. Not part of upstream mlx-swift-lm.
//
// Structure follows the reference sources: vLLM vllm/model_executor/models/plamo3.py and
// mlx-lm mlx_lm/models/plamo3.py. Points taken from them:
// - fused `qkv_proj` (q, k, v split) and fused `gate_up_proj` (first half is gate, SwiGLU), no biases
// - per-head `q_norm` / `k_norm` over head_dim, applied before RoPE
// - four norms per layer: pre/post mixer, pre/post mlp. Post-norm scales carry offsets 1/5 and
//   1/5^1.5; the pre norms, q/k norms and the final norm carry offset 1.0
// - full-attention layers use `rope_theta`, sliding-window layers use `rope_local_theta`
// - the checkpoint is not renamed (no key remapping in either reference)
//
// Not implemented: YaRN / scaled RoPE for full-attention layers (mlx-lm applies it only when a
// scaling factor is configured). The checkpoint key names are UNVERIFIED (gated repo), so the
// module keys follow the vLLM attribute names.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// RMSNorm whose effective scale is `weight + offset`. The weight starts at zero, so the
/// untrained scale equals the offset. The offset is applied at run time, as in mlx-lm.
final class SharpAIPlamo3RMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let offset: Float

    init(dimensions: Int, eps: Float, offset: Float) {
        self.eps = eps
        self.offset = offset
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight + offset, eps: eps)
    }
}

// MARK: - Attention

final class SharpAIPlamo3Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let qSize: Int
    let kvSize: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIPlamo3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIPlamo3RMSNorm

    init(_ config: SharpAIPlamo3Configuration, isSliding: Bool) {
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.headDim = config.resolvedHeadDim
        self.qSize = nHeads * headDim
        self.kvSize = nKVHeads * headDim
        self.scale = pow(Float(headDim), -0.5)

        let dim = config.hiddenSize
        self._qkvProj.wrappedValue = Linear(dim, qSize + 2 * kvSize, bias: false)
        self._oProj.wrappedValue = Linear(qSize, dim, bias: false)
        self._qNorm.wrappedValue = SharpAIPlamo3RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps, offset: 1.0)
        self._kNorm.wrappedValue = SharpAIPlamo3RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps, offset: 1.0)

        let base = isSliding ? config.ropeLocalTheta : config.ropeTheta
        self.rope = RoPE(dimensions: headDim, traditional: false, base: base)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = qkvProj(x)
        let queries = qkv[.ellipsis, ..<qSize].reshaped(B, L, nHeads, headDim)
        let keys = qkv[.ellipsis, qSize ..< (qSize + kvSize)].reshaped(B, L, nKVHeads, headDim)
        let values = qkv[.ellipsis, (qSize + kvSize)...].reshaped(B, L, nKVHeads, headDim)

        // Per-head QK norm happens before RoPE, as in vLLM.
        var q = qNorm(queries).transposed(0, 2, 1, 3)
        var k = kNorm(keys).transposed(0, 2, 1, 3)
        let v = values.transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
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

final class SharpAIPlamo3MLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateUp.wrappedValue = Linear(hiddenSize, 2 * intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    /// SwiGLU over the fused projection: the first half is the gate, the second half is up.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = gateUp(x).split(parts: 2, axis: -1)
        return down(silu(parts[0]) * parts[1])
    }
}

// MARK: - Decoder layer

final class SharpAIPlamo3DecoderLayer: Module {
    @ModuleInfo(key: "mixer") var mixer: SharpAIPlamo3Attention
    @ModuleInfo(key: "mlp") var mlp: SharpAIPlamo3MLP

    @ModuleInfo(key: "pre_mixer_norm") var preMixerNorm: SharpAIPlamo3RMSNorm
    @ModuleInfo(key: "post_mixer_norm") var postMixerNorm: SharpAIPlamo3RMSNorm
    @ModuleInfo(key: "pre_mlp_norm") var preMlpNorm: SharpAIPlamo3RMSNorm
    @ModuleInfo(key: "post_mlp_norm") var postMlpNorm: SharpAIPlamo3RMSNorm

    init(_ config: SharpAIPlamo3Configuration, isSliding: Bool, intermediateSize: Int) {
        let eps = config.rmsNormEps
        let dim = config.hiddenSize
        self._mixer.wrappedValue = SharpAIPlamo3Attention(config, isSliding: isSliding)
        self._mlp.wrappedValue = SharpAIPlamo3MLP(
            hiddenSize: dim, intermediateSize: intermediateSize)
        self._preMixerNorm.wrappedValue = SharpAIPlamo3RMSNorm(
            dimensions: dim, eps: eps, offset: 1.0)
        self._postMixerNorm.wrappedValue = SharpAIPlamo3RMSNorm(
            dimensions: dim, eps: eps, offset: 1.0 / 5.0)
        self._preMlpNorm.wrappedValue = SharpAIPlamo3RMSNorm(
            dimensions: dim, eps: eps, offset: 1.0)
        self._postMlpNorm.wrappedValue = SharpAIPlamo3RMSNorm(
            dimensions: dim, eps: eps, offset: Float(1.0 / pow(5.0, 1.5)))
        super.init()
    }

    /// Order: pre-norm, mixer, post-norm, residual add, pre-norm, MLP, post-norm, residual add.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = postMixerNorm(mixer(preMixerNorm(x), mask: mask, cache: cache))
        let r = x + h
        h = postMlpNorm(mlp(preMlpNorm(r)))
        return r + h
    }
}

// MARK: - Model

public class SharpAIPlamo3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIPlamo3DecoderLayer]
    let norm: SharpAIPlamo3RMSNorm
    let slidingWindow: Int?
    let layerTypes: [String]

    init(_ config: SharpAIPlamo3Configuration, intermediateSize: Int) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        let types = config.resolvedLayerTypes
        self.layerTypes = types
        self.layers = types.map { layerType in
            SharpAIPlamo3DecoderLayer(
                config,
                isSliding: layerType != SharpAIPlamo3Configuration.fullAttention,
                intermediateSize: intermediateSize)
        }
        self.norm = SharpAIPlamo3RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps, offset: 1.0)
        self.slidingWindow = config.slidingWindow
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let fullIdx = layerTypes.firstIndex(of: SharpAIPlamo3Configuration.fullAttention) ?? 0
        let slidingIdx = layerTypes.firstIndex(of: SharpAIPlamo3Configuration.slidingAttention) ?? 0
        let fullMask = createAttentionMask(h: h, cache: cache?[fullIdx])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingIdx], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask =
                layerTypes[i] == SharpAIPlamo3Configuration.fullAttention ? fullMask : slidingMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class SharpAIPlamo3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: SharpAIPlamo3ModelInner
    let configuration: SharpAIPlamo3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIPlamo3Configuration) throws {
        guard let intermediateSize = configuration.intermediateSize else {
            throw SharpAIPlamo3Error.invalidConfiguration("intermediate_size is required")
        }
        try configuration.validateModelConfiguration()
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0 ..< configuration.hiddenLayers).map { _ in configuration.kvHeads }
        self.model = SharpAIPlamo3ModelInner(configuration, intermediateSize: intermediateSize)
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
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        filterLMHeadWeights(
            from: SharpAIPlamo3Sanitizer.sanitize(weights),
            tiedWordEmbeddings: configuration.tieWordEmbeddings)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: model.slidingWindow,
                usesSlidingWindow: layerType != SharpAIPlamo3Configuration.fullAttention)
        }
    }
}

extension SharpAIPlamo3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
