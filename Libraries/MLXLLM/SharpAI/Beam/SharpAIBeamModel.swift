import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Ported from vLLM `vllm/models/beam/nvidia/model.py`.

/// Sandwich-norm decoder layer with depth-scaled residuals.
///
/// Per sublayer, the residual is `h1 = post_norm(sub(x)) * x_scale + h * r_scale`. The sublayer
/// input is `pre_norm(h1)` for the MLP. The layer input norm is `input_layernorm`.
final class SharpAIBeamDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIBeamAttention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    let attentionXScale: Float
    let attentionRScale: Float
    let mlpXScale: Float
    let mlpRScale: Float

    init(_ config: SharpAIBeamConfiguration, layerIndex: Int) {
        self._selfAttn.wrappedValue = SharpAIBeamAttention(config, layerIndex: layerIndex)
        if config.mlpLayerTypes[layerIndex] == "dense" {
            self.mlp = SharpAIBeamMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        } else {
            self.mlp = SharpAIBeamSparseMoE(config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        let attentionScales = sharpAIBeamDepthScales(
            layerIndex: layerIndex, isAttention: true)
        let mlpScales = sharpAIBeamDepthScales(layerIndex: layerIndex, isAttention: false)
        self.attentionXScale = attentionScales.x
        self.attentionRScale = attentionScales.r
        self.mlpXScale = mlpScales.x
        self.mlpRScale = mlpScales.r

        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attention = selfAttn(inputLayerNorm(hidden), mask: mask, cache: cache)
        let afterAttention =
            postAttentionLayerNorm(attention) * attentionXScale + hidden * attentionRScale
        let feedForward = mlp(preFeedforwardLayerNorm(afterAttention))
        return postFeedforwardLayerNorm(feedForward) * mlpXScale + afterAttention * mlpRScale
    }
}

final class SharpAIBeamModelInner: Module {
    let layerTypes: [String]
    let slidingWindow: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [SharpAIBeamDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIBeamConfiguration) {
        self.layerTypes = config.layerTypes
        self.slidingWindow = config.slidingWindow
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIBeamDecoderLayer(config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)

        // Each mask type reads from the first cache of that type.
        let fullIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
        let slidingIndex = layerTypes.firstIndex(of: "sliding_attention") ?? 0
        let fullMask = createAttentionMask(h: hidden, cache: cache?[fullIndex])
        let slidingMask = createAttentionMask(
            h: hidden, cache: cache?[slidingIndex], windowSize: slidingWindow)

        for (index, layer) in layers.enumerated() {
            let mask = layerTypes[index] == "full_attention" ? fullMask : slidingMask
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return norm(hidden)
    }
}

/// Beam causal LM. Untied LM head, optional final logit softcap.
public class SharpAIBeamForCausalLM: Module, LLMModel, LoRAModel {
    public let config: SharpAIBeamConfiguration

    @ModuleInfo var model: SharpAIBeamModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: SharpAIBeamConfiguration) {
        self.config = config
        self.model = SharpAIBeamModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var logits = lmHead(model(inputs, cache: cache))
        if let cap = config.finalLogitSoftcapping {
            // Softcap in fp32, then return to the head dtype.
            let capped = tanh(logits.asType(.float32) / cap) * cap
            logits = capped.asType(logits.dtype)
        }
        return logits
    }

    /// Full-attention layers get the standard cache. Sliding layers get a rotating cache
    /// of `sliding_window` entries.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        var caches = [KVCache]()
        caches.reserveCapacity(config.layerTypes.count)
        for layerType in config.layerTypes {
            if layerType == "full_attention" {
                caches.append(try makeAttentionKVCache(parameters: parameters))
            } else {
                caches.append(
                    try makeSlidingWindowKVCache(
                        parameters: parameters, window: config.slidingWindow))
            }
        }
        return caches
    }

    public var loraLayers: [Module] {
        model.layers
    }

    /// Renames NVFP4 tensors, drops rotary buffers, and stacks per-expert weights into `switch_mlp`.
    ///
    /// Checkpoint names are kept. Only the NVFP4 tensors and the expert layout change.
    /// UNVERIFIED: the quantized `switch_mlp` and the NVFP4 quantization config are not wired
    /// to the factory's quantize pass. Quantized expert loading is untested.
    /// UNVERIFIED: `switch_mlp`, attention and dense projections are plain modules with no
    /// `scales`, `global_scale` or `input_global_scale` parameters. Their uint32 `weight`
    /// shape also differs from `Linear.weight`. An NVFP4 shard fails the load on unused keys.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result = try SharpAIBeamNVFP4Checkpoint.normalize(weights)
        result = result.filter { key, _ in !key.contains("rotary_emb.inv_freq") }

        for layer in 0 ..< config.hiddenLayers where config.mlpLayerTypes[layer] == "sparse" {
            let prefix = "model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in [
                    "weight", "scales", "biases", "global_scale", "input_global_scale",
                ] {
                    guard result["\(prefix).experts.0.\(projection).\(suffix)"] != nil else {
                        continue
                    }
                    var parts = [MLXArray]()
                    parts.reserveCapacity(config.numExperts)
                    for expert in 0 ..< config.numExperts {
                        let key = "\(prefix).experts.\(expert).\(projection).\(suffix)"
                        guard let part = result.removeValue(forKey: key) else {
                            throw SharpAIBeamError("Missing expert tensor \(key).")
                        }
                        parts.append(part)
                    }
                    result["\(prefix).switch_mlp.\(projection).\(suffix)"] = stacked(parts)
                }
            }
        }
        return result
    }
}
