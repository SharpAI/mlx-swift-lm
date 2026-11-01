// SharpAI fork overlay: step5_vl dense text decoder. Not part of upstream mlx-swift-lm.
//
// Port of the dense text path of NVIDIA NeMo Automodel's step5 model, read 2026-10-30 from
// https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/nemo_automodel/components/models/step5/
// (model.py, layers.py, rope.py, configuration.py):
// - pre-norm decoder: h = x + attn(norm(x)); out = h + mlp(norm(h))
// - RMSNorm: the reference computes normalized * (weight + 1). The checkpoint stores weight, so
//   the sanitizer adds 1 to every `norm.weight` and the module applies a plain RMSNorm.
// - attention: GQA, bias-free q/k/v/o, q_norm and k_norm before RoPE, scale head_dim^-0.5
// - partial half-split RoPE over int(head_dim * partial_rotary_factor) leading dims, theta per layer.
//   Defaults when absent: factor 1/3 on full_attention layers and 1 on sliding layers.
// - head-wise output gate: attended * sigmoid(g_proj(x)) (on by default, as in the reference config)
// - sliding_attention layers attend to the last `sliding_window` tokens (default 512). Full layers
//   attend causally. layer_types defaults to "every fourth layer is full" when absent.
// - dense MLP: down(clip(silu(gate), max: L) * clip(up, -L, L)), L = swiglu_limits[layer] (default 7)
// - lm_head is untied, as the reference requires.
//
// Not implemented. SharpAIStep5VlSupport.requireDenseText throws `unsupported` for these, so the
// model is never built for a checkpoint that needs them:
// - the CSA sparse indexer (sparse_config.enabled)
// - MoE layers (use_moe)
// - the perception_encoder vision tower and projector. Image and video input throws in prepare().
// - MTP layers (num_nextn_predict_layers). The sanitizer drops keys at layer index >= hidden layers.
//
// Deviations from the reference, because they only matter for absent config keys:
// - use_moe absent means dense. The reference Python default is True. A full checkpoint always
//   writes the key, and the tiny forward-pass test config omits it.
// - rope_scaling (llama3, factor 1.0 in the fixture) is not applied. With factor 1.0 that
//   scaling is the identity.
// - rope_theta absent falls back to 10000 for every layer. The reference uses 1e7 on full layers.
//
// UNVERIFIED against a real step5_vl checkpoint: the expert key layout and the weight names of the
// dense MLP are assumed to follow the Hugging Face names above. The tiny forward-pass test is the
// only check so far.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Config helpers

/// Picks `values[index]`, or `fallback` when the array is empty or short.
fileprivate func step5Pick<T>(_ values: [T], _ index: Int, _ fallback: T) -> T {
    values.indices.contains(index) ? values[index] : fallback
}

/// True for a `sliding_attention` layer. Absent `layer_types` means every fourth layer is full.
func sharpAIStep5VlLayerIsSliding(_ config: SharpAIStep5VlConfiguration, _ index: Int) -> Bool {
    if let types = config.textConfig.layerTypes, types.indices.contains(index) {
        return types[index] == "sliding_attention"
    }
    return index % 4 != 3
}

/// Rotary span for one layer, `int(head_dim * partial_rotary_factor)`.
func sharpAIStep5VlRopeDims(_ config: SharpAIStep5VlConfiguration, _ index: Int) -> Int {
    let text = config.textConfig
    let defaultFactor: Float = sharpAIStep5VlLayerIsSliding(config, index) ? 1.0 : 1.0 / 3.0
    let factor = step5Pick(text.partialRotaryFactors ?? [], index, defaultFactor)
    return Int(Float(text.headDim) * factor)
}

/// Rejects the parts of the architecture this overlay does not implement yet, and any rotary
/// span the rotary layer cannot apply. Runs before any module is allocated.
public enum SharpAIStep5VlSupport {
    public static func requireDenseText(_ configuration: SharpAIStep5VlConfiguration) throws {
        let text = configuration.textConfig
        if text.sparseConfig?.enabled == true {
            throw SharpAIStep5VlError.unsupported(
                "CSA sparse indexer (sparse_config.enabled) is not implemented")
        }
        if text.useMoe == true {
            throw SharpAIStep5VlError.unsupported(
                "MoE layers (use_moe) are not implemented in this overlay")
        }
        for index in 0 ..< text.hiddenLayers {
            let span = sharpAIStep5VlRopeDims(configuration, index)
            guard span > 0, span % 2 == 0 else {
                throw SharpAIStep5VlError.invalidConfiguration(
                    "rotary span \(span) at layer \(index) must be a positive even number")
            }
        }
    }
}

// MARK: - Attention

final class SharpAIStep5VlAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    /// Head-wise output gate. Nil when `use_head_wise_attn_gate` is false.
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    let rope: RoPELayer

    init(_ configuration: SharpAIStep5VlConfiguration, layerIndex: Int) {
        let text = configuration.textConfig
        let heads = text.attentionHeads
        let kvHeads = text.attentionGroups
        let dim = text.headDim

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = dim
        self.scale = pow(Float(dim), -0.5)

        _qProj.wrappedValue = Linear(text.hiddenSize, heads * dim, bias: false)
        _kProj.wrappedValue = Linear(text.hiddenSize, kvHeads * dim, bias: false)
        _vProj.wrappedValue = Linear(text.hiddenSize, kvHeads * dim, bias: false)
        _oProj.wrappedValue = Linear(heads * dim, text.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: dim, eps: text.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: dim, eps: text.rmsNormEps)
        let useGate = text.useHeadWiseAttnGate ?? true
        _gProj.wrappedValue = useGate ? Linear(text.hiddenSize, heads, bias: false) : nil

        let theta = step5Pick(text.ropeTheta, layerIndex, text.ropeTheta.first ?? 10_000)
        self.rope = initializeRope(
            dims: sharpAIStep5VlRopeDims(configuration, layerIndex),
            base: theta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: nil
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var queries = qNorm(qProj(x).reshaped(batch, length, numHeads, headDim))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, numKVHeads, headDim))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        // B, H, L, D -> B, L, H, D so the head-wise gate broadcasts over D.
        output = output.transposed(0, 2, 1, 3)
        if let gProj {
            output = output * sigmoid(gProj(x)).expandedDimensions(axis: -1)
        }
        return oProj(output.reshaped(batch, length, -1))
    }
}

// MARK: - MLP

/// SwiGLU MLP with the reference's asymmetric clamp when `limit` is positive.
final class SharpAIStep5VlMLP: Module {
    let limit: Float?

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, limit: Float) {
        self.limit = limit > 0 ? limit : nil
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let limit {
            let gate = clip(silu(gateProj(x)), max: limit)
            let up = clip(upProj(x), min: -limit, max: limit)
            return downProj(gate * up)
        }
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer

final class SharpAIStep5VlDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIStep5VlAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIStep5VlMLP

    init(_ configuration: SharpAIStep5VlConfiguration, layerIndex: Int) {
        let text = configuration.textConfig
        _selfAttn.wrappedValue = SharpAIStep5VlAttention(configuration, layerIndex: layerIndex)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _mlp.wrappedValue = SharpAIStep5VlMLP(
            hiddenSize: text.hiddenSize,
            intermediateSize: text.intermediateSize,
            limit: step5Pick(text.swigluLimits ?? [], layerIndex, 7.0))
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Backbone

final class SharpAIStep5VlBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [SharpAIStep5VlDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let slidingWindow: Int
    /// Per-layer flag, true for `sliding_attention`.
    let slidingFlags: [Bool]
    /// First full-attention layer, used for the causal mask.
    let fullIndex: Int
    /// First sliding-attention layer, if any.
    let slidingIndex: Int?

    init(_ configuration: SharpAIStep5VlConfiguration) {
        let text = configuration.textConfig
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: text.vocabSize, dimensions: text.hiddenSize)
        self.layers = (0 ..< text.hiddenLayers).map {
            SharpAIStep5VlDecoderLayer(configuration, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)

        let flags = (0 ..< text.hiddenLayers).map {
            sharpAIStep5VlLayerIsSliding(configuration, $0)
        }
        self.slidingWindow = text.slidingWindow ?? 512
        self.slidingFlags = flags
        self.fullIndex = flags.firstIndex(of: false) ?? 0
        self.slidingIndex = flags.firstIndex(of: true)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: cache?[fullIndex])
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if let slidingIndex {
            slidingMask = createAttentionMask(
                h: h, cache: cache?[slidingIndex], windowSize: slidingWindow)
        }

        for (index, layer) in layers.enumerated() {
            h = layer(
                h,
                mask: slidingFlags[index] ? slidingMask : fullMask,
                cache: cache?[index])
        }
        return norm(h)
    }
}

// MARK: - Causal LM

public class SharpAIStep5VlModel: Module, LanguageModel {
    public let config: SharpAIStep5VlConfiguration

    @ModuleInfo var model: SharpAIStep5VlBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public var vocabularySize: Int { config.textConfig.vocabSize }

    public init(_ configuration: SharpAIStep5VlConfiguration) throws {
        try configuration.validateModelConfiguration()
        try SharpAIStep5VlSupport.requireDenseText(configuration)
        self.config = configuration
        _model.wrappedValue = SharpAIStep5VlBackbone(configuration)
        _lmHead.wrappedValue = Linear(
            configuration.textConfig.hiddenSize, configuration.textConfig.vocabSize, bias: false)
        super.init()
    }

    /// One cache per layer. Sliding layers get a rotating window, full layers an attention cache.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let window = config.textConfig.slidingWindow ?? 512
        return try (0 ..< config.textConfig.hiddenLayers).map { (index: Int) throws -> KVCache in
            if sharpAIStep5VlLayerIsSliding(config, index) {
                return try makeSlidingWindowKVCache(parameters: parameters, window: window)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    private func forward(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var ids = tokens
        if ids.ndim == 1 { ids = ids.expandedDimensions(axis: 0) }
        return lmHead(model(ids, cache: cache))
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil else {
            throw SharpAIStep5VlError.unsupported(
                "image and video input need the perception_encoder vision tower, which is not implemented")
        }
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }

        let totalTokens = tokens.dim(1)
        let processed = try prefill.forEachChunk(total: totalTokens) { range in
            _ = forward(tokens[0..., range], cache: cache)
            eval(cache)
        }

        let logits = forward(tokens[0..., processed...], cache: cache)
        prefill.progress?(totalTokens, totalTokens)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        LMOutput(logits: forward(input.tokens, cache: cache))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache)
    }

    /// Keeps the text backbone and lm_head. Drops the vision tower and projector, MTP layers
    /// (layer index >= num_hidden_layers), and the precomputed rotary tables. Adds 1 to every
    /// `norm.weight`, because the reference applies `weight + 1`.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let hiddenLayers = config.textConfig.hiddenLayers
        var result: [String: MLXArray] = [:]
        for (key, value) in SharpAIStep5VlSanitize.sanitize(weights: weights) {
            if key.hasPrefix("vision_model.") || key.hasPrefix("vit_large_projector.") {
                continue
            }
            if key.contains("mtp") || key.contains("shared_head") {
                continue
            }
            if let layer = step5LayerIndex(of: key), layer >= hiddenLayers {
                continue
            }
            result[key] = key.hasSuffix("norm.weight") ? value + Float(1) : value
        }
        return result
    }
}

/// Layer index for `model.layers.<n>.*` keys, or nil for other keys.
fileprivate func step5LayerIndex(of key: String) -> Int? {
    let parts = key.split(separator: ".")
    guard parts.count > 2, parts[0] == "model", parts[1] == "layers" else { return nil }
    return Int(parts[2])
}
