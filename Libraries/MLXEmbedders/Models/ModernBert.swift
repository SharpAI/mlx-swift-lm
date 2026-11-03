// SharpAI fork addition: ModernBERT encoder for the embedder registry. Not part of upstream.
//
// Ported from the SL-47 LLM overlay (Libraries/MLXLLM/SharpAI/Modernbert), which MLXEmbedders
// cannot import: MLXLLM is a separate target and that overlay's model is an `LLMModel`, not an
// `EmbeddingModel`. The overlay's `modernbert` creator is deliberately NOT in `llmWiredKeys`.
//
// Module tree and weight names follow HF ModernBertModel:
//   model.embeddings.tok_embeddings, model.embeddings.norm,
//   model.layers.N.attn.Wqkv, model.layers.N.attn.Wo, model.layers.N.attn_norm (absent in layer 0),
//   model.layers.N.mlp_norm, model.layers.N.mlp.Wi, model.layers.N.mlp.Wo, model.final_norm.
//
// Layers are pre-norm and bidirectional. Layer N is global when N % global_attn_every_n_layers
// is 0; other layers attend within a local window |i - j| <= local_attention / 2.
//
// UNVERIFIED against a reference implementation (no checkpoint, no Python, air-gapped): layer 0
// has no attn_norm; Wi output is split (activation input, gate) with GELU on the first half; the
// RoPE form is rotate-half with a theta chosen per layer type. A ColBERT projection head
// (pylate `linear`) is not implemented, so its keys are dropped.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct ModernBertConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var vocabularySize: Int
    public var normEps: Float = 1e-5
    public var normBias: Bool = false
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false
    public var hiddenActivation: String = "gelu"
    public var localAttention: Int = 128
    public var globalAttnEveryNLayers: Int = 3
    public var globalRopeTheta: Float = 160_000
    public var localRopeTheta: Float = 10_000
    public var maxPositionEmbeddings: Int?
    /// HF `classifier_pooling`: "cls" (default) or "mean". Overridden by `1_Pooling/config.json`.
    public var classifierPooling: String = "cls"

    var headDimensions: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case vocabularySize = "vocab_size"
        case normEps = "norm_eps"
        case normBias = "norm_bias"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case hiddenActivation = "hidden_activation"
        case localAttention = "local_attention"
        case globalAttnEveryNLayers = "global_attn_every_n_layers"
        case globalRopeTheta = "global_rope_theta"
        case localRopeTheta = "local_rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case classifierPooling = "classifier_pooling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? normEps
        normBias = try c.decodeIfPresent(Bool.self, forKey: .normBias) ?? normBias
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? mlpBias
        hiddenActivation =
            try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? hiddenActivation
        localAttention = try c.decodeIfPresent(Int.self, forKey: .localAttention) ?? localAttention
        globalAttnEveryNLayers =
            try c.decodeIfPresent(Int.self, forKey: .globalAttnEveryNLayers)
            ?? globalAttnEveryNLayers
        globalRopeTheta =
            try c.decodeIfPresent(Float.self, forKey: .globalRopeTheta) ?? globalRopeTheta
        localRopeTheta =
            try c.decodeIfPresent(Float.self, forKey: .localRopeTheta) ?? localRopeTheta
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        classifierPooling =
            try c.decodeIfPresent(String.self, forKey: .classifierPooling) ?? classifierPooling
        try validate()
    }

    /// Reject configurations the encoder cannot build, before any weights are allocated.
    func validate() throws {
        func bad(_ message: String) -> DecodingError {
            .dataCorrupted(.init(codingPath: [], debugDescription: "modernbert: \(message)"))
        }
        let positive: [(String, Int)] = [
            ("hidden_size", hiddenSize), ("num_hidden_layers", hiddenLayers),
            ("intermediate_size", intermediateSize), ("num_attention_heads", attentionHeads),
            ("vocab_size", vocabularySize), ("local_attention", localAttention),
            ("global_attn_every_n_layers", globalAttnEveryNLayers),
        ]
        for (name, value) in positive where value <= 0 { throw bad("\(name) must be positive") }
        if hiddenSize % attentionHeads != 0 {
            throw bad("hidden_size must be divisible by num_attention_heads")
        }
        if headDimensions % 2 != 0 { throw bad("head dimension must be even for rotary") }
        if hiddenActivation != "gelu" {
            throw bad("hidden_activation \(hiddenActivation) is not supported; only gelu")
        }
    }
}

// MARK: - Layers

private final class ModernBertAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ config: ModernBertConfiguration, ropeTheta: Float) {
        heads = config.attentionHeads
        headDim = config.headDimensions
        scale = pow(Float(headDim), -0.5)
        _wqkv.wrappedValue = Linear(
            config.hiddenSize, 3 * config.hiddenSize, bias: config.attentionBias)
        _wo.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.attentionBias)
        rope = initializeRope(
            dims: headDim, base: ropeTheta, traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray
    {
        let (B, L) = (x.dim(0), x.dim(1))
        let parts = wqkv(x).reshaped(B, L, 3, heads, headDim).split(parts: 3, axis: 2)
        var queries = parts[0].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: nil)
        keys = applyRotaryPosition(rope, to: keys, offset: nil)
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: nil, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return wo(output)
    }
}

private final class ModernBertMLP: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ config: ModernBertConfiguration) {
        _wi.wrappedValue = Linear(
            config.hiddenSize, 2 * config.intermediateSize, bias: config.mlpBias)
        _wo.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    /// GeGLU: the first half goes through GELU and multiplies the second half.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let halves = wi(x).split(parts: 2, axis: -1)
        return wo(gelu(halves[0]) * halves[1])
    }
}

private final class ModernBertLayer: Module {
    let isGlobal: Bool
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm?
    @ModuleInfo(key: "attn") var attn: ModernBertAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: ModernBertMLP

    init(_ config: ModernBertConfiguration, layerIndex: Int) {
        let global = layerIndex % config.globalAttnEveryNLayers == 0
        isGlobal = global
        _attnNorm.wrappedValue =
            layerIndex == 0
            ? nil
            : LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps, affine: true,
                bias: config.normBias)
        _attn.wrappedValue = ModernBertAttention(
            config, ropeTheta: global ? config.globalRopeTheta : config.localRopeTheta)
        _mlpNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, affine: true, bias: config.normBias)
        _mlp.wrappedValue = ModernBertMLP(config)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray
    {
        let h = x + attn(attnNorm.map { $0(x) } ?? x, mask: mask)
        return h + mlp(mlpNorm(h))
    }
}

private final class ModernBertEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: ModernBertConfiguration) {
        _tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, affine: true, bias: config.normBias)
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray { norm(tokEmbeddings(inputs)) }
}

private final class ModernBertInner: Module {
    @ModuleInfo(key: "embeddings") var embeddings: ModernBertEmbeddings
    let layers: [ModernBertLayer]
    let localHalfWindow: Int
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(_ config: ModernBertConfiguration) {
        _embeddings.wrappedValue = ModernBertEmbeddings(config)
        layers = (0 ..< config.hiddenLayers).map { ModernBertLayer(config, layerIndex: $0) }
        localHalfWindow = config.localAttention / 2
        _finalNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, affine: true, bias: config.normBias)
    }

    /// `padding` is an optional Bool mask [B, 1, 1, L]; true marks a real token.
    func callAsFunction(_ inputs: MLXArray, padding: MLXArray?) -> MLXArray {
        var h = embeddings(inputs)
        let length = h.dim(1)
        let positions = MLXArray((0 ..< length).map { Int32($0) })
        let distance = (positions.reshaped(length, 1) - positions.reshaped(1, length)).abs()
        let window = (distance .<= Int32(localHalfWindow)).reshaped(1, 1, length, length)

        let globalMode: MLXFast.ScaledDotProductAttentionMaskMode
        let localMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let padding {
            globalMode = .array(padding)
            localMode = .array(window & padding)
        } else {
            globalMode = .none
            localMode = .array(window)
        }
        for layer in layers { h = layer(h, mask: layer.isGlobal ? globalMode : localMode) }
        return finalNorm(h)
    }
}

// MARK: - Model

/// Bidirectional ModernBERT encoder for `model_type: "modernbert"`.
public class ModernBertModel: Module, EmbeddingModel {
    public let vocabularySize: Int
    public let poolingStrategy: Pooling.Strategy?

    public var maxPositionEmbeddings: Int? { _maxPositionEmbeddings }
    private let _maxPositionEmbeddings: Int?

    @ModuleInfo(key: "model") private var model: ModernBertInner

    public init(_ config: ModernBertConfiguration) {
        vocabularySize = config.vocabularySize
        _maxPositionEmbeddings = config.maxPositionEmbeddings
        poolingStrategy = config.classifierPooling == "mean" ? .mean : .cls
        _model.wrappedValue = ModernBertInner(config)
    }

    /// `attentionMask` is [B, L], nonzero for real tokens; padding keys are excluded in every
    /// layer. `positionIds` and `tokenTypeIds` are ignored (RoPE, no segment embeddings).
    public func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        let padding = attentionMask.map {
            $0.asType(.bool).reshaped(inputs.dim(0), 1, 1, inputs.dim(1))
        }
        return EmbeddingModelOutput(
            hiddenStates: model(inputs, padding: padding), pooledOutput: nil)
    }

    /// Adds the `model.` prefix that bare `ModernBertModel` checkpoints omit and drops the
    /// masked-LM head, rotary buffers and the (unimplemented) ColBERT `linear.` projection.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        let dropped = ["head.", "decoder.", "model.head.", "model.decoder.", "linear."]
        for (key, value) in weights {
            if key.contains("rotary_emb") || key.contains("position_ids") { continue }
            if dropped.contains(where: { key.hasPrefix($0) }) { continue }
            if key.hasPrefix("embeddings.") || key.hasPrefix("layers.")
                || key.hasPrefix("final_norm.")
            {
                result["model." + key] = value
            } else {
                result[key] = value
            }
        }
        return result
    }
}
