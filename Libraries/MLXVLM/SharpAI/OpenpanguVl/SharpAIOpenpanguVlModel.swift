// SharpAI fork overlay: openpangu_vl text model and causal LM. Not part of upstream mlx-swift-lm.
//
// Ports the text path of FreedomIntelligence/openPangu-VL-7B (modeling_openpangu_vl.py and
// modeling_openpangu_embedded.py, read from the Hugging Face repo on 2026-10-30):
// - pre-norm decoder layers, RMSNorm with eps = rms_norm_eps, residual adds
// - attention: GQA, q/k/v/o all carry bias when config "bias" is true (the checkpoint index has
//   self_attn.o_proj.bias for every layer), RoPE on q and k, scale head_dim^-0.5
// - MLP: down(silu(gate(x)) * up(x)), no bias
// - multimodal RoPE: interleaved axis selection when rope_scaling.mrope_interleaved is set,
//   otherwise contiguous mrope_section blocks
//
// Not implemented yet: the ViT (visual.* weights, dropped by SharpAIOpenpanguVlSanitizer) and
// image or video placeholder merging. Image and video input throws instead of being dropped.
// UNVERIFIED: the greedy order that interleaves the t/h/w axes is ported from a summary of
// get_mrope_interleaved_id_list, not from a run against a checkpoint. Text-only input is not
// affected, because all three position rows are equal there.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Multimodal RoPE

/// Port of get_mrope_interleaved_id_list(t, h, w, force_last=True): the axis (0 = t, 1 = h, 2 = w)
/// for each of the (t + h + w) rotary frequencies, greedily interleaved.
fileprivate func openPanguInterleavedAxes(_ tCount: Int, _ hCount: Int, _ wCount: Int) -> [Int] {
    let counts = [tCount - 1, hCount, wCount]
    var placed = [0, 0, 0]
    var remaining = counts
    var sequence = [Int]()
    var last: Int? = nil
    for _ in 0 ..< counts.reduce(0, +) {
        var candidates = (0 ..< 3).filter { remaining[$0] > 0 && $0 != last }
        if candidates.isEmpty {
            candidates = (0 ..< 3).filter { remaining[$0] > 0 }
        }
        // Candidates always have remaining > 0, so counts[axis] > 0 here.
        let best = candidates.min { lhs, rhs in
            let left = Double(placed[lhs]) / Double(counts[lhs])
            let right = Double(placed[rhs]) / Double(counts[rhs])
            return left != right ? left < right : lhs < rhs
        }!
        sequence.append(best)
        placed[best] += 1
        remaining[best] -= 1
        last = best
    }
    // force_last: one trailing temporal slot.
    sequence.append(0)
    return sequence
}

/// Axis for each of the head_dim / 2 rotary frequencies.
fileprivate func openPanguAxisForFrequency(
    sections: [Int], interleaved: Bool, half: Int
) -> [Int] {
    if sections.isEmpty {
        return Array(repeating: 0, count: half)
    }
    if interleaved {
        return openPanguInterleavedAxes(sections[0], sections[1], sections[2])
    }
    var axes = [Int]()
    for (axis, size) in sections.enumerated() {
        axes += Array(repeating: axis, count: size)
    }
    return axes
}

fileprivate func openPanguRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

/// Rotary tables for mRoPE. Position ids have shape [3, batch, length].
struct OpenPanguMRoPE {
    /// [head_dim / 2]
    let invFreq: MLXArray
    /// [3, head_dim / 2]: one-hot over the mRoPE axis that supplies each frequency.
    let selector: MLXArray

    init(headDim: Int, theta: Double, sections: [Int], interleaved: Bool) {
        let half = headDim / 2
        let freqIndices = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
        invFreq = 1.0 / pow(MLXArray(Float(theta)), freqIndices / Float(headDim))

        let axes = openPanguAxisForFrequency(sections: sections, interleaved: interleaved, half: half)
        var flat = [Float](repeating: 0, count: 3 * half)
        for (frequency, axis) in axes.enumerated() {
            flat[axis * half + frequency] = 1
        }
        selector = MLXArray(flat).reshaped(3, half)
    }

    /// cos and sin shaped [batch, 1, length, head_dim], ready to broadcast over heads.
    func cosSin(positionIds: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        let positions = positionIds.asType(.float32)
        // [3, batch, length, half]
        let frequencies = positions.expandedDimensions(axis: -1) * invFreq
        // Keep, for each frequency, only the axis chosen by the selector.
        let selected = (frequencies * selector.reshaped(3, 1, 1, -1)).sum(axis: 0)
        let emb = concatenated([selected, selected], axis: -1)
        let cosValues = MLX.cos(emb).asType(dtype).expandedDimensions(axis: 1)
        let sinValues = MLX.sin(emb).asType(dtype).expandedDimensions(axis: 1)
        return (cosValues, sinValues)
    }
}

// MARK: - Text decoder

class OpenPanguAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIOpenpanguVlConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

        queries = queries * cos + openPanguRotateHalf(queries) * sin
        keys = keys * cos + openPanguRotateHalf(keys) * sin

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

class OpenPanguMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: SharpAIOpenpanguVlConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class OpenPanguDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: OpenPanguAttention
    @ModuleInfo var mlp: OpenPanguMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIOpenpanguVlConfiguration) {
        _selfAttn.wrappedValue = OpenPanguAttention(config)
        _mlp.wrappedValue = OpenPanguMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache, cos: cos, sin: sin)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}

class OpenPanguTextBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [OpenPanguDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    let rope: OpenPanguMRoPE

    init(_ config: SharpAIOpenpanguVlConfiguration, rope: OpenPanguMRoPE) {
        self.rope = rope
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            OpenPanguDecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        super.init()
    }

    func callAsFunction(
        _ tokens: MLXArray, positionIds: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        var h = embedTokens(tokens)
        let (cos, sin) = rope.cosSin(positionIds: positionIds, dtype: h.dtype)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index], cos: cos, sin: sin)
        }
        return norm(h)
    }
}

// MARK: - Causal LM

public class SharpAIOpenpanguVlModel: Module, VLMModel {
    @ModuleInfo var model: OpenPanguTextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: SharpAIOpenpanguVlConfiguration

    public var vocabularySize: Int { config.vocabularySize }

    public var loraLayers: [Module] {
        [model]
    }

    public init(_ config: SharpAIOpenpanguVlConfiguration) throws {
        try config.validateModelConfiguration()
        let sections = config.mropeSection
        if !sections.isEmpty {
            guard sections.count == 3, sections.allSatisfy({ $0 > 0 }) else {
                throw ModelFactoryError.invalidConfiguration(
                    "openpangu_vl: mrope_section must have three positive entries")
            }
        }
        self.config = config
        let rope = OpenPanguMRoPE(
            headDim: config.headDim, theta: config.ropeTheta, sections: sections,
            interleaved: config.mropeInterleaved)
        _model.wrappedValue = OpenPanguTextBackbone(config, rope: rope)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try (0 ..< config.hiddenLayers).map { _ in
            try makeAttentionKVCache(parameters: parameters)
        }
    }

    /// Sequential positions, repeated across the three mRoPE rows. Text-only input has no
    /// image grid, so every row carries the same index.
    private func forward(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var ids = tokens
        if ids.ndim == 1 { ids = ids.expandedDimensions(axis: 0) }
        let batch = ids.dim(0)
        let length = ids.dim(1)
        let start = cache?.first?.offset ?? 0

        let row: [Int32] = (start ..< start + length).map { Int32($0) }
        var values = [Int32]()
        values.reserveCapacity(3 * batch * length)
        for _ in 0 ..< 3 {
            for _ in 0 ..< batch {
                values += row
            }
        }
        let positionIds = MLXArray(values).reshaped(3, batch, length)

        let hidden = model(ids, positionIds: positionIds, cache: cache)
        if config.tieWordEmbeddings {
            return model.embedTokens.asLinear(hidden)
        }
        return lmHead(hidden)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil else {
            throw SharpAIOpenpanguVlError.visionNotImplemented
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

    /// Drops the vision weights and the recomputed rotary tables, and unties lm_head when the
    /// config ties the word embeddings.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let untied = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: config.tieWordEmbeddings)
        let sanitized = try SharpAIOpenpanguVlSanitizer.sanitize(untied)
        return sanitized.filter { !$0.key.contains("rotary_emb") }
    }
}
