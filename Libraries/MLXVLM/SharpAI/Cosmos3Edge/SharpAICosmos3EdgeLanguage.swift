// SharpAI fork overlay: cosmos3_edge text decoder. Not part of upstream mlx-swift-lm.
//
// Structure follows the HF transformers `Cosmos3EdgeTextModel` (28 pre-norm decoder layers,
// GQA attention with q/k/v/o projections, ReLU-squared MLP with fc1/fc2, RMSNorm, interleaved
// multimodal RoPE with `mrope_section`, rope_theta from `rope_parameters`). Sources:
// https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cosmos3_edge/modeling_cosmos3_edge.py
// https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cosmos3_edge/configuration_cosmos3_edge.py
//
// UNVERIFIED:
// - The exact index layout of the interleaved MRoPE (which frequency slots take the H and W
//   positions). It follows the Qwen-style `mrope_section[d] * 3` slot pattern described for the
//   HF `recomposition_frequencies`. Text-only inputs (all three position axes equal) do not
//   depend on it.
// - Checkpoint key names for this tower. The nvidia/Cosmos3-Edge transformer shards use
//   diffusers-style names (to_q, to_out, up_proj), which are not the names used here.
// - The vision tower, projector and processor are not written in this file.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// ReLU-squared, the `hidden_act` of the text tower.
private func cosmos3EdgeReluSquared(_ x: MLXArray) -> MLXArray {
    let r = MLX.maximum(x, MLXArray(Float(0)))
    return r * r
}

/// Rotate-half RoPE on [B, H, T, D] with cos/sin of shape [B, 1, T, D].
private func cosmos3EdgeApplyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let parts = split(x, parts: 2, axis: -1)
    let rotated = MLX.concatenated([MLX.negative(parts[1]), parts[0]], axis: -1)
    return x * cos + rotated * sin
}

final class SharpAICosmos3EdgeTextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAICosmos3EdgeTextConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        let bias = config.attentionBias
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: bias)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: bias)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: bias)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

        q = cosmos3EdgeApplyRotary(q, cos: cos, sin: sin)
        k = cosmos3EdgeApplyRotary(k, cos: cos, sin: sin)

        // Grouped-query attention: repeat each kv head for its group of query heads.
        if heads != kvHeads {
            let repeats = heads / kvHeads
            k = MLX.repeated(k, count: repeats, axis: 1)
            v = MLX.repeated(v, count: repeats, axis: 1)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAICosmos3EdgeTextMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: SharpAICosmos3EdgeTextConfiguration) {
        _fc1.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _fc2.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(cosmos3EdgeReluSquared(fc1(x)))
    }
}

final class SharpAICosmos3EdgeTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAICosmos3EdgeTextAttention
    let mlp: SharpAICosmos3EdgeTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAICosmos3EdgeTextConfiguration) {
        _attention.wrappedValue = SharpAICosmos3EdgeTextAttention(config)
        self.mlp = SharpAICosmos3EdgeTextMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), cos: cos, sin: sin, mask: mask)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// Embedding, decoder stack and final norm (HF `Cosmos3EdgeTextModel`).
final class SharpAICosmos3EdgeTextBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAICosmos3EdgeTextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    /// Inverse frequencies, length head_dim / 2. Kept as plain values so they are not parameters.
    let inverseFrequencies: [Float]
    /// Shape [3 * head_dim / 2] flattened: 1 where a frequency slot takes the axis's position.
    let axisWeights: [Float]
    let halfDim: Int

    init(_ config: SharpAICosmos3EdgeTextConfiguration, ropeTheta: Double, mropeSection: [Int]) {
        let half = config.headDim / 2
        self.halfDim = half
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAICosmos3EdgeTextDecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.inverseFrequencies = (0 ..< half).map { j in
            Float(1.0 / pow(ropeTheta, Double(2 * j) / Double(config.headDim)))
        }

        // Interleaved MRoPE: H takes slots 1, 4, 7, ... below 3 * section[1]; W takes slots
        // 2, 5, 8, ... below 3 * section[2]; T takes every other slot.
        var weights = [Float](repeating: 0, count: 3 * half)
        for j in 0 ..< half {
            var axis = 0
            if j % 3 == 1 && j < 3 * mropeSection[1] {
                axis = 1
            } else if j % 3 == 2 && j < 3 * mropeSection[2] {
                axis = 2
            }
            weights[axis * half + j] = 1
        }
        self.axisWeights = weights
    }

    /// cos and sin, each [1 or B, 1, T, D], for text positions 0 ..< length (identical on all axes).
    func rotary(batch: Int, length: Int) -> (MLXArray, MLXArray) {
        let base = MLXArray((0 ..< length).map { Float($0) }).reshaped(1, 1, length)
        let positions = broadcast(base, to: [3, batch, length])
        let inverse = MLXArray(inverseFrequencies)
        let weights = MLXArray(axisWeights).reshaped(3, 1, 1, halfDim)
        let angles = positions.expandedDimensions(axis: -1) * inverse  // [3, B, T, half]
        let selected = (angles * weights).sum(axis: 0)  // [B, T, half]
        let emb = MLX.concatenated([selected, selected], axis: -1)  // [B, T, D]
        return (
            MLX.cos(emb).expandedDimensions(axis: 1),
            MLX.sin(emb).expandedDimensions(axis: 1)
        )
    }

    /// Additive causal mask [T, T]: 0 where a query may attend, -inf elsewhere.
    func causalMask(length: Int) -> MLXArray {
        let idx = MLXArray((0 ..< length).map { Float($0) })
        let allowed = greaterEqual(idx.reshaped(length, 1), idx.reshaped(1, length))
        return MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let (batch, length) = (inputIds.dim(0), inputIds.dim(1))
        var h = embedTokens(inputIds)
        let (cos, sin) = rotary(batch: batch, length: length)
        let mask = causalMask(length: length)
        for layer in layers {
            h = layer(h, cos: cos, sin: sin, mask: mask)
        }
        return norm(h)
    }
}

/// Text-only cosmos3_edge language model: backbone plus untied `lm_head` (HF `tie_word_embeddings`
/// is false). Checkpoint loading and the vision path are not wired here.
public class SharpAICosmos3EdgeTextForCausalLM: Module {
    @ModuleInfo var model: SharpAICosmos3EdgeTextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: SharpAICosmos3EdgeTextConfiguration) throws {
        guard config.hiddenAct == "relu2" else {
            throw ModelFactoryError.invalidConfiguration(
                "cosmos3_edge text: hidden_act must be relu2, got \(config.hiddenAct)")
        }
        guard let theta = config.ropeTheta, let section = config.mropeSection else {
            throw ModelFactoryError.invalidConfiguration(
                "cosmos3_edge text: rope_parameters must give rope_theta and mrope_section")
        }
        guard config.headDim % 2 == 0, section.count == 3,
            section.allSatisfy({ $0 >= 0 }), section.reduce(0, +) == config.headDim / 2
        else {
            throw ModelFactoryError.invalidConfiguration(
                "cosmos3_edge text: mrope_section must have 3 non-negative entries summing to head_dim / 2"
            )
        }
        _model.wrappedValue = SharpAICosmos3EdgeTextBackbone(
            config, ropeTheta: theta, mropeSection: section)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    /// Logits [B, T, vocab] for token ids [B, T].
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        lmHead(model(inputIds))
    }
}
