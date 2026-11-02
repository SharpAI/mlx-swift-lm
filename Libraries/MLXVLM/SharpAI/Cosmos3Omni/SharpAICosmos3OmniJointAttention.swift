// SharpAI fork overlay: cosmos3_omni two-way joint attention. Not part of upstream mlx-swift-lm.
//
// Rule ("joint_attn_implementation": "two_way"), one-way conditioning:
//   - understanding tokens attend causally to understanding tokens only;
//   - generation tokens attend bidirectionally to [understanding K/V ; generation K/V].
// RoPE is applied separately per tower, each with its own position ids. The understanding K that
// generation tokens see is the post-norm, post-RoPE K (no extra norm: `use_und_k_norm_for_gen`
// only takes effect when `qk_norm_for_text` is false, and Cosmos3-Nano sets it true).
// Verified against: https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_cosmos3.py
// (Cosmos3AttnProcessor.__call__, Cosmos3VLTextMoTDecoderLayer.forward, Cosmos3VLTextRotaryEmbedding).
//
// The understanding layers are the upstream Qwen3VL language layers, used as-is. Qwen3VL keeps its
// language model in a `private` property, so the layers are reached through the Module tree
// (`namedModules()`), which needs no edit to Qwen3VL.swift.
//
// UNVERIFIED: the original uses one `rotary_emb` for both towers. Here both towers use the rotary
// object owned by the understanding layer's attention (rope_theta and mrope_section come from the
// understanding text config). If a checkpoint ever had different values for the two towers this
// would need a separate rotary for the generation side.

import Foundation
import MLX
import MLXNN

enum SharpAICosmos3OmniJointAttention {

    /// The `Qwen3VLLanguage.DecoderLayer`s of `model`, in layer order.
    static func understandingLayers(of model: Qwen3VL) -> [Qwen3VLLanguage.DecoderLayer] {
        var found = [(index: Int?, order: Int, layer: Qwen3VLLanguage.DecoderLayer)]()
        for (order, entry) in model.namedModules().enumerated() {
            guard let layer = entry.1 as? Qwen3VLLanguage.DecoderLayer else { continue }
            let last = entry.0.split(separator: ".").last.flatMap { Int($0) }
            found.append((last, order, layer))
        }
        return found.sorted { lhs, rhs in
            if let a = lhs.index, let b = rhs.index { return a < b }
            return lhs.order < rhs.order
        }.map { $0.layer }
    }

    /// rotate_half as in the original: cat(-x[..., half:], x[..., :half]).
    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    /// q or k [B,H,N,D] with cos/sin [B,N,D].
    private static func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let c = expandedDimensions(cos, axis: 1)
        let s = expandedDimensions(sin, axis: 1)
        return (x * c) + (rotateHalf(x) * s)
    }

    /// Two-way attention for one layer. Inputs are the already input-layernormed hidden states.
    /// Returns the output-projected attention results (understanding, generation).
    static func attend(
        und: Qwen3VLLanguage.Attention,
        gen: SharpAICosmos3OmniGenAttention,
        undNormed: MLXArray,
        genNormed: MLXArray,
        undPositions: MLXArray,
        genPositions: MLXArray
    ) -> (und: MLXArray, gen: MLXArray) {
        let (batch, undLength) = (undNormed.dim(0), undNormed.dim(1))
        let genLength = genNormed.dim(1)

        // Understanding projections (the upstream layer's own weights and QK norm).
        var qUnd = und.wq(undNormed).reshaped(batch, undLength, und.heads, und.headDim)
        var kUnd = und.wk(undNormed).reshaped(batch, undLength, und.kvHeads, und.headDim)
        let vUnd = und.wv(undNormed).reshaped(batch, undLength, und.kvHeads, und.headDim)
            .transposed(0, 2, 1, 3)
        qUnd = und.qNorm(qUnd).transposed(0, 2, 1, 3)
        kUnd = und.kNorm(kUnd).transposed(0, 2, 1, 3)

        let projected = gen.project(genNormed)
        var qGen = projected.q
        var kGen = projected.k
        let vGen = projected.v

        // RoPE per tower, each with its own 3 x B x N position ids.
        let (cosUnd, sinUnd) = und.rotaryEmbedding(positionIds: undPositions, dtype: undNormed.dtype)
        let (cosGen, sinGen) = und.rotaryEmbedding(positionIds: genPositions, dtype: genNormed.dtype)
        qUnd = rotate(qUnd, cos: cosUnd, sin: sinUnd)
        kUnd = rotate(kUnd, cos: cosUnd, sin: sinUnd)
        qGen = rotate(qGen, cos: cosGen, sin: sinGen)
        kGen = rotate(kGen, cos: cosGen, sin: sinGen)

        // Understanding: causal over understanding tokens only. SDPA handles GQA natively.
        let undOut = MLXFast.scaledDotProductAttention(
            queries: qUnd, keys: kUnd, values: vUnd, scale: und.scale, mask: .causal
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, undLength, -1)

        // Generation: full attention over [und K/V ; gen K/V].
        let allKeys = concatenated([kUnd, kGen], axis: 2)
        let allValues = concatenated([vUnd, vGen], axis: 2)
        let genOut = MLXFast.scaledDotProductAttention(
            queries: qGen, keys: allKeys, values: allValues, scale: gen.scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, genLength, -1)

        return (und.wo(undOut), gen.wo(genOut))
    }

    /// One MoT decoder layer: pre-norm per tower, joint attention, residuals, per-tower SwiGLU MLP.
    static func layerForward(
        undLayer: Qwen3VLLanguage.DecoderLayer,
        genLayer: SharpAICosmos3OmniGenLayer,
        und: MLXArray,
        gen: MLXArray,
        undPositions: MLXArray,
        genPositions: MLXArray
    ) -> (und: MLXArray, gen: MLXArray) {
        let (undAttn, genAttn) = attend(
            und: undLayer.attention, gen: genLayer.attention,
            undNormed: undLayer.inputLayerNorm(und),
            genNormed: genLayer.inputLayerNorm(gen),
            undPositions: undPositions, genPositions: genPositions)
        let undResidual = und + undAttn
        let genResidual = gen + genAttn
        let undOut = undResidual + undLayer.mlp(undLayer.postAttentionLayerNorm(undResidual))
        let genOut = genResidual + genLayer.mlp(genLayer.postAttentionLayerNorm(genResidual))
        return (undOut, genOut)
    }
}
