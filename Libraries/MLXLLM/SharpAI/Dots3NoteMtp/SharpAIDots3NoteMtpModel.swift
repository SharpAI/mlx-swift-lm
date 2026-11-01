// SharpAI fork overlay: dots3_note_mtp draft head. Not part of upstream.
//
// Math follows sglang's Dots3 NextN draft (python/sglang/srt/models/dots3_common/nextn.py):
//   h0 = eh_proj(concat(enorm(embed(ids)), hnorm(target_hidden)))
//   h  = Decoder(h0)   (pre-norm: x + attn(ln(x)); h + mlp(post_ln(h)))
//   out = shared_head.norm(h)
// The decoder is the sliding-attention MLA block: q/kv latents rescaled by sqrt(hidden / rank)
// (transformers `q_lora_scale` / `kv_lora_scale`), interleaved RoPE on the rope slice, and a
// headwise sigmoid gate applied to the attention output before `o_proj` (transformers
// `Dots3NoteTextAttention.forward`).
//
// The shared lm_head is not in this module: the checkpoint has no `shared_head.head`, and the
// caller applies the target model's head to `out`. This forward runs on a full sequence and does
// not use a KV cache; `offset` shifts only the RoPE positions.

import Foundation
import MLX
import MLXNN

/// Sliding-window causal mask: query i attends to key j iff 0 <= i - j < window.
/// This matches HF `sliding_window_overlay` (`kv_idx > q_idx - sliding_window`).
func dots3NoteMtpSlidingMask(length: Int, window: Int) -> MLXArray {
    let positions = MLXArray((0 ..< length).map { Int32($0) })
    let queryIdx = expandedDimensions(positions, axis: 1)
    let keyIdx = expandedDimensions(positions, axis: 0)
    let distance = queryIdx - keyIdx
    return logicalAnd(
        greaterEqual(distance, MLXArray(Int32(0))),
        less(distance, MLXArray(Int32(window))))
}

/// Sliding-attention MLA block used by the MTP layer.
final class Dots3NoteMtpAttention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qkHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let ropeTheta: Float
    let scaling: Float
    /// `sqrt(hidden / q_lora_rank)` when the rescale flag is set, else 1 (transformers `q_lora_scale`).
    let qLoraScale: Float
    /// `sqrt(hidden / kv_lora_rank)` when the rescale flag is set, else 1 (transformers `kv_lora_scale`).
    let kvLoraScale: Float

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayernorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayernorm: RMSNorm
    @ModuleInfo(key: "k_rope_only_layernorm") var kRopeOnlyLayernorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIDots3NoteMtpConfiguration) {
        let hidden = config.hiddenSize
        let heads = config.swaNumAttentionHeads
        let nope = config.swaQkNopeHeadDim
        let rope = config.swaQkRopeHeadDim
        let qkHead = nope + rope
        let vHead = config.swaVHeadDim
        let qLora = config.swaQLoraRank
        let kvLora = config.swaKvLoraRank
        let eps = config.rmsNormEps

        self.numHeads = heads
        self.qkNopeHeadDim = nope
        self.qkRopeHeadDim = rope
        self.qkHeadDim = qkHead
        self.vHeadDim = vHead
        self.kvLoraRank = kvLora
        self.ropeTheta = config.swaRopeTheta
        self.scaling = pow(Float(qkHead), -0.5)
        if config.applyMlaQkvLoraRescale {
            self.qLoraScale = Float(sqrt(Double(hidden) / Double(qLora)))
            self.kvLoraScale = Float(sqrt(Double(hidden) / Double(kvLora)))
        } else {
            self.qLoraScale = 1
            self.kvLoraScale = 1
        }

        self._qAProj.wrappedValue = Linear(hidden, qLora, bias: false)
        self._qALayernorm.wrappedValue = RMSNorm(dimensions: qLora, eps: eps)
        self._qBProj.wrappedValue = Linear(qLora, heads * qkHead, bias: false)
        self._kvAProjWithMqa.wrappedValue = Linear(hidden, kvLora + rope, bias: false)
        self._kvALayernorm.wrappedValue = RMSNorm(dimensions: kvLora, eps: eps)
        self._kRopeOnlyLayernorm.wrappedValue = RMSNorm(dimensions: rope, eps: eps)
        self._kvBProj.wrappedValue = Linear(kvLora, heads * (nope + vHead), bias: false)
        self._gProj.wrappedValue = Linear(hidden, heads, bias: false)
        self._oProj.wrappedValue = Linear(heads * vHead, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, offset: Int) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Query path: q_a -> norm -> rescale -> q_b, split into nope and rope slices.
        let qLatent = qALayernorm(qAProj(x)) * qLoraScale
        let q = qBProj(qLatent).reshaped(B, L, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let qParts = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qParts[0]
        let qRot = MLXFast.RoPE(
            qParts[1], dimensions: qkRopeHeadDim, traditional: true, base: ropeTheta,
            scale: 1, offset: offset)

        // Key/value path: the shared rope key is normalized before RoPE (k_rope_only_layernorm).
        let compressed = kvAProjWithMqa(x)
        let compressedParts = split(compressed, indices: [kvLoraRank], axis: -1)
        let kvLatent = kvALayernorm(compressedParts[0]) * kvLoraScale
        let kRot = kRopeOnlyLayernorm(compressedParts[1]).reshaped(B, 1, L, qkRopeHeadDim)
        let kRotRoped = MLXFast.RoPE(
            kRot, dimensions: qkRopeHeadDim, traditional: true, base: ropeTheta,
            scale: 1, offset: offset)

        let kv = kvBProj(kvLatent)
            .reshaped(B, L, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let kvParts = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvParts[0]
        let values = kvParts[1]

        let queries = concatenated([qNope, qRot], axis: -1)
        let keys = concatenated([kNope, repeated(kRotRoped, count: numHeads, axis: 1)], axis: -1)

        let attn = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scaling, mask: mask)

        // Headwise sigmoid gate: g_proj gives one gate per head, broadcast over v_head_dim.
        let gate = sigmoid(gProj(x)).transposed(0, 2, 1).expandedDimensions(axis: -1)
        let gated = (attn * gate).transposed(0, 2, 1, 3).reshaped(B, L, numHeads * vHeadDim)
        return oProj(gated)
    }
}

/// Dense SiLU MLP. The MTP layer is dense (sglang `_is_layer_sparse` returns False when is_nextn).
final class Dots3NoteMtpMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class Dots3NoteMtpDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Dots3NoteMtpAttention
    @ModuleInfo(key: "mlp") var mlp: Dots3NoteMtpMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: SharpAIDots3NoteMtpConfiguration) {
        self._selfAttn.wrappedValue = Dots3NoteMtpAttention(config)
        self._mlp.wrappedValue = Dots3NoteMtpMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, offset: Int) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, offset: offset)
        return h + mlp(postAttentionLayernorm(h))
    }
}

final class Dots3NoteMtpSharedHead: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIDots3NoteMtpConfiguration) {
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
}

/// The dots3_note MTP draft head. Its parameter keys match the checkpoint after
/// `SharpAIDots3NoteMtpSanitize` (`embed_tokens.*`, `enorm.*`, `hnorm.*`, `eh_proj.*`,
/// `decoder.*`, `shared_head.norm.*`).
public final class SharpAIDots3NoteMtpHead: Module {
    public let vocabSize: Int
    public let slidingWindowSize: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "decoder") var decoder: Dots3NoteMtpDecoderLayer
    @ModuleInfo(key: "shared_head") var sharedHead: Dots3NoteMtpSharedHead

    public init(_ config: SharpAIDots3NoteMtpConfiguration) {
        self.vocabSize = config.vocabSize
        self.slidingWindowSize = config.slidingWindowSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._enorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._hnorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ehProj.wrappedValue = Linear(2 * config.hiddenSize, config.hiddenSize, bias: false)
        self._decoder.wrappedValue = Dots3NoteMtpDecoderLayer(config)
        self._sharedHead.wrappedValue = Dots3NoteMtpSharedHead(config)
    }

    /// - Parameters:
    ///   - inputIds: `[batch, length]` token ids of the drafted positions.
    ///   - hiddenStates: `[batch, length, hidden]` target-model hidden states for the same positions.
    ///   - offset: RoPE position offset of the first position.
    /// - Returns: `[batch, length, hidden]` states after `shared_head.norm`, before the shared lm_head.
    public func callAsFunction(
        inputIds: MLXArray, hiddenStates: MLXArray, offset: Int = 0
    ) -> MLXArray {
        let length = inputIds.dim(1)
        let tokenEmbedding = enorm(embedTokens(inputIds))
        let targetHidden = hnorm(hiddenStates)
        var x = ehProj(concatenated([tokenEmbedding, targetHidden], axis: -1))
        let mask = dots3NoteMtpSlidingMask(length: length, window: slidingWindowSize)
        x = decoder(x, mask: mask, offset: offset)
        return sharedHead.norm(x)
    }
}
