// SharpAI fork overlay: minimax_m3_dspark draft backbone (forward pass). Not part of upstream mlx-swift-lm.
//
// Ported from NVIDIA NeMo Automodel (Apache-2.0):
//   nemo_automodel/components/speculative/dspark/draft_minimax_m3.py
//     MiniMaxM3DSparkModel._forward_backbone, MiniMaxM3DSparkAttention, MiniMaxM3DSparkDecoderLayer
//   nemo_automodel/components/attention/dflash_mask.py
//     create_dflash_sdpa_mask with causal=False and no sliding window
//   nemo_automodel/components/speculative/dspark/common.py
//     create_noise_embed, create_position_ids
//   nemo_automodel/components/models/gpt_oss/rope_utils.py
//     apply_rotary_emb (partial rotary, split-half), RoPE angles with concentration 1
// Gemma RMSNorm and SwiGLU-OAI MLP come from the repo's MiniMax M3 Text overlay (same module).
//
// NOT ported yet (see the report): anchor sampling (sample_anchor_positions), the loss and eval
// masks, the markov head (markov_head.py was read only as a summary), the confidence head, and the
// frozen target model. The caller passes anchors and keep flags, and the forward returns base
// draft logits without the markov correction.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Backbone shape. Field names follow the NeMo config reads (hidden_size, rope fields, and so on).
struct SharpAIMinimaxM3DsparkBackboneConfig {
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var vocabSize: Int
    /// Number of target layers whose hidden states are concatenated into the context input.
    var targetLayerCount: Int
    var blockSize: Int
    var maskTokenId: Int
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 5_000_000
    var partialRotaryFactor: Float = 1.0
    var swigluAlpha: Float = 1.702
    var swigluLimit: Float = 7.0

    /// int(head_dim * partial_rotary_factor), as in NeMo's RotaryEmbedding.
    var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) }
}

/// DFlash block mask, SDPA form. Returns a [B, 1, Q, KV] additive mask with 0 where attention is
/// allowed and -inf elsewhere, where Q = N * blockSize and KV = ctxLen + Q.
///
/// Query block b sees context positions below its anchor (only when the block is kept), and its own
/// block's noise slots. Other blocks are invisible. Mirrors create_dflash_sdpa_mask (causal=False,
/// no sliding window, no packing).
func sharpAIMinimaxM3DsparkDFlashMask(
    anchors: [[Int]], keep: [[Bool]], ctxLen: Int, blockSize: Int
) -> MLXArray {
    let batch = anchors.count
    let numBlocks = anchors.first?.count ?? 0
    let qLen = numBlocks * blockSize
    let kvLen = ctxLen + qLen
    var values = [Float](repeating: -Float.infinity, count: batch * qLen * kvLen)
    for b in 0 ..< batch {
        for q in 0 ..< qLen {
            let qBlock = q / blockSize
            let anchor = anchors[b][qBlock]
            let kept = keep[b][qBlock]
            let row = (b * qLen + q) * kvLen
            for kv in 0 ..< kvLen {
                let visible: Bool
                if kv < ctxLen {
                    visible = kept && kv < anchor
                } else {
                    visible = (kv - ctxLen) / blockSize == qBlock
                }
                if visible {
                    values[row + kv] = 0
                }
            }
        }
    }
    return MLXArray(values, [batch, 1, qLen, kvLen])
}

/// cos/sin tables [B, L, rotaryDim / 2] for absolute positions [B][L]. Matches RotaryEmbedding with
/// scaling factor 1: inv_freq[i] = 1 / base^(2i / rotaryDim), angle = position * inv_freq.
func sharpAIMinimaxM3DsparkRopeTables(
    positions: [[Int]], rotaryDim: Int, base: Float
) -> (cos: MLXArray, sin: MLXArray) {
    let half = rotaryDim / 2
    let invFreq: [Float] = (0 ..< half).map { i -> Float in
        1 / pow(base, Float(2 * i) / Float(rotaryDim))
    }
    let batch = positions.count
    let length = positions.first?.count ?? 0
    let flat: [Float] = positions.flatMap { row in row.map { Float($0) } }
    let angles = MLXArray(flat, [batch, length, 1]) * MLXArray(invFreq, [1, 1, half])
    return (MLX.cos(angles), MLX.sin(angles))
}

/// apply_rotary_emb: rotates the first 2 * cos.dim(-1) channels of x [B, L, H, D] in the split-half
/// convention. Channels past the rotary width pass through unchanged.
func sharpAIMinimaxM3DsparkRotate(_ x: MLXArray, cos cosTable: MLXArray, sin sinTable: MLXArray)
    -> MLXArray
{
    let c = cosTable.expandedDimensions(axis: -2).asType(x.dtype)
    let s = sinTable.expandedDimensions(axis: -2).asType(x.dtype)
    let rotaryDim = cosTable.dim(-1) * 2
    let headDim = x.dim(-1)
    if rotaryDim < headDim {
        let xRot = x[0..., 0..., 0..., ..<rotaryDim]
        let xPass = x[0..., 0..., 0..., rotaryDim...]
        let halves = split(xRot, parts: 2, axis: -1)
        let o1 = halves[0] * c - halves[1] * s
        let o2 = halves[1] * c + halves[0] * s
        return concatenated([o1, o2, xPass], axis: -1)
    }
    let halves = split(x, parts: 2, axis: -1)
    let o1 = halves[0] * c - halves[1] * s
    let o2 = halves[1] * c + halves[0] * s
    return concatenated([o1, o2], axis: -1)
}

/// Dense, non-causal GQA attention over the [context | noise block] layout.
/// Per-head Gemma RMSNorm on Q and K before RoPE, as in MiniMaxM3DSparkAttention.
class SharpAIMinimaxM3DsparkAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIMinimaxM3TextRMSNorm

    init(_ config: SharpAIMinimaxM3DsparkBackboneConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// - Parameters:
    ///   - hidden: noise-block states [B, Q, hidden]
    ///   - target: projected context states [B, S, hidden]
    ///   - cos, sin: full-position tables [B, S + Q, rotaryDim / 2]
    ///   - mask: additive DFlash mask [B, 1, Q, S + Q]
    func callAsFunction(
        _ hidden: MLXArray, target: MLXArray, cos cosTable: MLXArray, sin sinTable: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let qLen = hidden.dim(1)
        let ctxLen = target.dim(1)
        let total = ctxLen + qLen
        let groups = numHeads / numKVHeads

        // Queries come from the noise block only. Their RoPE uses the draft suffix of the tables.
        var queries = qNorm(qProj(hidden).reshaped(batch, qLen, numHeads, headDim))
        queries = sharpAIMinimaxM3DsparkRotate(
            queries,
            cos: cosTable[0..., ctxLen..., 0...],
            sin: sinTable[0..., ctxLen..., 0...])

        // Keys and values span [context | noise]. Projecting the two parts separately and then
        // concatenating is equivalent, since the projections are per token.
        var keys = concatenated([kProj(target), kProj(hidden)], axis: 1)
            .reshaped(batch, total, numKVHeads, headDim)
        let values = concatenated([vProj(target), vProj(hidden)], axis: 1)
            .reshaped(batch, total, numKVHeads, headDim)
        keys = kNorm(keys)
        keys = sharpAIMinimaxM3DsparkRotate(keys, cos: cosTable, sin: sinTable)

        let q = queries.transposed(0, 2, 1, 3)
        let k = repeated(keys.transposed(0, 2, 1, 3), count: groups, axis: 1)
        let v = repeated(values.transposed(0, 2, 1, 3), count: groups, axis: 1)

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: nil,
            scale: scale,
            mask: .array(mask)
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, qLen, -1)

        return oProj(output)
    }
}

/// Pre-norm residual block: draft attention, then a dense SwiGLU-OAI MLP.
class SharpAIMinimaxM3DsparkDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinimaxM3DsparkAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIMinimaxM3TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm:
        SharpAIMinimaxM3TextRMSNorm

    init(_ config: SharpAIMinimaxM3DsparkBackboneConfig) {
        _selfAttn.wrappedValue = SharpAIMinimaxM3DsparkAttention(config)
        _mlp.wrappedValue = SharpAIMinimaxM3TextMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize,
            alpha: config.swigluAlpha, limit: config.swigluLimit)
        _inputLayerNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray, target: MLXArray, cos cosTable: MLXArray, sin sinTable: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        var out =
            hidden
            + selfAttn(
                inputLayerNorm(hidden), target: target, cos: cosTable, sin: sinTable, mask: mask)
        out = out + mlp(postAttentionLayerNorm(out))
        return out
    }
}

/// The DSpark draft backbone: noise-block embedding, context fusion, dense layers, final norm and
/// lm_head. Returns base logits per block position. The markov head and confidence head are not
/// applied here.
class SharpAIMinimaxM3DsparkBackbone: Module {
    let blockSize: Int
    let maskTokenId: Int
    let rotaryDim: Int
    let ropeTheta: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: SharpAIMinimaxM3TextRMSNorm
    var layers: [SharpAIMinimaxM3DsparkDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAIMinimaxM3DsparkBackboneConfig) {
        self.blockSize = config.blockSize
        self.maskTokenId = config.maskTokenId
        self.rotaryDim = config.rotaryDim
        self.ropeTheta = config.ropeTheta

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _fc.wrappedValue = Linear(
            config.targetLayerCount * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            SharpAIMinimaxM3DsparkDecoderLayer(config)
        }
        _norm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - inputIds: context tokens [B][S]. Only the anchor tokens are read.
    ///   - targetHidden: concatenated target-layer hidden states [B, S, targetLayerCount * hidden]
    ///   - anchors: anchor positions [B][N]. Dropped blocks should carry 0, as in NeMo.
    ///   - keep: per-block keep flags [B][N]
    /// - Returns: base draft logits [B, N, blockSize, vocab]
    func callAsFunction(
        inputIds: [[Int]], targetHidden: MLXArray, anchors: [[Int]], keep: [[Bool]]
    ) -> MLXArray {
        let batch = targetHidden.dim(0)
        let ctxLen = targetHidden.dim(1)
        let numBlocks = anchors[0].count
        let qLen = numBlocks * blockSize

        // Noise block: each block starts with its anchor token when kept, else the mask token.
        // The remaining slots are mask tokens.
        var noiseIds = [Int32](repeating: Int32(maskTokenId), count: batch * qLen)
        for b in 0 ..< batch {
            for n in 0 ..< numBlocks {
                let slot = b * qLen + n * blockSize
                noiseIds[slot] =
                    keep[b][n] ? Int32(inputIds[b][anchors[b][n]]) : Int32(maskTokenId)
            }
        }
        let noise = embedTokens(MLXArray(noiseIds, [batch, qLen]))

        let context = hiddenNorm(fc(targetHidden))

        // Positions: context 0..<S, then each block at anchor + offset, as in create_position_ids.
        let positions: [[Int]] = (0 ..< batch).map { b in
            Array(0 ..< ctxLen)
                + anchors[b].flatMap { anchor in (0 ..< blockSize).map { anchor + $0 } }
        }
        let (cosTable, sinTable) = sharpAIMinimaxM3DsparkRopeTables(
            positions: positions, rotaryDim: rotaryDim, base: ropeTheta)
        let mask = sharpAIMinimaxM3DsparkDFlashMask(
            anchors: anchors, keep: keep, ctxLen: ctxLen, blockSize: blockSize)

        var hidden = noise
        for layer in layers {
            hidden = layer(
                hidden, target: context, cos: cosTable, sin: sinTable, mask: mask)
        }
        let logits = lmHead(norm(hidden))
        return logits.reshaped(batch, numBlocks, blockSize, -1)
    }
}
