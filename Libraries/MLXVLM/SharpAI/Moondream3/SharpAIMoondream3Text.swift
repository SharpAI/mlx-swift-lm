// SharpAI fork overlay: moondream3 text decoder (tau attention, dense and MoE feed-forward).
// Not part of upstream mlx-swift-lm.
//
// Port of vLLM's Moondream3Attention, Moondream3TextMoE and Moondream3DecoderLayer.
// Each block is parallel: x + attn(ln(x)) + mlp(ln(x)).
// UNVERIFIED: the MoE runs every expert and weights the outputs with a dense routing matrix.
// That gives the same result as top-k dispatch but costs about E / k times the compute.
// UNVERIFIED: the HF bidirectional prefix mask (prefix_attn) is not applied. vLLM does not
// apply it either, so this port uses causal attention everywhere.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Tau scaling weights for queries and values. Checkpoint keys: `attn.tau.{alpha,wq,wv}`.
final class SharpAIMoondream3Tau: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "wq") var wq: MLXArray
    @ParameterInfo(key: "wv") var wv: MLXArray

    init(heads: Int, qkvDim: Int) {
        _alpha.wrappedValue = MLXArray.zeros([heads])
        _wq.wrappedValue = MLXArray.zeros([heads, qkvDim])
        _wv.wrappedValue = MLXArray.zeros([heads, qkvDim])
        super.init()
    }
}

final class SharpAIMoondream3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    @ModuleInfo var tau: SharpAIMoondream3Tau

    init(_ config: SharpAIMoondream3TextConfiguration) {
        numHeads = config.heads
        numKVHeads = config.kvHeads
        headDim = config.dim / config.heads
        scale = pow(Float(config.dim / config.heads), -0.5)
        // Half of each head is rotated, as vLLM does (32 of 64 dims).
        rope = RoPE(
            dimensions: (config.dim / config.heads) / 2, traditional: false,
            base: config.ropeTheta, scale: 1)

        let qkvWidth = (config.heads + 2 * config.kvHeads) * (config.dim / config.heads)
        _qkv.wrappedValue = Linear(config.dim, qkvWidth, bias: true)
        _proj.wrappedValue = Linear(config.heads * (config.dim / config.heads), config.dim, bias: true)
        _tau.wrappedValue = SharpAIMoondream3Tau(heads: config.heads, qkvDim: qkvWidth)
        super.init()
    }

    /// Token-and-position gate in `[B, L, H]`, as vLLM's `tau_q` and `tau_v`.
    private func tauGates(
        _ qkvOut: MLXArray, offset: Int
    ) -> (q: MLXArray, v: MLXArray) {
        let length = qkvOut.dim(1)
        let features = MLXNN.gelu(qkvOut)
        let tokQ = MLX.tanh(MLX.matmul(features, tau.wq.transposed(1, 0)))
        let tokV = MLX.tanh(MLX.matmul(features, tau.wv.transposed(1, 0)))

        let positions = MLXArray((0 ..< length).map { Float($0 + offset) })
        let posLog = MLX.log(positions + 1)
        let slope = tau.alpha.expandedDimensions(axis: 1) * posLog.expandedDimensions(axis: 0)
        // `1 + (sigmoid - 0.5)` from vLLM, as `[H, L]`, then moved to `[L, H]`.
        let posGate = (MLX.sigmoid(slope) - 0.5) + 1
        let posT = posGate.transposed(1, 0)
        return (tokQ + posT, tokV + posT)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let qkvOut = qkv(x)
        let qWidth = numHeads * headDim
        let kvWidth = numKVHeads * headDim

        var q = qkvOut[0..., 0..., ..<qWidth].reshaped(batch, length, numHeads, headDim)
        var k = qkvOut[0..., 0..., qWidth ..< (qWidth + kvWidth)]
            .reshaped(batch, length, numKVHeads, headDim)
        var v = qkvOut[0..., 0..., (qWidth + kvWidth)...]
            .reshaped(batch, length, numKVHeads, headDim)

        let tokenOffset = cache?.offset ?? 0
        let gates = tauGates(qkvOut, offset: tokenOffset)
        q = q * gates.q.expandedDimensions(axis: -1)
        v = v * gates.v[0..., 0..., ..<numKVHeads].expandedDimensions(axis: -1)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let ropeOffset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: ropeOffset)
        k = applyRotaryPosition(rope, to: k, offset: ropeOffset)

        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return proj(out.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

/// Stacked expert tensor. Checkpoint keys: `mlp.fc1.weight` and `mlp.fc2.weight` (3-D).
final class SharpAIMoondream3ExpertWeight: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ shape: [Int]) {
        _weight.wrappedValue = MLXArray.zeros(shape)
        super.init()
    }
}

/// Mixture of experts with the GeGLU variant `gelu(h) * (g + 1)` over the fused `fc1` output.
final class SharpAIMoondream3MoE: Module {
    let numExperts: Int
    let topK: Int
    let innerDim: Int

    @ModuleInfo var router: Linear
    @ModuleInfo var fc1: SharpAIMoondream3ExpertWeight
    @ModuleInfo var fc2: SharpAIMoondream3ExpertWeight

    init(dim: Int, innerDim: Int, experts: Int, topK: Int) {
        numExperts = experts
        self.topK = topK
        self.innerDim = innerDim
        _router.wrappedValue = Linear(dim, experts, bias: true)
        _fc1.wrappedValue = SharpAIMoondream3ExpertWeight([experts, innerDim * 2, dim])
        _fc2.wrappedValue = SharpAIMoondream3ExpertWeight([experts, dim, innerDim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2))
        let tokens = batch * length
        let flat = x.reshaped(tokens, dim)

        let logits = router(flat)
        let chosen = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let selected = takeAlong(logits, chosen, axis: -1)
        let weights = MLX.softmax(selected.asType(.float32), axis: -1).asType(x.dtype)

        // Dense `[N, E]` routing matrix: zero for experts that are not chosen.
        let expertIDs = MLXArray((0 ..< numExperts).map { Int32($0) })
        let hit = MLX.equal(chosen.expandedDimensions(axis: -1), expertIDs).asType(x.dtype)
        let routing = (hit * weights.expandedDimensions(axis: -1)).sum(axis: 1)

        // All experts at once: `[N, 1, 1, D] @ [E, D, 2I]` gives `[N, E, 1, 2I]`.
        let up = MLX.matmul(
            flat.reshaped(tokens, 1, 1, dim), fc1.weight.transposed(0, 2, 1))
            .reshaped(tokens, numExperts, innerDim * 2)
        let halves = up.split(parts: 2, axis: -1)
        let activated = MLXNN.gelu(halves[0]) * (halves[1] + 1)

        let down = MLX.matmul(
            activated.reshaped(tokens, numExperts, 1, innerDim), fc2.weight.transposed(0, 2, 1))
            .reshaped(tokens, numExperts, dim)
        let combined = (down * routing.expandedDimensions(axis: -1)).sum(axis: 1)
        return combined.reshaped(batch, length, dim)
    }
}

/// Dense feed-forward for the first `moe_start_layer` blocks.
final class SharpAIMoondream3DenseMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(dim: Int, ffDim: Int) {
        _fc1.wrappedValue = Linear(dim, ffDim, bias: true)
        _fc2.wrappedValue = Linear(ffDim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))
    }
}

final class SharpAIMoondream3Block: Module {
    @ModuleInfo var ln: LayerNorm
    @ModuleInfo var attn: SharpAIMoondream3Attention
    /// `SharpAIMoondream3DenseMLP` or `SharpAIMoondream3MoE`, chosen by layer index.
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ config: SharpAIMoondream3TextConfiguration, layerIndex: Int) {
        _ln.wrappedValue = LayerNorm(dimensions: config.dim, eps: 1e-5)
        _attn.wrappedValue = SharpAIMoondream3Attention(config)
        if layerIndex >= config.moeStartLayer {
            _mlp.wrappedValue = SharpAIMoondream3MoE(
                dim: config.dim, innerDim: config.moeExpertInnerDim,
                experts: config.moeExperts, topK: config.moeExpertsPerToken)
        } else {
            _mlp.wrappedValue = SharpAIMoondream3DenseMLP(dim: config.dim, ffDim: config.ffDim)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let normed = ln(x)
        let attnOut = attn(normed, mask: mask, cache: cache)
        let mlpOut: MLXArray
        if let moe = mlp as? SharpAIMoondream3MoE {
            mlpOut = moe(normed)
        } else {
            mlpOut = (mlp as! SharpAIMoondream3DenseMLP)(normed)
        }
        return x + attnOut + mlpOut
    }
}

/// Text decoder and LM head. Keys sit under `text.*`.
final class SharpAIMoondream3TextModel: Module {
    @ModuleInfo(key: "wte") var wte: Embedding
    let blocks: [SharpAIMoondream3Block]
    @ModuleInfo(key: "post_ln") var postLn: LayerNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAIMoondream3TextConfiguration) {
        _wte.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dim)
        self.blocks = (0 ..< config.layers).map { index in
            SharpAIMoondream3Block(config, layerIndex: index)
        }
        _postLn.wrappedValue = LayerNorm(dimensions: config.dim, eps: 1e-5)
        _lmHead.wrappedValue = Linear(config.dim, config.vocabularySize, bias: true)
        super.init()
    }

    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, block) in blocks.enumerated() {
            h = block(h, mask: mask, cache: cache?[index])
        }
        return lmHead(postLn(h))
    }
}
