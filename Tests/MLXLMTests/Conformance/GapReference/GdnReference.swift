// Pure reference implementations of the gated-delta-rule family (GDN / KDA), built ONLY from MLX ops.
//
// Round 3, lane r3-gdn (#254 SL-29 / SL-31). Resolves the two ops the kernel audit left "unknown":
//   * Qwen4Exp "GDN deltas" and its decode-state carry / speculative rollback,
//   * MuseSpark "GDN2".
// Test support only; nothing here is used by shipping code.
//
// upstream-watch: SharpAI-only namespace. If ml-explore/mlx-swift-lm lands a per-key-gate (KDA)
// GatedDelta path, a `chunkedGatedDelta`, or a Qwen4Exp / MuseSpark model, compare it against these
// functions before retiring them.
//
// Provenance (vLLM snapshot under scratchpad/survey/vllm_src, read-only):
//   recurrence, gate, beta, L2-norm, q scale, per-token state slots, num_accepted_tokens
//     -> vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py (kernel body; IS_KDA branch)
//   KDA gate with optional lower bound
//     -> vllm/third_party/flash_linear_attention/ops/kda.py (kda_gate_fwd_kernel, :1477-1527)
//   Qwen4Exp layer = QwenGatedDeltaNetAttention(gqa_interleaved_layout=False), output gate
//     silu|sigmoid -> vllm/models/qwen4_exp/nvidia/model.py:227, vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:668
//   MuseSpark GDN2: INFERRED. vllm/models/muse_spark/nvidia/model.py:1836 raises NotImplementedError;
//     only the config (vllm/transformers_utils/configs/muse_spark.py:49-62) exists. See kernel-plan.md.
//
// Shapes (all float32 inside):
//   q, k      [B, T, Hk, Dk]       (already L2-normalised / scaled, see `prepQK`)
//   v         [B, T, Hv, Dv]       Hv a multiple of Hk (GQA: k heads are repeated)
//   logDecay  [B, T, Hv, Dk]       log of the per-step decay, <= 0. Scalar gate: last dim 1 (broadcast).
//   beta      [B, T, Hv]
//   state     [B, Hv, Dv, Dk]      same layout as Libraries/MLXLMCommon/GatedDelta.swift
//
// Per step (decay first, as in GatedDelta.swift): S <- S * exp(logDecay)  ;  kv = S k  ;
//   delta = beta * (v - kv)  ;  S <- S + delta k^T  ;  y = S q.

import Foundation
import MLX

public enum SharpAIOverlayGdnReference {

    // MARK: gates and input prep

    /// log(1 + e^x), accurate for every x whatever the backend's log1p / logAddExp quality.
    ///
    /// CI evidence (PR #94, e6d4239): on the hosted runner `logAddExp(x, 0)` returned values quantised to ~1.2e-7
    /// (float32 spacing at 1.0) for x << 0, i.e. up to 40% relative error where softplus(x) ~ e^x ~ 1e-7. The
    /// reference must not inherit that, so it is computed in float64 on the CPU stream from `exp` alone:
    /// with t = -|x| and e = exp(t) <= 1, softplus(x) = max(x, 0) + log(1 + e); for e < 0.1 the alternating series
    /// e - e^2/2 + e^3/3 - ... (14 terms, truncation < 1e-15) replaces log(1 + e), which would lose e's low bits.
    static func softplus(_ x: MLXArray) -> MLXArray {
        let outType = x.dtype
        return Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let xd = x.asType(.float64)
            let e = exp(-abs(xd))
            let terms = 14
            var acc = MLXArray(Double(1) / Double(terms)).asType(.float64)
            var k = terms - 1
            while k >= 1 {
                acc = MLXArray(Double(1) / Double(k)).asType(.float64) - e * acc
                k -= 1
            }
            let series = e * acc
            let direct = log(MLXArray(Double(1)).asType(.float64) + e)
            let tail = which(e .< MLXArray(Double(0.1)).asType(.float64), series, direct)
            return (maximum(xd, MLXArray(Double(0)).asType(.float64)) + tail).asType(outType)
        }
    }

    /// Scalar (per value head) GDN gate as a log-decay: `-exp(aLog) * softplus(a + dtBias)`.
    /// a: [B, T, Hv]; aLog, dtBias: [Hv]. Returns [B, T, Hv, 1]. `exp()` of it is `computeGatedDeltaG`.
    public static func scalarLogDecay(aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray {
        let x = a.asType(.float32) + dtBias.asType(.float32)
        return expandedDimensions(-exp(aLog.asType(.float32)) * softplus(x), axis: -1)
    }

    /// KDA per-key-dimension gate as a log-decay (kda.py:1477): with `lowerBound == nil`
    /// `-exp(aLog_h) * softplus(g + bias)`, otherwise the "safe gate" `lowerBound * sigmoid(exp(aLog_h) * (g + bias))`
    /// (lowerBound < 0, so the log-decay is bounded below by it).
    /// g: [B, T, Hv, Dk]; aLog: [Hv]; bias: [Hv, Dk]. Returns [B, T, Hv, Dk].
    public static func kdaLogDecay(aLog: MLXArray, g: MLXArray, bias: MLXArray, lowerBound: Float? = nil) -> MLXArray {
        let x = g.asType(.float32) + bias.asType(.float32)
        let a = exp(aLog.asType(.float32)).reshaped(1, 1, -1, 1)
        if let lowerBound { return lowerBound * sigmoid(a * x) }
        return -a * softplus(x)
    }

    /// beta = sigmoid(b); `allowNegEigval` doubles it (OlmoHybrid, range (0, 2) -> negative eigenvalues).
    public static func beta(_ b: MLXArray, allowNegEigval: Bool = false) -> MLXArray {
        let s = sigmoid(b.asType(.float32))
        return allowNegEigval ? s * 2 : s
    }

    /// vLLM `use_qk_l2norm_in_kernel`: x * rsqrt(sum(x^2) + 1e-6) per head, then q * Dk^-0.5.
    public static func prepQK(q: MLXArray, k: MLXArray) -> (q: MLXArray, k: MLXArray) {
        func l2(_ x: MLXArray) -> MLXArray {
            let x = x.asType(.float32)
            return x * rsqrt(x.square().sum(axis: -1, keepDims: true) + 1e-6)
        }
        let scale = 1 / Float(q.dim(-1)).squareRoot()
        return (l2(q) * scale, l2(k))
    }

    /// Output gate of the Qwen GDN block: RMSNorm over the value head dim, then * act(z).
    /// `kind` is the `output_gate_type` config key: "silu" | "swish" (= silu) | "sigmoid".
    public static func gatedRMSNorm(_ x: MLXArray, z: MLXArray, weight: MLXArray, eps: Float, kind: String) -> MLXArray {
        let x32 = x.asType(.float32), z32 = z.asType(.float32)
        let normed = x32 * rsqrt(x32.square().mean(axis: -1, keepDims: true) + eps) * weight.asType(.float32)
        switch kind {
        case "silu", "swish": return normed * (z32 * sigmoid(z32))
        case "sigmoid": return normed * sigmoid(z32)
        default: fatalError("unsupported output_gate_type \(kind)")
        }
    }

    /// Qwen4Exp (gqa_interleaved_layout=False) / Qwen3.5 flat projection layout: the fused qkvz output is
    /// [q | k | v | z] along the last dim (key_dim, key_dim, value_dim, value_dim) and the fused ba output is
    /// [b | a]. (Qwen3-Next instead interleaves per key head group.) x: [..., 2*key + 2*value]; ba: [..., 2*Hv].
    public static func splitFlatQKVZBA(
        qkvz: MLXArray, ba: MLXArray, keyDim: Int, valueDim: Int, numVHeads: Int
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray) {
        let p = MLX.split(qkvz, indices: [keyDim, 2 * keyDim, 2 * keyDim + valueDim], axis: -1)
        let ab = MLX.split(ba, indices: [numVHeads], axis: -1)
        return (p[0], p[1], p[2], p[3], ab[0], ab[1])
    }

    // MARK: recurrent form (the oracle)

    static func repeatHeads(_ x: MLXArray, to hv: Int) -> MLXArray {
        let hk = x.dim(2)
        return hv == hk ? x : repeated(x, count: hv / hk, axis: 2)
    }

    /// Step-by-step delta rule. Returns y [B,T,Hv,Dv], final state, and (if asked) the state after EVERY
    /// token, [B, T, Hv, Dv, Dk] -- the "per-token state slots" vLLM keeps for speculative decoding.
    public static func recurrent(
        q: MLXArray, k: MLXArray, v: MLXArray, logDecay: MLXArray, beta: MLXArray,
        state: MLXArray? = nil, keepStates: Bool = false
    ) -> (y: MLXArray, state: MLXArray, states: MLXArray?) {
        let (b, t, hv, dv) = (v.dim(0), v.dim(1), v.dim(2), v.dim(3))
        let dk = k.dim(-1)
        let q = repeatHeads(q.asType(.float32), to: hv), k = repeatHeads(k.asType(.float32), to: hv)
        let v = v.asType(.float32)
        var s = state?.asType(.float32) ?? MLXArray.zeros([b, hv, dv, dk])
        var ys = [MLXArray](), all = [MLXArray]()
        for i in 0 ..< t {
            let ki = expandedDimensions(k[0..., i], axis: -2)  // [B,Hv,1,Dk]
            let g = exp(expandedDimensions(logDecay[0..., i].asType(.float32), axis: -2))  // [B,Hv,1,Dk|1]
            s = s * g
            let kv = (s * ki).sum(axis: -1)
            let delta = (v[0..., i] - kv) * expandedDimensions(beta[0..., i].asType(.float32), axis: -1)
            s = s + ki * expandedDimensions(delta, axis: -1)
            ys.append((s * expandedDimensions(q[0..., i], axis: -2)).sum(axis: -1))
            if keepStates { all.append(s) }
        }
        return (stacked(ys, axis: 1), s, keepStates ? stacked(all, axis: 1) : nil)
    }

    // MARK: chunked form (WY / UT representation, the prefill algorithm)

    /// Chunked delta rule. Within a chunk of C tokens, with G = cumsum(logDecay):
    ///   u_t = beta_t * (v_t - (S0 * e^{G_t}) k_t - sum_{s<t} A_ts u_s),  A_ts = sum_d k_t[d] k_s[d] e^{G_t[d]-G_s[d]}
    ///   y_t = (S0 * e^{G_t}) q_t + sum_{s<=t} B_ts u_s,                   B_ts = sum_d q_t[d] k_s[d] e^{G_t[d]-G_s[d]}
    ///   S_C = S0 * e^{G_C} + sum_s u_s (k_s * e^{G_C - G_s})^T
    /// Pairwise differences are formed before exponentiation (t >= s only), so nothing overflows for any
    /// decay. The C x C system is unit lower triangular and is solved by forward substitution.
    public static func chunked(
        q: MLXArray, k: MLXArray, v: MLXArray, logDecay: MLXArray, beta: MLXArray,
        state: MLXArray? = nil, chunk: Int
    ) -> (y: MLXArray, state: MLXArray) {
        let (b, t, hv, dv) = (v.dim(0), v.dim(1), v.dim(2), v.dim(3))
        let dk = k.dim(-1)
        let qh = repeatHeads(q.asType(.float32), to: hv), kh = repeatHeads(k.asType(.float32), to: hv)
        let v32 = v.asType(.float32)
        let ld = broadcast(logDecay.asType(.float32), to: [b, t, hv, dk])
        var s = state?.asType(.float32) ?? MLXArray.zeros([b, hv, dv, dk])
        var ys = [MLXArray]()

        for start in stride(from: 0, to: t, by: chunk) {
            let end = min(start + chunk, t), c = end - start
            // to [B, Hv, C, D]
            func head(_ x: MLXArray) -> MLXArray { x[0..., start ..< end].transposed(0, 2, 1, 3) }
            let qc = head(qh), kc = head(kh), vc = head(v32)
            let g = cumsum(head(ld), axis: 2)  // [B,Hv,C,Dk]
            let bc = beta[0..., start ..< end].asType(.float32).transposed(0, 2, 1)  // [B,Hv,C]

            let rows = expandedDimensions(MLX.arange(c), axis: 1)
            let cols = expandedDimensions(MLX.arange(c), axis: 0)
            let lower = (rows .>= cols)  // t >= s
            // exp(G_t - G_s) for t >= s, 0 elsewhere: [B,Hv,C(t),C(s),Dk]
            let diff = expandedDimensions(g, axis: 3) - expandedDimensions(g, axis: 2)
            let mask5 = lower.reshaped(1, 1, c, c, 1)
            let e = exp(MLX.where(mask5, diff, MLXArray(Float(0)))) * mask5.asType(.float32)
            let kk = (expandedDimensions(kc, axis: 3) * expandedDimensions(kc, axis: 2) * e).sum(axis: -1)  // A (use strict part)
            let qk = (expandedDimensions(qc, axis: 3) * expandedDimensions(kc, axis: 2) * e).sum(axis: -1)  // B (t >= s)

            let gExp = exp(g)  // [B,Hv,C,Dk]
            // (S0 * e^{G_t}) k_t  and  (S0 * e^{G_t}) q_t : [B,Hv,C,Dv]
            let s0 = expandedDimensions(s, axis: 2)  // [B,Hv,1,Dv,Dk]
            let s0k = (s0 * expandedDimensions(gExp * kc, axis: 3)).sum(axis: -1)
            let s0q = (s0 * expandedDimensions(gExp * qc, axis: 3)).sum(axis: -1)

            let rhs = expandedDimensions(bc, axis: -1) * (vc - s0k)
            var us = [MLXArray]()
            for i in 0 ..< c {
                var u = rhs[0..., 0..., i]
                if i > 0 {
                    let prev = stacked(us, axis: 2)  // [B,Hv,i,Dv]
                    let coef = expandedDimensions(bc[0..., 0..., i], axis: -1) * kk[0..., 0..., i, ..<i]  // [B,Hv,i]
                    u = u - (expandedDimensions(coef, axis: -1) * prev).sum(axis: 2)
                }
                us.append(u)
            }
            let u = stacked(us, axis: 2)  // [B,Hv,C,Dv]
            let y = s0q + (expandedDimensions(qk, axis: -1) * expandedDimensions(u, axis: 2)).sum(axis: 3)
            ys.append(y.transposed(0, 2, 1, 3))  // [B,C,Hv,Dv]

            let gLast = g[0..., 0..., (c - 1) ..< c]  // [B,Hv,1,Dk]
            let kDecayed = kc * exp(gLast - g)  // [B,Hv,C,Dk]
            s = s * exp(gLast).reshaped(b, hv, 1, dk) + (expandedDimensions(u, axis: -1) * expandedDimensions(kDecayed, axis: 3)).sum(axis: 2)
        }
        return (concatenated(ys, axis: 1), s)
    }

    // MARK: decode-state carry and speculative rollback

    /// Recurrent state after the first `accepted` of the verified tokens. `states` is the per-token stack
    /// from `recurrent(keepStates: true)`; `accepted == 0` returns the state the verification started from.
    /// vLLM: `fused_sigmoid_gating_delta_rule_update_kernel` stores the state after token i in slot i and
    /// the next step reads slot `num_accepted_tokens - 1` (IS_SPEC_DECODING, fused_sigmoid_gating.py:105-110).
    public static func stateAfterAccepting(_ accepted: Int, states: MLXArray, initial: MLXArray) -> MLXArray {
        accepted == 0 ? initial.asType(.float32) : states[0..., accepted - 1]
    }

    /// Causal depthwise short-conv state after consuming `accepted` of the new tokens: the last (K-1) rows of
    /// [prev ++ x[:accepted]]. prev: [B, K-1, C]; x: [B, T, C]. (Qwen35 `checkpointAfter` does the same slice.)
    public static func convStateAfterAccepting(_ accepted: Int, prev: MLXArray, x: MLXArray) -> MLXArray {
        let km1 = prev.dim(1)
        return concatenated([prev, x], axis: 1)[0..., accepted ..< (accepted + km1)]
    }
}
