// SharpAI fork overlay: inkling_model checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Source of the names: the `conversion_mapping` entry for `inkling_mm_model` in transformers
// (`src/transformers/conversion_mapping.py`), checked against the tensor shapes in the
// thinkingmachines/Inkling safetensors headers (bf16, 6144 hidden, 201024 vocab, 256 experts).
//
// - `model.llm.*` is the text backbone. It is renamed to the `model.*` keys this module builds.
// - `w13` tensors interleave gate and up rows (even = gate, odd = up), as transformers' `Interleave`
//   plus `Chunk` read them. They are split into separate gate and up tensors.
// - `model.audio.*`, `model.visual.*` and `model.mtp.*` are dropped (text-only load).
// - Other keys pass through unchanged.
// - ModelOpt NVFP4 routed-expert tensors (`w13_weight` / `w2_weight` as packed uint8, with `.scale`,
//   `.scale2`, `.input_amax` and `.original_shape` siblings) are dequantized to BF16 by
//   `dequantizeNVFP4`, before the renames above. The nibble order and scale semantics were checked
//   numerically against the BF16 experts of https://huggingface.co/thinkingmachines/Inkling-Small
//   (the NVFP4 counterpart is https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4).

import Foundation
import MLX

public enum SharpAIInklingModelSanitizer {

    private static let llmPrefix = "model.llm."
    private static let droppedPrefixes = ["model.audio.", "model.visual.", "model.mtp."]

    private static let topLevelRenames: [String: String] = [
        "embed.weight": "model.embed_tokens.weight",
        "embed_norm.weight": "model.embed_norm.weight",
        "norm.weight": "model.norm.weight",
        "unembed.weight": "lm_head.weight",
    ]

    /// Renames inside `model.llm.layers.<n>.`. Sparse-layer MLP keys become `moe.*`.
    private static let layerRenames: [String: String] = [
        "attn_norm.weight": "input_layernorm.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "attn.wq_du.weight": "self_attn.q_proj.weight",
        "attn.wk_dv.weight": "self_attn.k_proj.weight",
        "attn.wv_dv.weight": "self_attn.v_proj.weight",
        "attn.wr_du.weight": "self_attn.r_proj.weight",
        "attn.wo_ud.weight": "self_attn.o_proj.weight",
        "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.k_norm.weight": "self_attn.k_norm.weight",
        "attn.k_sconv.weight": "self_attn.k_sconv.weight",
        "attn.v_sconv.weight": "self_attn.v_sconv.weight",
        "attn.rel_logits_proj.proj": "self_attn.rel_logits_proj.proj",
        "attn_sconv.weight": "attn_sconv.weight",
        "mlp_sconv.weight": "mlp_sconv.weight",
        "mlp.w2_md.weight": "mlp.down_proj.weight",
        "mlp.global_scale": "mlp.global_scale",
        "mlp.gate.weight": "moe.gate.weight",
        "mlp.gate.global_scale": "moe.gate.global_scale",
        "mlp.gate.bias": "moe.gate.e_score_correction_bias",
        "mlp.experts.w2_weight": "moe.experts.down_proj.weight",
        "mlp.shared_experts.shared_w2_weight": "moe.shared_experts.down_proj.weight",
    ]

    /// Fused gate-up tensors: source suffix, gate target, up target, and the row axis.
    private static let fusedSplits:
        [String: (gate: String, up: String, axis: Int)] = [
            "mlp.w13_dn.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight", 0),
            "mlp.experts.w13_weight": (
                "moe.experts.gate_proj.weight", "moe.experts.up_proj.weight", 1
            ),
            "mlp.shared_experts.shared_w13_weight": (
                "moe.shared_experts.gate_proj.weight", "moe.shared_experts.up_proj.weight", 1
            ),
        ]

    /// ModelOpt NVFP4 siblings of a packed expert tensor. They are consumed by dequantization.
    private static let nvfp4Siblings = [".scale", ".scale2", ".input_amax", ".original_shape"]

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        // Dequantize each NVFP4 expert tensor first. Its metadata siblings are dropped.
        var dequantized: [String: MLXArray] = [:]
        var consumed = Set<String>()
        for (key, packed) in weights where key.hasPrefix(llmPrefix) {
            guard let blockScale = weights[key + ".scale"] else {
                continue
            }
            guard let globalScale = weights[key + ".scale2"] else {
                preconditionFailure(
                    "NVFP4 tensor \(key) has a block scale but no .scale2 global scale")
            }
            dequantized[key] = dequantizeNVFP4(
                packed: packed, blockScale: blockScale, globalScale: globalScale)
            for suffix in nvfp4Siblings {
                consumed.insert(key + suffix)
            }
        }

        for (key, raw) in weights {
            if consumed.contains(key) || droppedPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }
            let value = dequantized[key] ?? raw
            guard key.hasPrefix(llmPrefix) else {
                out[key] = value
                continue
            }

            let rest = String(key.dropFirst(llmPrefix.count))
            if let renamed = topLevelRenames[rest] {
                out[renamed] = value
                continue
            }

            guard let parsed = splitLayer(rest) else {
                out[key] = value
                continue
            }
            let sub = parsed.1
            let prefix = "model.layers.\(parsed.0)."

            if let renamed = layerRenames[sub] {
                out[prefix + renamed] = value
            } else if let split = fusedSplits[sub] {
                let (gate, up) = deinterleave(value, axis: split.axis)
                out[prefix + split.gate] = gate
                out[prefix + split.up] = up
            } else {
                out[key] = value
            }
        }
        return out
    }

    /// Splits `layers.<n>.<rest>` into `(n, rest)`.
    private static func splitLayer(_ rest: String) -> (Int, String)? {
        let parts = rest.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "layers", let layer = Int(parts[1]) else {
            return nil
        }
        return (layer, String(parts[2]))
    }

    /// Gate and up rows alternate along `axis`: row `2a` is gate, row `2a + 1` is up.
    /// Returns the gate and up tensors, each with half the rows.
    static func deinterleave(_ tensor: MLXArray, axis: Int) -> (MLXArray, MLXArray) {
        var grouped = tensor.shape
        precondition(grouped[axis] % 2 == 0, "fused gate-up axis must be even")
        let half = grouped[axis] / 2
        grouped.replaceSubrange(axis ... axis, with: [half, 2])
        let rows = tensor.reshaped(grouped)
        let gate = MLX.take(rows, MLXArray([Int32(0)]), axis: axis + 1).squeezed(axis: axis + 1)
        let up = MLX.take(rows, MLXArray([Int32(1)]), axis: axis + 1).squeezed(axis: axis + 1)
        return (gate, up)
    }

    /// Dequantizes ModelOpt NVFP4 to BF16: `value = e2m1(code) * e4m3(blockScale) * globalScale`.
    ///
    /// - `packed`: uint8 `[..., C / 2]`. Element `2i` is the low nibble of byte `i`, and element
    ///   `2i + 1` is the high nibble.
    /// - `blockScale`: uint8 bit patterns of float8 e4m3, `[..., C / 16]`. Each one covers 16
    ///   consecutive elements. The loader maps `F8_E4M3` to uint8, so `MLXFast.fromFp8` decodes it.
    /// - `globalScale`: float32 `[E]`, where `E` is the leading dimension. It multiplies the result.
    ///
    /// Checked against the BF16 experts of thinkingmachines/Inkling-Small. Both `w13` and `w2`
    /// decode to within about 8% relative error, the size of FP4 rounding, with the nibble order
    /// and the multiplicative `globalScale` above.
    static func dequantizeNVFP4(
        packed: MLXArray, blockScale: MLXArray, globalScale: MLXArray
    ) -> MLXArray {
        precondition(
            packed.dtype == .uint8 && blockScale.dtype == .uint8,
            "NVFP4 packed weights and block scales must be uint8 bit patterns")
        precondition(packed.ndim >= 2, "NVFP4 expert tensors must have a leading expert axis")
        let lead = Array(packed.shape.dropLast())
        let cols = packed.dim(-1) * 2
        precondition(cols % 16 == 0, "NVFP4 rows must hold whole 16-element blocks")
        precondition(
            blockScale.shape == lead + [cols / 16],
            "NVFP4 block scale must have one e4m3 value per 16 elements")
        precondition(
            globalScale.size == lead[0],
            "NVFP4 global scale must have one value per expert")

        let low = packed & UInt8(0x0F)
        let high = packed >> UInt8(4)
        let codes = MLX.stacked([low, high], axis: -1).reshaped(lead + [cols])
        // FP4 E2M1 values indexed by the 4-bit code. Bit 3 is the sign, bits 0-2 the magnitude index.
        // Built per call, not as a static: a static MLXArray is non-Sendable shared mutable state
        // and fails strict concurrency checking.
        let e2m1Values = MLXArray(
            [Float(0), 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6])
        let magnitudes = MLX.take(e2m1Values, codes.asType(.int32), axis: 0)

        let blockScales = MLXFast.fromFp8(blockScale, dtype: .float32)
        let global = globalScale.asType(.float32).reshaped(
            [lead[0]] + Array(repeating: 1, count: lead.count + 1))
        let blocks = magnitudes.reshaped(lead + [cols / 16, 16])
        return (blocks * blockScales.expandedDimensions(axis: -1) * global)
            .reshaped(lead + [cols])
            .asType(.bfloat16)
    }
}
