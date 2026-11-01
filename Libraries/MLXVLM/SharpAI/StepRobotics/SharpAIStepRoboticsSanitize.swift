// SharpAI fork overlay: StepVL checkpoint key remap. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

/// Maps StepVL checkpoint keys onto `SharpAIStepRoboticsModel` module paths.
///
/// Checkpoint layout, from vLLM's `StepVLForConditionalGeneration.hf_to_vllm_mapper` and
/// `Step3TextModel.load_weights`:
/// - `model.*` is the text decoder. It moves under `language_model.`.
/// - `lm_head.*` moves under `language_model.`.
/// - `vision_model.*` and `vit_large_projector.*` stay at the top level.
/// - Perception Encoder fused attention and MLP names are renamed to the Swift module keys.
/// - Text `q_proj`, `k_proj` and `v_proj` are concatenated into `qkv_proj`.
/// - Routed expert weights `moe.gate_proj|up_proj|down_proj` move under `moe.switch_mlp`.
/// - Conv weights are transposed from PyTorch `[out, in, kH, kW]` to MLX `[out, kH, kW, in]`.
/// The `rotary_emb` and `freqs_cache` tensors are dropped. They are not parameters.
public enum SharpAIStepRoboticsSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.contains("rotary_emb") || key.hasSuffix("freqs_cache") {
                continue
            }
            let name = mappedName(key)
            renamed[name] = patchLayout(name, value)
        }
        return fuseTextQKV(renamed)
    }

    /// Module path for one checkpoint key. Pure rename, no layout change.
    static func mappedName(_ key: String) -> String {
        var name = key
        if name.hasPrefix("model.") || name.hasPrefix("lm_head.") {
            name = "language_model." + name
        }

        if name.hasPrefix("vision_model.") {
            name = name
                .replacingOccurrences(of: ".attn.in_proj_weight", with: ".attn.qkv_proj.weight")
                .replacingOccurrences(of: ".attn.in_proj_bias", with: ".attn.qkv_proj.bias")
                .replacingOccurrences(of: ".mlp.c_fc.", with: ".mlp.fc1.")
                .replacingOccurrences(of: ".mlp.c_proj.", with: ".mlp.fc2.")
        }

        for proj in ["gate_proj", "up_proj", "down_proj"] {
            name = name.replacingOccurrences(
                of: ".moe.\(proj).", with: ".moe.switch_mlp.\(proj).")
        }
        return name
    }

    /// Moves conv weights from PyTorch `[out, in, kH, kW]` to MLX `[out, kH, kW, in]`.
    /// UNVERIFIED: assumes the checkpoint uses PyTorch layout, which is the HF convention.
    static func patchLayout(_ name: String, _ value: MLXArray) -> MLXArray {
        guard value.ndim == 4, name.hasPrefix("vision_model.") else { return value }
        let convSuffixes = ["conv1.weight", "vit_downsampler1.weight", "vit_downsampler2.weight"]
        guard convSuffixes.contains(where: { name.hasSuffix($0) }) else { return value }
        return value.transposed(0, 2, 3, 1)
    }

    /// Concatenates `q_proj`, `k_proj`, `v_proj` into `qkv_proj` along the output axis.
    /// The order matches vLLM's `qkv_params_mapping`: q, then k, then v.
    static func fuseTextQKV(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights
        let qSuffix = ".self_attn.q_proj.weight"
        for key in weights.keys where key.hasSuffix(qSuffix) {
            let prefix = String(key.dropLast(qSuffix.count))
            let kKey = prefix + ".self_attn.k_proj.weight"
            let vKey = prefix + ".self_attn.v_proj.weight"
            guard let k = weights[kKey], let v = weights[vKey], let q = weights[key] else {
                continue
            }
            out[prefix + ".self_attn.qkv_proj.weight"] = concatenated([q, k, v], axis: 0)
            out.removeValue(forKey: key)
            out.removeValue(forKey: kKey)
            out.removeValue(forKey: vKey)
        }
        return out
    }
}
