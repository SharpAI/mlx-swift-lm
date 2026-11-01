// SharpAI fork overlay: qwen3_omni_moe key sanitization. Not part of upstream mlx-swift-lm.

import Foundation
import MLX
import MLXLMCommon

/// Maps `thinker.*` checkpoint names onto `SharpAIQwen3OmniMoeThinker` module paths.
/// Talker and code2wav weights are dropped. Per-expert MoE weights are stacked into SwitchGLU.
/// UNVERIFIED: the checkpoint key layout (`thinker.visual.merger_list.*`, per-expert
/// `mlp.experts.N.*`) follows the transformers naming as recalled.
public enum SharpAIQwen3OmniMoeSanitize {

    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIQwen3OmniMoeConfiguration
    ) -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights {
            guard let name = checkpointName(key) else { continue }
            renamed[name] = value
        }

        if config.textConfiguration.tieWordEmbeddings {
            renamed = renamed.filter { !$0.key.contains(".lm_head.") }
        }
        renamed = renamed.filter { !$0.key.contains("rotary_emb.inv_freq") }

        stackExperts(in: &renamed, config: config)

        // PyTorch conv2d weights are [out, in, kH, kW]. MLX expects [out, kH, kW, in].
        let convInputChannels: [(name: String, inChannels: Int)] = [
            ("audio_tower.conv2d1.weight", 1),
            ("audio_tower.conv2d2.weight", config.audioConfiguration.downsampleHiddenSize),
            ("audio_tower.conv2d3.weight", config.audioConfiguration.downsampleHiddenSize),
        ]
        for (name, inChannels) in convInputChannels {
            guard let weight = renamed[name], weight.ndim == 4 else { continue }
            if weight.dim(1) == inChannels, weight.dim(3) != inChannels {
                renamed[name] = weight.transposed(0, 2, 3, 1)
            }
        }

        // Patch-embed transposes and position_ids removal for the ViT.
        return Qwen3VLVision.VisionModel.sanitize(
            weights: renamed, config: config.visionConfiguration)
    }

    /// Returns the module-path name, or nil for weights this model does not load.
    static func checkpointName(_ key: String) -> String? {
        var name = key
        if name.hasPrefix("thinker.") {
            name = String(name.dropFirst("thinker.".count))
        } else if name.hasPrefix("talker.") || name.hasPrefix("code2wav.") {
            return nil
        }

        // The Omni patch merger holds `ln_q` and an `mlp` list (Linear, GELU, Linear), so its
        // checkpoint keys are `ln_q`, `mlp.0` and `mlp.2`. Qwen3VLVision.PatchMerger names them
        // `norm`, `linear_fc1` and `linear_fc2`.
        if name.hasPrefix("visual.merger_list.") {
            return "vision_tower.deepstack_merger_list."
                + mergerSubKeys(String(name.dropFirst("visual.merger_list.".count)))
        }
        if name.hasPrefix("visual.merger.") {
            return "vision_tower.merger." + mergerSubKeys(String(name.dropFirst("visual.merger.".count)))
        }
        if name.hasPrefix("visual.") {
            return "vision_tower." + String(name.dropFirst("visual.".count))
        }
        if name.hasPrefix("model.") {
            return "language_model." + name
        }
        if name.hasPrefix("lm_head.") {
            return "language_model." + name
        }
        return name
    }

    /// Renames Omni merger sub-keys to the Qwen3-VL names. Keys already in the Qwen3-VL form are
    /// left unchanged.
    private static func mergerSubKeys(_ rest: String) -> String {
        rest.replacingOccurrences(of: "ln_q.", with: "norm.")
            .replacingOccurrences(of: "mlp.0.", with: "linear_fc1.")
            .replacingOccurrences(of: "mlp.2.", with: "linear_fc2.")
    }

    /// Joins `layers.N.mlp.experts.E.{gate,up,down}_proj.weight` for every expert E into
    /// `layers.N.mlp.switch_mlp.{...}.weight`, as `Qwen3VLMoELanguage` expects.
    private static func stackExperts(
        in weights: inout [String: MLXArray], config: SharpAIQwen3OmniMoeConfiguration
    ) {
        let text = config.textConfiguration
        for layer in 0 ..< text.numHiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let keys = (0 ..< text.numExperts).map {
                    "\(prefix).experts.\($0).\(projection).weight"
                }
                guard keys.allSatisfy({ weights[$0] != nil }) else { continue }
                let experts = keys.map { weights.removeValue(forKey: $0)! }
                weights["\(prefix).switch_mlp.\(projection).weight"] = MLX.stacked(experts)
            }
        }
    }
}
