// SharpAI fork overlay: nemo_speechlm checkpoint key sanitization. Not part of upstream.
//
// Verified against: vLLM nemotron3_5_transcribe.py (hf_to_vllm_mapper, _unpack_experts,
// _mtp_weights, load_weights) and the checkpoint header of
// https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (model.safetensors,
// 1395 tensors: llm.model.*, llm.lm_head.*, llm.mtp.*, perception.*, ctc_timestamp.*).
//
// Rules, in the order they apply:
//  - `._extra_state` (TransformerEngine bookkeeping), `llm.mtp.*` (MTP head, not ported) and
//    `ctc_timestamp.*` (timestamp head, not ported) are dropped.
//  - `llm.model.embed_tokens.weight` -> `backbone.embeddings.weight`,
//    `llm.model.norm.weight` -> `backbone.norm_f.weight`, `llm.lm_head.weight` -> `lm_head.weight`.
//  - `llm.model.layers.N.*` -> `backbone.layers.N.*`, with `mixer._fp32_params.` flattened to `mixer.`.
//  - Stacked routed experts `mixer.experts.gate_and_up_projs` (E, hidden, inter) and
//    `mixer.experts.down_projs` (E, inter, hidden) become per-expert `up_proj.weight` (inter, hidden)
//    and `down_proj.weight` (hidden, inter), by taking expert e and transposing.
//    NemotronHModel.sanitize then stacks them into `switch_mlp`.
//  - `perception.*` keys are kept under their NeMo names. Other keys pass through unchanged.
//
// The conv1d axis swap and the lm_head/MTP filtering are left to NemotronHModel.sanitize.

import Foundation
import MLX

public enum SharpAINemoSpeechlmSanitizer {

    /// Returns the weights keyed by the names the decoder and audio encoder expect.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            guard !key.isEmpty else {
                throw SharpAINemoSpeechlmError.emptyWeightKey
            }
            if key.hasSuffix("._extra_state") || key.hasPrefix("llm.mtp.")
                || key.hasPrefix("ctc_timestamp.")
            {
                continue
            }
            guard key.hasPrefix("llm.") else {
                sanitized[key] = value
                continue
            }
            try mapDecoderWeight(key: key, value: value, into: &sanitized)
        }

        return sanitized
    }

    private static func mapDecoderWeight(
        key: String, value: MLXArray, into sanitized: inout [String: MLXArray]
    ) throws {
        if key == "llm.lm_head.weight" {
            sanitized["lm_head.weight"] = value
            return
        }
        if key == "llm.model.embed_tokens.weight" {
            sanitized["backbone.embeddings.weight"] = value
            return
        }
        if key == "llm.model.norm.weight" {
            sanitized["backbone.norm_f.weight"] = value
            return
        }

        let layerPrefix = "llm.model.layers."
        guard key.hasPrefix(layerPrefix) else {
            throw SharpAINemoSpeechlmError.unmappedDecoderWeight(key)
        }
        let rest = key.dropFirst(layerPrefix.count)
        guard let dot = rest.firstIndex(of: "."), Int(rest[..<dot]) != nil else {
            throw SharpAINemoSpeechlmError.unmappedDecoderWeight(key)
        }
        let layer = String(rest[..<dot])
        let tail = String(rest[rest.index(after: dot)...])
            .replacingOccurrences(of: "mixer._fp32_params.", with: "mixer.")
        let base = "backbone.layers.\(layer).\(tail)"

        if tail.hasSuffix("experts.gate_and_up_projs") {
            try unpackExperts(stacked: value, prefix: String(base.dropLast("gate_and_up_projs".count)),
                projection: "up_proj", into: &sanitized)
        } else if tail.hasSuffix("experts.down_projs") {
            try unpackExperts(stacked: value, prefix: String(base.dropLast("down_projs".count)),
                projection: "down_proj", into: &sanitized)
        } else {
            sanitized[base] = value
        }
    }

    /// `stacked` is (experts, in, out) in NeMo's `x @ W` layout. Expert e becomes (out, in).
    private static func unpackExperts(
        stacked: MLXArray, prefix: String, projection: String,
        into sanitized: inout [String: MLXArray]
    ) throws {
        guard stacked.ndim == 3 else {
            throw SharpAINemoSpeechlmError.unmappedDecoderWeight(
                "\(prefix)\(projection): expected a rank-3 expert tensor")
        }
        // `prefix` ends with "experts." so the per-expert names are "<prefix><e>.<projection>.weight".
        for expert in 0 ..< stacked.dim(0) {
            sanitized["\(prefix)\(expert).\(projection).weight"] =
                stacked[expert].transposed(1, 0)
        }
    }
}
