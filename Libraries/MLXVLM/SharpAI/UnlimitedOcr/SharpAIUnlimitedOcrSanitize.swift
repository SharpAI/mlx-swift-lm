// SharpAI fork overlay: unlimited_ocr checkpoint key sanitization. Not part of upstream.
//
// The checkpoint key names already match the module tree in SharpAIUnlimitedOcrModel
// (model.embed_tokens, model.layers.N.*, model.sam_model.*, model.vision_model.*,
// model.projector.layers.*, model.image_newline, model.view_seperator, lm_head).
// Verified against: https://huggingface.co/baidu/Unlimited-OCR/raw/main/model.safetensors.index.json
// (key prefixes checked on a partial listing of the index; the full index was not read).
//
// Only two things change: PyTorch conv weights (out, in, kH, kW) become MLX layout
// (out, kH, kW, in), and the rotary inv_freq buffers are dropped. Expert stacking happens in
// SharpAIUnlimitedOcrModel.sanitize(weights:).

import Foundation
import MLX

public enum SharpAIUnlimitedOcrSanitizer {

    /// Conv weights stored in PyTorch layout in the checkpoint, matched by suffix.
    static let convWeightSuffixes = [
        "sam_model.patch_embed.proj.weight",
        "sam_model.neck.c0.weight",
        "sam_model.neck.c2.weight",
        "sam_model.net_2.weight",
        "sam_model.net_3.weight",
        "vision_model.embeddings.patch_embedding.weight",
    ]

    static func isConvWeight(_ key: String) -> Bool {
        convWeightSuffixes.contains { key.hasSuffix($0) }
    }

    /// Neck children are keyed "c0", "n1", "c2", "n3" in the module tree (numeric keys would unflatten
    /// to an array); the checkpoint names them 0 to 3.
    static func renameNeckKey(_ key: String) -> String {
        guard let range = key.range(of: "sam_model.neck.") else { return key }
        let rest = key[range.upperBound...]
        guard let first = rest.first, let index = Int(String(first)), (0 ... 3).contains(index),
            rest.dropFirst().first == "."
        else { return key }
        let prefix = index % 2 == 0 ? "c" : "n"
        return String(key[..<range.upperBound]) + prefix + String(rest)
    }

    /// Converts conv weights to MLX layout and removes inv_freq buffers. Other keys pass through.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        for (originalKey, value) in weights {
            let key = renameNeckKey(originalKey)
            if key.contains("rotary_emb.inv_freq") {
                continue
            }
            if isConvWeight(key), value.ndim == 4 {
                out[key] = value.transposed(0, 2, 3, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }
}
