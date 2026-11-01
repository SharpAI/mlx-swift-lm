// SharpAI fork overlay: exaone4_5 checkpoint key sanitization. Not part of upstream.
//
// Keeps HF key names, so module property names must match the checkpoint. This pass only drops
// keys that no loaded module uses.
// UNVERIFIED: the MTP key prefix ("mtp") and the rotary buffer name are assumed. Confirm them
// against a real exaone4_5 safetensors index.

import Foundation
import MLX

public enum SharpAIExaone45Sanitizer {

    /// Drops multi-token-prediction and precomputed rotary keys. Other keys are unchanged.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { entry in
            let key = entry.key
            let isMTP = key.split(separator: ".").contains("mtp")
            let isRotary = key.hasSuffix("rotary_emb.inv_freq")
            return !isMTP && !isRotary
        }
    }
}
