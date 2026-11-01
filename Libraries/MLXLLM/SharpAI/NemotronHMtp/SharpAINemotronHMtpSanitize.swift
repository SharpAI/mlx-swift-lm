// SharpAI fork overlay: nemotron_h_mtp checkpoint key sanitization. Not part of upstream.
//
// Drops the `mtp.*` head tensors before the NemotronH sanitize runs. The text backbone
// keeps the HF key names (`backbone.*`, `lm_head.*`), so nothing else is remapped here.

import Foundation
import MLX

public enum SharpAINemotronHMtpSanitizer {

    /// Checkpoint prefix of the multi-token-prediction head tensors.
    public static let mtpPrefix = "mtp."

    /// Returns the weights without any `mtp.*` tensor.
    public static func dropMTPWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.hasPrefix(mtpPrefix) }
    }
}
