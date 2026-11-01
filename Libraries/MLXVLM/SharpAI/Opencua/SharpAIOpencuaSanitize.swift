// SharpAI fork overlay: opencua key renaming. Not part of upstream mlx-swift-lm.
//
// Runs before Qwen25VL.sanitize. The prefixes follow vLLM's OpenCUA weight mapper.
// Checked against xlangai/OpenCUA-7B model.safetensors.index.json: its keys are already
// `language_model.model.*`, `language_model.lm_head.*` and `vision_tower.*`, which renameKeys
// leaves unchanged. The rules above only matter for other OpenCUA layouts.

import Foundation
import MLX

enum SharpAIOpencuaSanitizer {

    private static let prefixes: [(from: String, to: String)] = [
        ("model.visual.", "vision_tower."),
        ("visual.", "vision_tower."),
        ("model.language_model.", "model."),
    ]

    static func renameKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        for (key, value) in weights {
            renamed[renameKey(key)] = value
        }
        return renamed
    }

    static func renameKey(_ key: String) -> String {
        for prefix in prefixes where key.hasPrefix(prefix.from) {
            return prefix.to + String(key.dropFirst(prefix.from.count))
        }
        return key
    }
}
