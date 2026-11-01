// SharpAI fork overlay: muse_glimmer_assistant checkpoint key sanitize. Not part of upstream
// mlx-swift-lm.
//
// UNVERIFIED: the repo has no checkpoint or key list for this drafter, so no renaming is done.
// Keys keep their HF names. Only rotary `inv_freq` buffers are dropped, as the other SharpAI
// drafter sanitizers do.

import Foundation
import MLX

enum SharpAIMuseGlimmerAssistantSanitizer {

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb") }
    }
}
