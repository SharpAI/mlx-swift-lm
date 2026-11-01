// SharpAI fork overlay: opencua model. Not part of upstream mlx-swift-lm.
//
// OpenCUA uses the Qwen2.5-VL vision tower and text decoder. This subclass reuses
// Qwen25VL and only renames checkpoint keys before Qwen25VL's own sanitize runs.

import Foundation
import MLX
import MLXLMCommon

public class SharpAIOpencuaModel: Qwen25VL {

    public override init(_ config: Qwen25VLConfiguration) {
        super.init(config)
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        super.sanitize(weights: SharpAIOpencuaSanitizer.renameKeys(weights))
    }
}
