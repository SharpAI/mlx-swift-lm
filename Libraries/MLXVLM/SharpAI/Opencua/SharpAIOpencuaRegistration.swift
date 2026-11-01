// SharpAI fork overlay: opencua creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as `create` in VLMModelFactory. VLMModelFactory is not
// edited, so the consumer merges `creators` itself. No processor is registered yet: the
// "OpenCUAProcessor" processor_class needs its own prompt path (<|media_placeholder|>, no vision
// start/end wrapping), which Qwen25VLProcessor does not provide.

import Foundation
import MLXLMCommon

public enum SharpAIOpencuaOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "opencua": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIOpencuaConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIOpencuaModel(try configuration.qwen25VLConfiguration(from: data))
        }
    ]
}
