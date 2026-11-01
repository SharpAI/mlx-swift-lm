// SharpAI fork overlay: step_robotics (StepVL) creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as `create` in VLMModelFactory. VLMModelFactory and
// VLMTypeRegistry are not edited, so the consumer merges `creators` itself. No processor is
// registered: StepVL's image processor (crop and patch logic in vLLM's `processors/step3_vl.py`)
// is not implemented here.

import Foundation
import MLXLMCommon

public enum SharpAIStepRoboticsOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "step_robotics": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIStepRoboticsConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIStepRoboticsModel(configuration)
        }
    ]
}
