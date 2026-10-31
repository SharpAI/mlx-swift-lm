// SL-41: fork-only registration of the `deepseek_v4` model_type, kept out of the upstream-owned
// `LLMModelFactory.swift` literal so a future upstream DeepseekV4 cannot collide with it.
//
// Policy (owner decision, SL-41): if upstream ever registers its own `deepseek_v4`, OURS WINS.
// The single hook is the last line of the `LLMTypeRegistry.shared` literal:
//     ].merging(SharpAIDeepseekV4Overlay.creators) { _, ours in ours })
// `applying(to:)` is the same merge in function form so the policy is unit-testable.
// The model_type string `deepseek_v4` and the HF architecture names are external and unchanged.
//
// upstream-watch: Libraries/MLXLLM/LLMModelFactory.swift (a `"deepseek_v4"` entry, or any
// DeepseekV4* type/file, appearing upstream; see scripts/conformance/upstream/UPSTREAM_WATCH.md).

import Foundation
import MLXLMCommon

public enum SharpAIDeepseekV4Overlay {
    public typealias Creator = (Data) throws -> any LanguageModel

    /// Same decode-validate-construct sequence as the (private) `create` helper in
    /// `LLMModelFactory.swift`, so `deepseek_v4` behaves exactly as it did when it was in the literal.
    private static func makeCreator() -> Creator {
        { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIDeepseekV4Configuration.self, from: data)
            if let validating = configuration as? ModelConfigurationValidating {
                try validating.validateModelConfiguration()
            }
            return SharpAIDeepseekV4Model(configuration)
        }
    }

    /// model_type -> creator entries this overlay owns.
    public static var creators: [String: Creator] {
        // The conformance scripts (gen_report.py, gen_expectations.py) read the keys below
        // statically: one `"model_type": makeCreator()` entry per line, 12 spaces of indent.
        [
            "deepseek_v4": makeCreator()
        ]
    }

    /// `base` with this overlay's entries applied. Ours wins on a key clash.
    public static func applying(to base: [String: Creator]) -> [String: Creator] {
        base.merging(creators) { _, ours in ours }
    }
}
