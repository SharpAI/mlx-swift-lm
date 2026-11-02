// SharpAI fork overlay aggregate for the MLXVLM target. Not part of upstream mlx-swift-lm.
//
// Collects the SL-47 per-model overlays that are wired into `VLMTypeRegistry.shared` (SL-51).
// Policy (owner decision, SL-50): UPSTREAM WINS. `VLMModelFactory` merges these dictionaries with a closure
// that keeps the upstream value for any key the upstream-owned literal already registers.
//
// PR1 wires models whose vision and audio towers are not ported, plus one that reuses the upstream Qwen3.5
// processor. Everything else in `Libraries/MLXVLM/SharpAI` stays unregistered on purpose until it has a
// processor or a working creator (see docs/research/sl51-vlm-scout-2026-11-01.md).

import Foundation
import MLXLMCommon

public enum SharpAIVLMModelOverlays {

    /// Merge closure for `Dictionary.merging(_:uniquingKeysWith:)`: when both sides have a key, keep upstream.
    public static func upstreamWins<T>(_ upstream: T, _ ours: T) -> T { upstream }

    /// `model_type`s whose model runs on text only. They share ``SharpAITextOnlyProcessor``.
    public static let textOnlyKeys: Set<String> = [
        "interns1_pro",
        "interns2_mobius",
        "kimi_k3",
        "qwen4_exp",
        "step1",
    ]

    /// `model_type`s registered in `VLMTypeRegistry.shared` through this aggregate.
    /// `intern_s2_preview` subclasses upstream Qwen3.5 and keeps its real (Qwen3-VL) processor.
    /// One per line: `scripts/conformance/gen_expectations.py` and `missing-ops/gen_report.py` read this list.
    public static let vlmWiredKeys: Set<String> = [
        "intern_s2_preview",
        "interns1_pro",
        "interns2_mobius",
        "kimi_k3",
        "qwen4_exp",
        "step1",
    ]

    /// Creators for `vlmWiredKeys`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let vlm: [String: (Data) throws -> any LanguageModel] = {
        var all: [String: (Data) throws -> any LanguageModel] = [:]
        func add(_ source: [String: (Data) throws -> any LanguageModel]) {
            for (key, creator) in source where vlmWiredKeys.contains(key) { all[key] = creator }
        }
        add(SharpAIInternS2PreviewOverlay.creators)
        add(SharpAIInterns1ProOverlay.creators)
        add(SharpAIInterns2MobiusOverlay.creators)
        add(SharpAIKimiK3Overlay.creators)
        add(SharpAIQwen4ExpOverlay.creators)
        add(SharpAIStep1Overlay.creators)
        return all
    }()

    /// Processor types added to `VLMProcessorTypeRegistry.shared`.
    nonisolated(unsafe) public static let processors:
        [String: (Data, any Tokenizer) throws -> any UserInputProcessor] = [
            SharpAITextOnlyProcessor.processorType: { data, tokenizer in
                let configuration =
                    (try? JSONDecoder.json5().decode(
                        SharpAITextOnlyProcessorConfiguration.self, from: data))
                    ?? SharpAITextOnlyProcessorConfiguration()
                return SharpAITextOnlyProcessor(configuration, tokenizer: tokenizer)
            }
        ]
}
