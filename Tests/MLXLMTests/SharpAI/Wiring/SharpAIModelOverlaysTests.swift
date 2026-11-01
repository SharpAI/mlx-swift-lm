// SharpAI SL-50: the LLM overlay aggregate wired into LLMTypeRegistry (upstream wins).
// Not part of upstream.

import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("SharpAI LLM overlay wiring (SL-50)")
struct SharpAIModelOverlaysTests {

    /// Overlay keys that are deliberately NOT registered; see the header of SharpAIModelOverlays.swift.
    static let excluded: [String] = ["dots3_note_mtp", "k3_dspark", "kimi_k3_mtp", "minimax_m2", "minimax_m3_dspark", "minimax_m3_mtp", "muse_spark_dspark", "sarvam_mla", "unknown", "params.json", "inkling_model", "inkling_mm_model", "pangu_ultra_moe_mtp", "step5", "modernbert", "llama_bidirec", "deepseek_v3_bidirec", "openai_privacy_filter", "jina_embeddings_v5"]

    @Test("every wired key resolves in LLMTypeRegistry.shared")
    func everyWiredKeyIsRegistered() async {
        #expect(!SharpAIModelOverlays.llmWiredKeys.isEmpty)
        for key in SharpAIModelOverlays.llmWiredKeys.sorted() {
            #expect(await LLMTypeRegistry.shared.contains(key), "\(key) is wired but not registered")
        }
    }

    @Test("the aggregate holds a creator for exactly the wired keys")
    func aggregateMatchesWiredKeys() {
        #expect(Set(SharpAIModelOverlays.llm.keys) == SharpAIModelOverlays.llmWiredKeys)
    }

    @Test("excluded overlay keys stay unregistered, so contains() does not claim support")
    func excludedKeysAreNotRegistered() async {
        for key in Self.excluded {
            #expect(!SharpAIModelOverlays.llmWiredKeys.contains(key), "\(key) must not be wired")
            #expect(await !LLMTypeRegistry.shared.contains(key), "\(key) is registered but excluded")
        }
    }

    @Test("upstream wins: on a clash the upstream value is kept, other keys are added")
    func upstreamWinsPolicy() {
        let merged = ["a": 1, "b": 2].merging(["a": 100, "c": 3], uniquingKeysWith: SharpAIModelOverlays.upstreamWins)
        #expect(merged == ["a": 1, "b": 2, "c": 3])
    }

    @Test("the factory merges the aggregate with the upstream-wins function, and the V4 hook stays last")
    func factoryWiring() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Libraries/MLXLLM/LLMModelFactory.swift")
        let src = try String(contentsOf: url, encoding: .utf8)
        let overlay = try #require(
            src.range(of: ".merging(SharpAIModelOverlays.llm, uniquingKeysWith: SharpAIModelOverlays.upstreamWins)"))
        let v4 = try #require(src.range(of: ".merging(SharpAIDeepseekV4Overlay.creators)"))
        #expect(overlay.lowerBound < v4.lowerBound, "the ours-wins V4 merge must come after the upstream-wins merge")
    }
}
