// Conformance harness, tiers 1-2: build from a config-only JSON, random weights, run a
// forward pass. No checkpoint, no network. Tier 1 = the model builds and produces finite
// logits of the right shape. Tier 2 = a numeric property that shape checks cannot see.
//
// Known gaps are expressed with `withKnownIssue`: the test is green while the gap exists
// and goes RED when the gap is fixed (or upstream fixes it), which is the cue to delete the
// `withKnownIssue` wrapper -- so a fix is always paired with the test that proves it.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

enum ConformanceBuild {
    /// Build an LLM purely from config JSON via the public type registry, with the
    /// framework's random parameter initialisation.
    static func llm(_ json: String, modelType: String) async throws -> any LanguageModel {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), modelType: modelType)
        if let module = model as? Module { eval(module) }
        return model
    }

    static func logits(_ model: any LanguageModel, tokens: [Int]) throws -> MLXArray {
        let llm = try #require(model as? any LLMModel)
        let out = llm(MLXArray(tokens).reshaped(1, tokens.count), cache: nil)
        eval(out)
        return out
    }
}

struct ConformanceForwardTests {

    static func gemma3TextJSON(softcap: Double?) -> String {
        let cap = softcap.map { ", \"final_logit_softcapping\": \($0)" } ?? ""
        return """
            {"model_type": "gemma3_text", "hidden_size": 32, "num_hidden_layers": 2,
             "intermediate_size": 64, "num_attention_heads": 2, "head_dim": 16,
             "num_key_value_heads": 1, "vocab_size": 64, "sliding_window": 16,
             "sliding_window_pattern": 2, "max_position_embeddings": 128,
             "query_pre_attn_scalar": 16\(cap)}
            """
    }

    /// Tier 1: the build -> forward path itself works from config alone.
    @Test("gemma3_text builds from config JSON and yields finite logits")
    func gemma3TextForward() async throws {
        let model = try await ConformanceBuild.llm(Self.gemma3TextJSON(softcap: nil), modelType: "gemma3_text")
        let logits = try ConformanceBuild.logits(model, tokens: [1, 2, 3, 4])
        #expect(logits.shape == [1, 4, 64])
        #expect(logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// Tier 2: `final_logit_softcapping` (Rnj1 and other Gemma3-text derivatives) must bound the
    /// logits. SURVEY FINDING, confirmed by reading `Gemma3Text.swift` (the configuration has no
    /// such key and `callAsFunction` ends with a bare `lmHead`); this test makes it runnable.
    @Test("gemma3_text honours final_logit_softcapping")
    func gemma3TextSoftcap() async throws {
        let cap = 5.0
        let model = try await ConformanceBuild.llm(Self.gemma3TextJSON(softcap: cap), modelType: "gemma3_text")
        // Inflate the output projection so the unbounded logits are far above the cap.
        let text = try #require(model as? Gemma3TextModel)
        let w = text.lmHead.weight
        text.update(parameters: ModuleParameters.unflattened([("lm_head.weight", w * 1000)]))
        eval(text)
        let logits = try ConformanceBuild.logits(model, tokens: [1, 2, 3, 4]).asType(.float32)
        let maxAbs = abs(logits).max().item(Float.self)

        // KNOWN GAP: remove this wrapper in the PR that adds softcapping.
        withKnownIssue("Gemma3TextConfiguration ignores final_logit_softcapping") {
            #expect(maxAbs <= Float(cap) + 1e-3, "max |logit| = \(maxAbs), cap = \(cap)")
        }
    }
}
