// Conformance gap pins, part 1: Rnj1 / gemma3_text `final_logit_softcapping` (#254 section 2).
//
// Spec: with `final_logit_softcapping = cap` the model's logits must equal
// `cap * tanh(unbounded_logits / cap)`, where `unbounded_logits` are what the SAME weights produce
// without the key. This is stricter than ConformanceForwardTests.gemma3TextSoftcap (which only checks
// |logit| <= cap): it also catches a wrong formula (clipping instead of tanh, a missing 1/cap, ...).
//
// Known gap: `Gemma3TextConfiguration` does not decode the key and the model ends in a bare
// `lmHead`. The wrapper is `isIntermittent`, so the test is green before AND after the fix and an
// upstream sync that adds softcapping is never blocked. Delete the wrapper in the PR that
// implements it. upstream-watch: `final_logit_softcapping` in Gemma3Text.swift (mlx-swift-lm).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

struct ConformanceGapSoftcapTests {
    typealias Ref = SharpAIOverlayGapReference

    @Test("gemma3_text logits == cap * tanh(unbounded / cap), same weights")
    func softcapMatchesReference() async throws {
        let cap: Float = 5
        let plain = try #require(
            try await ConformanceBuild.llm(ConformanceForwardTests.gemma3TextJSON(softcap: nil), modelType: "gemma3_text")
                as? Gemma3TextModel)
        let capped = try #require(
            try await ConformanceBuild.llm(ConformanceForwardTests.gemma3TextJSON(softcap: Double(cap)), modelType: "gemma3_text")
                as? Gemma3TextModel)
        // Identical weights in both models (the two builds draw different random weights).
        capped.update(parameters: plain.parameters())
        eval(capped)

        // Scale the output projection so unbounded logits reach ~2*cap: the tanh is then clearly
        // non-linear (a tanh vs clip vs identity mix-up cannot pass by accident).
        let tokens = [1, 2, 3, 4]
        let probe = abs(try ConformanceBuild.logits(plain, tokens: tokens).asType(.float32)).max().item(Float.self)
        let scale = 2 * cap / probe
        for m in [plain, capped] {
            m.update(parameters: ModuleParameters.unflattened([("lm_head.weight", m.lmHead.weight * scale)]))
            eval(m)
        }
        let unbounded = try ConformanceBuild.logits(plain, tokens: tokens).asType(.float32)
        let actual = try ConformanceBuild.logits(capped, tokens: tokens).asType(.float32)
        let expected = Ref.tanhSoftcap(unbounded, cap: cap)

        // Precondition of the test itself (outside the known-issue): the case is discriminating.
        #expect(abs(unbounded).max().item(Float.self) > 1.5 * cap, "unbounded logits must exceed the cap")
        #expect(abs(expected).max().item(Float.self) <= cap + 1e-4)

        let maxDiff = abs(actual - expected).max().item(Float.self)
        ConformanceGap.knownIssue("Gemma3TextConfiguration ignores final_logit_softcapping") {
            ConformanceGap.expect(maxDiff <= 1e-3, "max |model - cap*tanh(unbounded/cap)| = \(maxDiff)")
        }
    }
}
