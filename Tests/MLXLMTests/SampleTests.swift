// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon
import XCTest

public class SampleTests: XCTestCase {

    private func sampleCounts(sampler: TopPSampler, logits: MLXArray, draws: Int) -> [Int: Int] {
        var counts: [Int: Int] = [:]
        for _ in 0 ..< draws {
            let token = sampler.sample(logits: logits).item(Int.self)
            counts[token, default: 0] += 1
        }
        return counts
    }

    private func frequency(_ counts: [Int: Int], token: Int, draws: Int) -> Float {
        Float(counts[token, default: 0]) / Float(draws)
    }

    private func assertOnlySampled(
        _ counts: [Int: Int], allowedTokens: Set<Int>, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for token in counts.keys {
            XCTAssertTrue(
                allowedTokens.contains(token), "Unexpected sampled token: \(token)", file: file,
                line: line)
        }
    }

    func testTopKSamplerKeepsOnlyTopToken() {
        let sampler = TopPSampler(temperature: 1.0, topK: 1)
        let logits = MLXArray([0.1 as Float, 2.0 as Float, 1.0 as Float])[.newAxis, .ellipsis]

        for _ in 0 ..< 10 {
            let token = sampler.sample(logits: logits).item(Int.self)
            XCTAssertEqual(token, 1)
        }
    }

    func testTopPSamplerLowThresholdKeepsMaxToken() {
        let probs = MLXArray([0.9 as Float, 0.0 as Float, 0.0 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let sampler = TopPSampler(temperature: 1.0, topP: 0.3)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: 200)

        XCTAssertEqual(counts[0], 200)
        assertOnlySampled(counts, allowedTokens: [0])
    }

    func testTopPSamplerPartialMassKeepsExpectedDistribution() {
        let probs = MLXArray([0.0 as Float, 0.5 as Float, 0.4 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topP: 0.6)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [1, 2])
        XCTAssertEqual(frequency(counts, token: 1, draws: draws), 0.5556, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 2, draws: draws), 0.4444, accuracy: 0.06)
    }

    func testTopPSamplerHighThresholdKeepsExpectedDistribution() {
        let probs = MLXArray([0.0 as Float, 0.5 as Float, 0.4 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topP: 0.95)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [1, 2, 3])
        XCTAssertEqual(frequency(counts, token: 1, draws: draws), 0.5, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 2, draws: draws), 0.4, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.1, accuracy: 0.04)
    }

    func testTopKSamplerTopTwoKeepsExpectedDistribution() {
        let probs = MLXArray([0.6 as Float, 0.0 as Float, 0.1 as Float, 0.3 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, topK: 2)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [0, 3])
        XCTAssertEqual(frequency(counts, token: 0, draws: draws), 0.6667, accuracy: 0.06)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.3333, accuracy: 0.06)
    }

    func testMinPSamplerKeepsOnlyHighProbabilityToken() {
        let sampler = TopPSampler(temperature: 1.0, minP: 0.95)
        let logits = MLXArray([0.0 as Float, 0.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]

        for _ in 0 ..< 10 {
            let token = sampler.sample(logits: logits).item(Int.self)
            XCTAssertEqual(token, 2)
        }
    }

    func testMinPSamplerLowThresholdKeepsExpectedDistribution() {
        let probs = MLXArray([0.9 as Float, 0.0 as Float, 0.0 as Float, 0.1 as Float])[
            .newAxis, .ellipsis]
        let draws = 4000
        let sampler = TopPSampler(temperature: 1.0, minP: 0.05)
        let counts = sampleCounts(sampler: sampler, logits: log(probs), draws: draws)

        assertOnlySampled(counts, allowedTokens: [0, 3])
        XCTAssertEqual(frequency(counts, token: 0, draws: draws), 0.9, accuracy: 0.05)
        XCTAssertEqual(frequency(counts, token: 3, draws: draws), 0.1, accuracy: 0.05)
    }

    func testGenerateParametersCreatesExpectedSampler() {
        XCTAssertTrue(GenerateParameters(temperature: 0.7, topK: 40).sampler() is TopPSampler)
        XCTAssertTrue(GenerateParameters(temperature: 0.7, minP: 0.1).sampler() is TopPSampler)
        XCTAssertTrue(GenerateParameters(temperature: 0).sampler() is ArgMaxSampler)
    }

    func testPresencePenaltyContextPenalizesSeenTokens() {
        var processor = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 20)
        processor.prompt(MLXArray([1, 1, 3]))

        let logits =
            MLXArray([1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 3.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 3.5, accuracy: 1e-6)
    }

    func testFrequencyPenaltyContextPenalizesByCount() {
        var processor = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 20)
        processor.prompt(MLXArray([1, 1, 3]))

        let logits =
            MLXArray([1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float])[.newAxis, .ellipsis]
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 3.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 3.5, accuracy: 1e-6)
    }

    func testGenerateParametersCreatesExpectedPenaltyProcessor() {
        XCTAssertNotNil(GenerateParameters(repetitionPenalty: 1.1).processor())
        XCTAssertNotNil(GenerateParameters(presencePenalty: 0.5).processor())
        XCTAssertNotNil(GenerateParameters(frequencyPenalty: 0.5).processor())
        XCTAssertNotNil(
            GenerateParameters(
                repetitionPenalty: 1.1, presencePenalty: 0.5, frequencyPenalty: 0.5
            ).processor()
        )
    }

    func testPresencePenaltyContextPenalizesUniqueSeenTokens() {
        var processor = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 5)
        processor.prompt(MLXArray([0, 0, 0, 1, 1]))

        let logits = MLXArray.zeros([1, 4], type: Float.self)
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)

        XCTAssertEqual(values[0], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 0.0, accuracy: 1e-6)
    }

    func testFrequencyPenaltyContextPenalizesByTokenCount() {
        var processor = FrequencyPenaltyContext(frequencyPenalty: 0.5, frequencyContextSize: 5)
        processor.prompt(MLXArray([0, 0, 0, 1, 1]))

        let logits = MLXArray.zeros([1, 4], type: Float.self)
        let processed = processor.process(logits: logits)
        let values = processed[0].asArray(Float.self)

        XCTAssertEqual(values[0], -1.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], -1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 0.0, accuracy: 1e-6)
    }

    /// Regression for `[broadcast_shapes] Shapes (capacity) and (N + capacity - 1)`
    /// fixed in SharpAI/mlx-swift-lm#24. VLM prefill passes `input.text.tokens`
    /// as `[1, N]`; before the fix, `TokenRing.loadPrompt` read `dim(0)` as the
    /// batch axis and produced a malformed buffer. A 2-D prompt must load
    /// identically to the equivalent 1-D prompt.
    ///
    /// Uses `presenceContextSize: 20` (n < capacity) — the same proven-passing
    /// branch exercised by `testPresencePenaltyContextPenalizesUniqueSeenTokens`.
    /// Presence penalty deduplicates by token identity, so tokens 0 (×3) and 1 (×2)
    /// are each penalised exactly once: expected logit deltas are −0.5 for indices
    /// 0 and 1, 0.0 for indices 2 and 3.
    func testPresencePenaltyContext2DPromptMatches1D() {
        let tokens: [Int32] = [0, 0, 0, 1, 1]

        var processor2D = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 20)
        processor2D.prompt(MLXArray(tokens).reshaped([1, 5]))

        var processor1D = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 20)
        processor1D.prompt(MLXArray(tokens))

        let logits = MLXArray.zeros([1, 4], type: Float.self)
        let values2D = processor2D.process(logits: logits)[0].asArray(Float.self)
        let values1D = processor1D.process(logits: logits)[0].asArray(Float.self)

        // Verify the 2-D result directly (presence penalty, not frequency).
        XCTAssertEqual(values2D[0], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values2D[1], -0.5, accuracy: 1e-6)
        XCTAssertEqual(values2D[2],  0.0, accuracy: 1e-6)
        XCTAssertEqual(values2D[3],  0.0, accuracy: 1e-6)

        // 2-D and 1-D paths must produce identical output after the fix.
        XCTAssertEqual(values2D.count, values1D.count)
        for (a, b) in zip(values2D, values1D) {
            XCTAssertEqual(a, b, accuracy: 1e-6)
        }
    }

    /// Regression for SharpAI/mlx-swift-lm#24: a 2-D prompt longer than
    /// `presenceContextSize` previously built a buffer of shape
    /// `[N + capacity - 1]` because `n` was read as the batch axis (1) instead
    /// of the token count. The resulting ring was inconsistent with `positions`
    /// (shape `[capacity]`), so the first `didSample(...)` after prompt
    /// ingestion crashed at `MLX.where(mask [capacity], token,
    /// buffer [N + capacity - 1])`. Driving the full prompt → process →
    /// didSample path with a 2-D prompt longer than capacity proves the ring
    /// is well-formed after 2-D ingestion.
    func testPenaltyProcessorAppendAfterLong2DPromptDoesNotCrash() {
        var processor = PresencePenaltyContext(presencePenalty: 0.5, presenceContextSize: 20)
        let tokens = MLXArray((0..<700).map { Int32($0 % 128) }).reshaped([1, 700])
        processor.prompt(tokens)

        let logits = MLXArray.zeros([1, 128], type: Float.self)
        _ = processor.process(logits: logits)
        processor.didSample(token: MLXArray([Int32(42)]))
    }

    func testGenerateParametersPenaltyProcessorComposesPenaltiesInOrder() {
        var processor = GenerateParameters(
            repetitionPenalty: 1.5, repetitionContextSize: 5,
            presencePenalty: 0.5, presenceContextSize: 5,
            frequencyPenalty: 0.25, frequencyContextSize: 5
        ).processor()
        XCTAssertNotNil(processor)

        processor?.prompt(MLXArray([0, 0, 0, 1, 1]))
        let logits = MLXArray([1.0 as Float, 0.5 as Float, 0.0 as Float, -0.5 as Float])[
            .newAxis, .ellipsis
        ]
        let processed = processor?.process(logits: logits)
        guard let values = processed?[0].asArray(Float.self) else {
            XCTFail("Expected processed logits")
            return
        }
        XCTAssertEqual(values[0], -0.5833, accuracy: 1e-4)
        XCTAssertEqual(values[1], -0.6667, accuracy: 1e-4)
        XCTAssertEqual(values[2], 0.0, accuracy: 1e-4)
        XCTAssertEqual(values[3], -0.5, accuracy: 1e-4)
    }
}
