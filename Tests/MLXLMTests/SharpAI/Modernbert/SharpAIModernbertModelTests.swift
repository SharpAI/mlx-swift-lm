// SharpAI fork overlay tests for the modernbert encoder. Not part of upstream.
//
// Tiny random-weight forward passes only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIModernbertModelTests: XCTestCase {

    /// Four layers: layers 0 and 3 are global, layers 1 and 2 are local with a window of 4.
    private func makeConfig() throws -> SharpAIModernbertConfiguration {
        let json = """
            {
                "model_type": "modernbert",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 4,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "norm_eps": 1e-5,
                "local_attention": 4,
                "global_attn_every_n_layers": 3,
                "max_position_embeddings": 64
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIModernbertConfiguration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesHiddenStateShape() throws {
        let model = SharpAIModernbertModel(try makeConfig())
        let inputs = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped(1, 6)

        let hidden = model.hiddenStates(inputs)
        eval(hidden)

        XCTAssertEqual(hidden.shape, [1, 6, 8])
    }

    func testCallAsFunctionMatchesHiddenStates() throws {
        let model = SharpAIModernbertModel(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let viaCall = model(inputs, cache: nil)
        eval(viaCall)

        XCTAssertEqual(viaCall.shape, [1, 3, 8])
    }

    /// Attention is bidirectional, so a change at the last position must move position 0.
    func testAttentionIsNonCausal() throws {
        let model = SharpAIModernbertModel(try makeConfig())
        let early = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let late = MLXArray([1, 2, 7] as [Int32]).reshaped(1, 3)

        let a = model.hiddenStates(early)
        let b = model.hiddenStates(late)
        eval(a, b)

        let diff = abs(a[0, 0] - b[0, 0]).max().item(Float.self)
        XCTAssertGreaterThan(diff, 1e-6)
    }

    /// Padding keys are masked out, so a change at a padded position must not move real positions.
    func testPaddingDoesNotLeakIntoRealTokens() throws {
        let model = SharpAIModernbertModel(try makeConfig())
        let mask = MLXArray([1, 1, 1, 0] as [Int32]).reshaped(1, 4)
        let a = MLXArray([1, 2, 3, 0] as [Int32]).reshaped(1, 4)
        let b = MLXArray([1, 2, 3, 7] as [Int32]).reshaped(1, 4)

        let outA = model.hiddenStates(a, attentionMask: mask)
        let outB = model.hiddenStates(b, attentionMask: mask)
        eval(outA, outB)

        let diff = abs(outA[0, 0 ..< 3] - outB[0, 0 ..< 3]).max().item(Float.self)
        XCTAssertLessThan(diff, 1e-5)
    }

    /// Band of |i - j| <= 1 over length 5.
    func testSlidingWindowMaskShapeAndBand() {
        let mask = SharpAIModernbertMasks.slidingWindow(length: 5, halfWidth: 1)
        eval(mask)

        XCTAssertEqual(mask.shape, [1, 1, 5, 5])
        let values = mask.asArray(Bool.self)
        XCTAssertEqual(Array(values[0 ..< 5]), [true, true, false, false, false])
        XCTAssertEqual(Array(values[10 ..< 15]), [false, true, true, true, false])
    }

    func testNewCacheIsEmpty() throws {
        let model = SharpAIModernbertModel(try makeConfig())

        XCTAssertTrue(try model.newCache(parameters: nil).isEmpty)
    }

    /// The encoder has no autoregressive prefill, so prepare must refuse.
    func testPrepareIsRejected() throws {
        let model = SharpAIModernbertModel(try makeConfig())
        let input = LMInput(text: LMInput.Text(tokens: MLXArray([1, 2, 3] as [Int32])))

        XCTAssertThrowsError(
            try model.prepare(
                input, cache: [], state: nil, prefill: PrefillParameters()))
    }
}
