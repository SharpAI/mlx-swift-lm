// SharpAI fork overlay tests for the openai_privacy_filter token classifier.
// Tiny-shape forward passes with random weights only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIOpenaiPrivacyFilterModelTests: XCTestCase {

    private let tinyJSON = """
        {
            "model_type": "openai_privacy_filter",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 8,
            "head_dim": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "sliding_window": 2,
            "id2label": {"0": "O", "1": "B-secret", "2": "I-secret", "3": "E-secret", "4": "S-secret"}
        }
        """

    private func tinyConfiguration() throws -> SharpAIOpenaiPrivacyFilterConfiguration {
        try JSONDecoder().decode(
            SharpAIOpenaiPrivacyFilterConfiguration.self, from: Data(tinyJSON.utf8))
    }

    func testBandMaskIsInclusiveAndBidirectional() {
        let mask = SharpAIOpenaiPrivacyFilterMasks.slidingWindow(length: 4, window: 1)
            .asArray(Bool.self)

        XCTAssertEqual(
            mask,
            [
                true, true, false, false,
                true, true, true, false,
                false, true, true, true,
                false, false, true, true,
            ])
    }

    func testYarnFrequenciesHaveHalfHeadDimAndUnitFirstWavelength() throws {
        let config = try tinyConfiguration()

        let rope = SharpAIOpenaiPrivacyFilterRope.yarn(config)
        let freqs = rope.freqs.asArray(Float.self)

        XCTAssertEqual(freqs.count, config.headDimensions / 2)
        // The first pair has no rotation scaling, so its wavelength is 1.
        XCTAssertEqual(freqs[0], 1.0, accuracy: 1e-4)
    }

    func testTokenLogitsHaveOneRowPerTokenPerLabel() throws {
        let config = try tinyConfiguration()
        let model = SharpAIOpenaiPrivacyFilterModel(config)

        let inputs = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped(1, 6)
        let logits = model.tokenLogits(inputs)

        XCTAssertEqual(logits.shape, [1, 6, 5])
    }

    func testPaddingMaskKeepsShapeAndCallReturnsTokenLogits() throws {
        let config = try tinyConfiguration()
        let model = SharpAIOpenaiPrivacyFilterModel(config)

        let inputs = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped(1, 6)
        let attentionMask = MLXArray([1, 1, 1, 1, 0, 0] as [Int32]).reshaped(1, 6)

        let padded = model.tokenLogits(inputs, attentionMask: attentionMask)
        let viaCall = model.callAsFunction(inputs, cache: nil)

        XCTAssertEqual(padded.shape, [1, 6, 5])
        XCTAssertEqual(viaCall.shape, [1, 6, 5])
    }

    func testEncoderHasNoCache() throws {
        let model = SharpAIOpenaiPrivacyFilterModel(try tinyConfiguration())

        XCTAssertEqual(try model.newCache(parameters: nil).count, 0)
    }
}
