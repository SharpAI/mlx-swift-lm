// SharpAI fork overlay tests for the llama_bidirec model. Not part of upstream.
//
// Tiny random-weight forward passes only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILlamaBidirecModelTests: XCTestCase {

    private func makeConfig() throws -> SharpAILlamaBidirecConfiguration {
        let json = """
            {
                "model_type": "llama_bidirec",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAILlamaBidirecConfiguration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesHiddenStateShape() throws {
        let model = SharpAILlamaBidirecModel(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let hidden = model.hiddenStates(inputs)
        eval(hidden)

        XCTAssertEqual(hidden.shape, [1, 3, 8])
    }

    func testCallAsFunctionMatchesHiddenStates() throws {
        let model = SharpAILlamaBidirecModel(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let viaCall = model(inputs, cache: nil)
        eval(viaCall)

        XCTAssertEqual(viaCall.shape, [1, 3, 8])
    }

    /// Attention is non-causal, so a change at the last position must move the output at
    /// position 0. A causal model would give exactly zero difference there.
    func testAttentionIsNonCausal() throws {
        let model = SharpAILlamaBidirecModel(try makeConfig())
        let early = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let late = MLXArray([1, 2, 7] as [Int32]).reshaped(1, 3)

        let a = model.hiddenStates(early)
        let b = model.hiddenStates(late)
        eval(a, b)

        let diff = abs(a[0, 0] - b[0, 0]).max().item(Float.self)
        XCTAssertGreaterThan(diff, 1e-6)
    }

    func testNewCacheIsEmpty() throws {
        let model = SharpAILlamaBidirecModel(try makeConfig())

        XCTAssertTrue(try model.newCache(parameters: nil).isEmpty)
    }

    /// Chunked prefill would hide later tokens from earlier positions, so prepare must refuse.
    func testPrepareIsRejected() throws {
        let model = SharpAILlamaBidirecModel(try makeConfig())
        let input = LMInput(text: LMInput.Text(tokens: MLXArray([1, 2, 3] as [Int32])))

        XCTAssertThrowsError(
            try model.prepare(
                input, cache: [], state: nil, prefill: PrefillParameters()))
    }
}
