// SharpAI fork overlay tests for the iquestloopcoder creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIIquestloopcoderRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "iquestloopcoder",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 512
        }
        """

    /// Two layers, two loops, window of 4 tokens: small enough to check the cache layout.
    private let dualCacheJSON = """
        {
            "model_type": "iquestloopcoder",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 512,
            "loop_num": 2,
            "loop_window_size": 4
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIIquestloopcoderOverlay.creators["iquestloopcoder"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIIquestloopcoderOverlay.creators["iquestloopcoder"] else {
            XCTFail("iquestloopcoder creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))
        XCTAssertTrue(model is SharpAIIquestloopcoderModel)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIIquestloopcoderOverlay.creators["iquestloopcoder"] else {
            XCTFail("iquestloopcoder creator missing")
            return
        }
        let json = """
            { "hidden_size": 64 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    /// Was `testLoopKnownIssueSkipped`. The loop combines the local and global attentions with
    /// the gate as a convex combination: out = local * (1 - g) + global * g.
    func testLoopMixIsConvexCombinationOfLocalAndGlobal() {
        let local = MLXArray.ones([1, 2, 3, 4]) * 2
        let global = MLXArray.zeros([1, 2, 3, 4])

        let halfway = SharpAIIquestloopcoderLoopGate.mix(
            local: local, global: global, gate: MLXArray(Float(0.25)))
        XCTAssertEqual(halfway.shape, [1, 2, 3, 4])
        for value in halfway.asArray(Float.self) {
            XCTAssertEqual(Double(value), 1.5, accuracy: 1e-6)
        }

        let allGlobal = SharpAIIquestloopcoderLoopGate.mix(
            local: local, global: global, gate: MLXArray(Float(1)))
        XCTAssertEqual(allGlobal.asArray(Float.self).max() ?? 1, 0, accuracy: 1e-6)

        let allLocal = SharpAIIquestloopcoderLoopGate.mix(
            local: local, global: global, gate: MLXArray(Float(0)))
        XCTAssertEqual(allLocal.asArray(Float.self).min() ?? 0, 2, accuracy: 1e-6)
    }

    /// Was `testGateKnownIssueSkipped`. gate[b, h, s] = sigmoid(sum_d q[b,h,s,d] * W[h,d] + b[h]).
    func testGateMatchesSigmoidOfRotatedQueryDot() {
        let heads = 2
        let headDim = 4
        let len = 3
        let weightValues: [Float] = [0.5, -1.0, 0.25, 2.0, -0.5, 0.75, 1.0, -0.25]
        let biasValues: [Float] = [0.1, -0.3]
        let queryValues: [Float] = (0 ..< heads * len * headDim).map { Float($0 % 7) * 0.2 - 0.6 }

        let gate = SharpAIIquestloopcoderLoopGate(
            weight: MLXArray(weightValues, [heads, headDim]),
            bias: MLXArray(biasValues, [heads]))
        let queries = MLXArray(queryValues, [1, heads, len, headDim])

        let output = gate(queries)
        XCTAssertEqual(output.shape, [1, heads, len, 1])
        let actual = output.asArray(Float.self)

        for h in 0 ..< heads {
            for s in 0 ..< len {
                var dot: Float = biasValues[h]
                for d in 0 ..< headDim {
                    dot += queryValues[(h * len + s) * headDim + d] * weightValues[h * headDim + d]
                }
                let expected = 1 / (1 + exp(-dot))
                XCTAssertEqual(
                    Double(actual[h * len + s]), Double(expected), accuracy: 1e-5,
                    "gate[\(h), \(s)]")
            }
        }
    }

    /// Was `testDualCacheKnownIssueSkipped`. Shared pass-0 cache keeps the full history; each
    /// window cache is capped at loop_window_size.
    func testDualCacheSharedFullLocalWindowed() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIIquestloopcoderConfiguration.self, from: Data(dualCacheJSON.utf8))
        let model = SharpAIIquestloopcoderModel(config)

        let caches = try model.newCache(parameters: nil)
        // 2 layers of shared caches, plus 2 layers of window caches for loop 1.
        XCTAssertEqual(caches.count, 4)
        XCTAssertTrue(caches[0] is KVCacheSimple)
        XCTAssertTrue(caches[1] is KVCacheSimple)
        XCTAssertTrue(caches[2] is RotatingKVCache)
        XCTAssertTrue(caches[3] is RotatingKVCache)
        XCTAssertEqual(caches[2].maxSize, 4)

        let prompt = MLXArray((0 ..< 10).map { Int32($0) }, [1, 10])
        let logits = model.callAsFunction(prompt, cache: caches)
        XCTAssertEqual(logits.shape, [1, 10, 512])
        XCTAssertEqual(caches[0].offset, 10)
        XCTAssertEqual(caches[0].state[0].dim(2), 10)
        XCTAssertEqual(caches[2].offset, 10)

        let next = MLXArray([Int32(3)], [1, 1])
        _ = model.callAsFunction(next, cache: caches)
        XCTAssertEqual(caches[0].state[0].dim(2), 11)
        XCTAssertEqual(caches[2].offset, 11)
        XCTAssertLessThanOrEqual(caches[2].state[0].dim(2), 4)
    }

    /// Was `testTinyForwardPassSkipped`. Random-init tiny model: logits have the right shape and
    /// are finite, with and without an explicit cache.
    func testTinyForwardPassProducesFiniteLogits() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIIquestloopcoderConfiguration.self, from: Data(validJSON.utf8))
        let model = SharpAIIquestloopcoderModel(config)

        let tokens = MLXArray((0 ..< 10).map { Int32($0) }, [1, 10])
        let logits = model.callAsFunction(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 10, 512])
        XCTAssertFalse(isNaN(logits).any().item(Bool.self))

        let cached = model.callAsFunction(tokens, cache: try model.newCache(parameters: nil))
        XCTAssertEqual(cached.shape, [1, 10, 512])
        XCTAssertFalse(isNaN(cached).any().item(Bool.self))
    }
}
