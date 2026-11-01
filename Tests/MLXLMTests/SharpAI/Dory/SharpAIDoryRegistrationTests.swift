// SharpAI fork overlay tests for the Dory creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDoryRegistrationTests: XCTestCase {

    func testDoryKeyIsRegistered() {
        XCTAssertNotNil(SharpAIDoryOverlay.creators["dory"])
    }

    /// The creator now builds the model. Tiny shape only: the 4B fixture would allocate ~8 GB.
    func testCreatorBuildsModelFromTinyConfig() throws {
        let creator = try XCTUnwrap(SharpAIDoryOverlay.creators["dory"])

        let model = try creator(Data(SharpAIDoryTestFixture.tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIDoryForCausalLM)
    }

    func testCreatorRejectsInvalidConfigBeforeCacheStep() throws {
        let creator = try XCTUnwrap(SharpAIDoryOverlay.creators["dory"])
        let json = SharpAIDoryTestFixture.tinyJSON.replacingOccurrences(
            of: "\"n_recurrent_loops\": 2", with: "\"n_recurrent_loops\": 0")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAIDoryError.invalidConfiguration = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    /// Runs the forward pass on random weights. The shape and cache-consistency checks are real
    /// assertions. The comparison against reference logits is an expected failure because no
    /// reference fixture exists.
    func testForwardPassMatchesReference() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIDoryConfiguration.self, from: Data(SharpAIDoryTestFixture.tinyJSON.utf8))
        let model = try SharpAIDoryForCausalLM(configuration)

        let tokens = MLXArray([Int32(3), 7, 11, 2, 5]).reshaped(1, 5)

        // One cache per attention slot: sliding input (1) plus full recurrent per loop (2).
        XCTAssertEqual(try model.newCache(parameters: nil).count, 3)

        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertFalse(isNaN(logits).any().item(Bool.self))

        // Prefill four tokens into a cache, then decode the fifth. The last position must match
        // the uncached pass. This checks per-loop cache offsets and the sliding window path.
        let cache = try model.newCache(parameters: nil)
        _ = model(tokens[0..., 0 ..< 4], cache: cache)
        let stepLogits = model(tokens[0..., 4 ..< 5], cache: cache)
        XCTAssertTrue(
            allClose(
                stepLogits[0..., 0, 0...], logits[0..., 4, 0...], rtol: 1e-3, atol: 1e-4
            ).item(Bool.self))

        XCTExpectFailure(
            "KNOWN GAP: no reference logits exist for tinyJSON with these random weights. The reference is "
                + "vLLM dory.py at b5007e42 (fetched, its formulas are mirrored in SharpAIDoryModel.swift), but "
                + "running it needs torch and vLLM, neither installed here, and the Swift weights come from "
                + "MLXRandom, so no logits can be computed from the reference for this config. The only public "
                + "Dory weights, nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 (rev b366b87f), are 4B and not downloaded, "
                + "and tiny-random/dory-1.0 has a different shape with no published logits. Closing this needs "
                + "deterministic weights shared with a reference run, then a fill-in of SharpAIDoryTestFixture.referenceLogits."
        ) {
            XCTAssertNotNil(
                SharpAIDoryTestFixture.referenceLogits,
                "reference logits must be present to compare the forward pass")
        }
    }
}
