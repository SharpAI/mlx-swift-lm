// SharpAI fork overlay tests for the minimax_m3_vl creator. Not part of upstream.
// The creator still refuses to build a model (the vision tower and projector are not written),
// so the registration assertions are unchanged. The forward-pass test runs the text backbone
// (SharpAIMinimaxM3VlForCausalLM) on a tiny random-weight config with the same layer mix as M3:
// one full-attention dense layer, then sparse-indexer MoE layers.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

enum SharpAIMinimaxM3VlTinyText {
    /// Tiny shapes chosen so the indexer sees 3 key blocks with padding (seq 10, block 4), and top-2 selection.
    static let configuration = SharpAIMinimaxM3VlTextConfiguration(
        hiddenSize: 32,
        numHiddenLayers: 4,
        numAttentionHeads: 4,
        numKeyValueHeads: 2,
        headDim: 8,
        partialRotaryFactor: 0.5,
        vocabularySize: 64,
        rmsNormEps: 1e-6,
        ropeTheta: 5_000_000,
        numLocalExperts: 4,
        numExpertsPerTok: 2,
        moeIntermediateSize: 8,
        denseIntermediateSize: 16,
        sharedIntermediateSize: 8,
        swigluAlpha: 1.702,
        swigluLimit: 7.0,
        routedScalingFactor: 2.0,
        indexNumHeads: 2,
        indexHeadDim: 8,
        indexBlockSize: 4,
        indexTopkBlocks: 2,
        indexLocalBlocks: 1,
        layerTypes: ["full_attention", "minimax_m3_sparse", "minimax_m3_sparse", "minimax_m3_sparse"],
        mlpLayerTypes: ["dense", "sparse", "sparse", "sparse"]
    )
}

final class SharpAIMinimaxM3VlRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMinimaxM3VlOverlay.creators["minimax_m3_vl"])
    }

    func testWrongArchitectureFailsAtValidation() throws {
        guard let creator = SharpAIMinimaxM3VlOverlay.creators["minimax_m3_vl"] else {
            XCTFail("minimax_m3_vl creator missing")
            return
        }
        let json = SharpAIMinimaxM3VlFixtures.tinyJSON.replacingOccurrences(
            of: "MiniMaxM3SparseForConditionalGeneration", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIMinimaxM3VlError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidConfigStopsAtConstructionAsUnsupported() throws {
        guard let creator = SharpAIMinimaxM3VlOverlay.creators["minimax_m3_vl"] else {
            XCTFail("minimax_m3_vl creator missing")
            return
        }

        XCTAssertThrowsError(
            try creator(Data(SharpAIMinimaxM3VlFixtures.tinyJSON.utf8))
        ) { error in
            guard let typed = error as? SharpAIMinimaxM3VlError,
                case .unsupported = typed
            else {
                XCTFail("expected unsupported, got \(error)")
                return
            }
        }
    }

    func testTinyForwardPassWithRandomWeights() throws {
        let config = SharpAIMinimaxM3VlTinyText.configuration
        let model = SharpAIMinimaxM3VlForCausalLM(config)

        let ids: [Int32] = (0 ..< 10).map { Int32(($0 * 7) % 64) }
        let logits = model(MLXArray(ids, [1, 10]))

        XCTAssertEqual(logits.shape, [1, 10, 64])
        let values = logits.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "logits must be finite")
    }

    /// Causality through the block-sparse path: changing the last token must not move earlier logits.
    func testForwardIsCausalWithSparseIndexer() throws {
        let config = SharpAIMinimaxM3VlTinyText.configuration
        let model = SharpAIMinimaxM3VlForCausalLM(config)

        let ids: [Int32] = (0 ..< 10).map { Int32(($0 * 7) % 64) }
        var altered = ids
        altered[9] = (altered[9] + 1) % 64

        let before = model(MLXArray(ids, [1, 10]))[0, ..<9].asArray(Float.self)
        let after = model(MLXArray(altered, [1, 10]))[0, ..<9].asArray(Float.self)

        XCTAssertEqual(before.count, after.count)
        let maxDiff = zip(before, after).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 1e-4, "future token leaked into earlier positions")
    }
}
