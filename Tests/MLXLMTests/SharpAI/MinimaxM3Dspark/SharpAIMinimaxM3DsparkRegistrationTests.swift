// SharpAI fork overlay tests for the minimax_m3_dspark creator and draft backbone. Not part of upstream.
// The creator still refuses to build a full model (no markov head, no confidence head, no target).
// The backbone forward pass and the DFlash mask are tested directly on tiny random weights.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3DsparkRegistrationTests: XCTestCase {

    private let validJSON = """
        {
          "architectures": ["DSparkMiniMaxDraftModel"],
          "model_type": "minimax_m3_dspark"
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMinimaxM3DsparkOverlay.creators["minimax_m3_dspark"])
    }

    func testValidConfigStopsAsUnsupported() throws {
        let creator = try XCTUnwrap(SharpAIMinimaxM3DsparkOverlay.creators["minimax_m3_dspark"])

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard let typed = error as? SharpAIMinimaxM3DsparkError,
                case .unsupported = typed
            else {
                XCTFail("expected unsupported, got \(error)")
                return
            }
        }
    }

    func testWrongArchitectureFailsAtValidation() throws {
        let creator = try XCTUnwrap(SharpAIMinimaxM3DsparkOverlay.creators["minimax_m3_dspark"])
        let json = validJSON.replacingOccurrences(
            of: "DSparkMiniMaxDraftModel", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIMinimaxM3DsparkError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testWrongModelTypeFailsAtValidation() throws {
        let creator = try XCTUnwrap(SharpAIMinimaxM3DsparkOverlay.creators["minimax_m3_dspark"])
        let json = validJSON.replacingOccurrences(
            of: "\"model_type\": \"minimax_m3_dspark\"", with: "\"model_type\": \"llama\"")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIMinimaxM3DsparkError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    /// Tiny shapes: 2 layers, hidden 16, 4 query heads over 2 KV heads, head_dim 4 with half the
    /// channels rotary, a 2-layer target context of width 2 * 16, and 2 blocks of 4 positions.
    private func tinyBackboneConfig() -> SharpAIMinimaxM3DsparkBackboneConfig {
        SharpAIMinimaxM3DsparkBackboneConfig(
            hiddenSize: 16, intermediateSize: 24, numHiddenLayers: 2,
            numAttentionHeads: 4, numKeyValueHeads: 2, headDim: 4, vocabSize: 32,
            targetLayerCount: 2, blockSize: 4, maskTokenId: 31,
            partialRotaryFactor: 0.5)
    }

    func testTinyForwardPassWithRandomWeights() throws {
        MLXRandom.seed(0)
        let backbone = SharpAIMinimaxM3DsparkBackbone(tinyBackboneConfig())
        let ctxLen = 12
        let targetHidden = MLXRandom.normal([1, ctxLen, 2 * 16])
        let inputIds = [(0 ..< ctxLen).map { ($0 * 5 + 1) % 31 }]

        // The second block is dropped (keep false, anchor 0), so it only attends to its own noise.
        let logits = backbone(
            inputIds: inputIds, targetHidden: targetHidden,
            anchors: [[1, 0]], keep: [[true, false]])
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 2, 4, 32])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testDFlashMaskSeesPrefixAndOwnBlockOnly() {
        // Context 6, block size 2. Block 0 has anchor 3 and is kept. Block 1 is dropped.
        let mask = sharpAIMinimaxM3DsparkDFlashMask(
            anchors: [[3, 0]], keep: [[true, false]], ctxLen: 6, blockSize: 2)
        XCTAssertEqual(mask.shape, [1, 1, 4, 10])
        let values = mask.asArray(Float.self)
        // Row q, column kv. Row width is KV = 10.
        func visible(_ q: Int, _ kv: Int) -> Bool { values[q * 10 + kv] == 0 }

        // Block 0 queries see context below the anchor (0..<3) and their own noise (6, 7).
        XCTAssertTrue(visible(0, 0))
        XCTAssertTrue(visible(0, 2))
        XCTAssertFalse(visible(0, 3))
        XCTAssertTrue(visible(0, 6))
        XCTAssertTrue(visible(0, 7))
        XCTAssertFalse(visible(0, 8))
        XCTAssertTrue(visible(1, 6))

        // The dropped block sees no context, only its own noise (8, 9).
        XCTAssertFalse(visible(2, 0))
        XCTAssertFalse(visible(2, 6))
        XCTAssertTrue(visible(2, 8))
        XCTAssertTrue(visible(2, 9))
    }
}
