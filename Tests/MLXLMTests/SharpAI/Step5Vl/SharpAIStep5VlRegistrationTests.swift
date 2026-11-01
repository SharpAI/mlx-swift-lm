// SharpAI fork overlay tests for step5_vl registration. Not part of upstream.
// The creator builds the dense text decoder. CSA sparse attention and MoE layers still throw
// `unsupported`, so those configs are checked here as rejections.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIStep5VlRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "step5_vl",
            "vision_config": {
                "image_size": 728,
                "patch_size": 14,
                "width": 64,
                "layers": 2,
                "heads": 4
            },
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4
            }
        }
        """

    private let sparseJSON = """
        {
            "model_type": "step5_vl",
            "vision_config": {
                "image_size": 728,
                "patch_size": 14,
                "width": 64,
                "layers": 2,
                "heads": 4
            },
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "sparse_config": { "enabled": true }
            }
        }
        """

    private let moeJSON = """
        {
            "model_type": "step5_vl",
            "vision_config": {
                "image_size": 728,
                "patch_size": 14,
                "width": 64,
                "layers": 2,
                "heads": 4
            },
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "use_moe": true
            }
        }
        """

    private func makeModel(_ json: String) throws -> any LanguageModel {
        guard let creator = SharpAIStep5VlOverlay.creators["step5_vl"] else {
            throw XCTSkip("step5_vl creator missing")
        }
        return try creator(Data(json.utf8))
    }

    func testRegistersStep5VlKey() {
        XCTAssertNotNil(SharpAIStep5VlOverlay.creators["step5_vl"])
    }

    func testCreatorBuildsDenseTextModelForValidConfiguration() throws {
        let model = try makeModel(validJSON)
        XCTAssertTrue(model is SharpAIStep5VlModel)
    }

    func testCreatorRejectsSparseIndexerAsUnsupported() {
        XCTAssertThrowsError(try makeModel(sparseJSON)) { error in
            guard let overlayError = error as? SharpAIStep5VlError,
                case .unsupported = overlayError
            else {
                return XCTFail("expected SharpAIStep5VlError.unsupported, got \(error)")
            }
        }
    }

    func testCreatorRejectsMoeAsUnsupported() {
        XCTAssertThrowsError(try makeModel(moeJSON)) { error in
            guard let overlayError = error as? SharpAIStep5VlError,
                case .unsupported = overlayError
            else {
                return XCTFail("expected SharpAIStep5VlError.unsupported, got \(error)")
            }
        }
    }

    func testCreatorRejectsInvalidConfigurationBeforeConstruct() {
        guard let creator = SharpAIStep5VlOverlay.creators["step5_vl"] else {
            return XCTFail("step5_vl creator missing")
        }
        let wrongType = validJSON.replacingOccurrences(
            of: "\"step5_vl\"", with: "\"step5\"")

        XCTAssertThrowsError(try creator(Data(wrongType.utf8))) { error in
            guard let overlayError = error as? SharpAIStep5VlError,
                case .invalidConfiguration = overlayError
            else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testForwardPassProducesFiniteLogitsForEveryPosition() throws {
        let model = try makeModel(validJSON)
        let tokens = MLXArray([Int32(1), 5, 9]).reshaped(1, 3)

        let logits = model.callAsFunction(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testForwardPassIsCausal() throws {
        let model = try makeModel(validJSON)
        let first = model.callAsFunction(
            MLXArray([Int32(1), 5, 9]).reshaped(1, 3), cache: nil)
        let second = model.callAsFunction(
            MLXArray([Int32(1), 5, 20]).reshaped(1, 3), cache: nil)

        // Changing the last token must not change the logits at the first two positions.
        let earlier = abs(first[0..., ..<2] - second[0..., ..<2]).max().item(Float.self)
        XCTAssertEqual(earlier, 0, accuracy: 1e-4)
    }
}
