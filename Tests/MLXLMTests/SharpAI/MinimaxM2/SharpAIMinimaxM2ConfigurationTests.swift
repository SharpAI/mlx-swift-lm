import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM2ConfigurationTests: XCTestCase {

    func testDecodesFixtureFields() throws {
        let config = try SharpAIMinimaxM2TestFixtures.decode()

        XCTAssertEqual(config.modelType, "minimax_m2")
        XCTAssertEqual(config.architectures, ["Eagle3MiniMaxM2ForCausalLM"])
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.intermediateSize, 32)
        XCTAssertEqual(config.numHiddenLayers, 1)
        XCTAssertEqual(config.numAttentionHeads, 4)
        XCTAssertEqual(config.numKeyValueHeads, 2)
        XCTAssertEqual(config.vocabSize, 64)
        XCTAssertEqual(config.draftVocabSize, 32)
        XCTAssertEqual(config.maxPositionEmbeddings, 128)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadDimDerivesFromHiddenSizeWhenAbsent() throws {
        let config = try SharpAIMinimaxM2TestFixtures.decode()

        XCTAssertNil(config.headDim)
        XCTAssertEqual(config.resolvedHeadDim, 4)
    }

    func testExplicitHeadDimIsUsed() throws {
        let json = SharpAIMinimaxM2TestFixtures.tinyJSON
            .replacingOccurrences(of: "\"hidden_size\": 16,", with: "\"hidden_size\": 16,\n  \"head_dim\": 8,")
        let config = try SharpAIMinimaxM2TestFixtures.decode(json)

        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.resolvedHeadDim, 8)
    }

    func testDefaultsApplyWhenOptionalKeysAreMissing() throws {
        let json = SharpAIMinimaxM2TestFixtures.tinyJSON
            .replacingOccurrences(of: "\"model_type\": \"minimax_m2\",", with: "")
            .replacingOccurrences(of: "\"architectures\": [\"Eagle3MiniMaxM2ForCausalLM\"],", with: "")
            .replacingOccurrences(of: "\"num_key_value_heads\": 2,", with: "")
            .replacingOccurrences(of: "\"num_hidden_layers\": 1,", with: "")
            .replacingOccurrences(of: "\"draft_vocab_size\": 32,", with: "")
            .replacingOccurrences(of: "\"rms_norm_eps\": 1e-6,", with: "")
            .replacingOccurrences(of: "\"rope_theta\": 10000,", with: "")
        let config = try SharpAIMinimaxM2TestFixtures.decode(json)

        XCTAssertEqual(config.modelType, "minimax_m2")
        XCTAssertEqual(config.architectures, [])
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.numHiddenLayers, 1)
        XCTAssertNil(config.draftVocabSize)
        XCTAssertEqual(config.rmsNormEps, 1e-6)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingHiddenSizeFailsDecoding() {
        let json = SharpAIMinimaxM2TestFixtures.tinyJSON
            .replacingOccurrences(of: "\"hidden_size\": 16,", with: "")

        XCTAssertThrowsError(try SharpAIMinimaxM2TestFixtures.decode(json))
    }

    func testWrongModelTypeFailsValidation() throws {
        let json = SharpAIMinimaxM2TestFixtures.tinyJSON
            .replacingOccurrences(of: "\"model_type\": \"minimax_m2\"", with: "\"model_type\": \"minimax\"")
        let config = try SharpAIMinimaxM2TestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case .invalidConfiguration = error as? SharpAIMinimaxM2Error else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testKeyValueHeadsMustDivideAttentionHeads() throws {
        let json = SharpAIMinimaxM2TestFixtures.tinyJSON
            .replacingOccurrences(of: "\"num_key_value_heads\": 2,", with: "\"num_key_value_heads\": 3,")
        let config = try SharpAIMinimaxM2TestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testDraftVocabLargerThanVocabFailsValidation() throws {
        let json = SharpAIMinimaxM2TestFixtures.tinyJSON
            .replacingOccurrences(of: "\"draft_vocab_size\": 32,", with: "\"draft_vocab_size\": 128,")
        let config = try SharpAIMinimaxM2TestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
