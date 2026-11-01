import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIKanana15VConfigurationTests: XCTestCase {

    func testDecodesNestedConfig() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()

        XCTAssertEqual(config.modelType, "kanana-1.5-v")
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.visionConfiguration.depth, 1)
        XCTAssertEqual(config.visionConfiguration.embedDimensions, 8)
        XCTAssertEqual(config.visionConfiguration.attentionHeads, 2)
        XCTAssertEqual(config.visionConfiguration.temporalPatchSize, 2)
        XCTAssertEqual(config.visionConfiguration.spatialMergeSize, 2)
        XCTAssertEqual(config.visionConfiguration.inChannels, 3)
        XCTAssertEqual(config.visionConfiguration.hiddenAct, "quick_gelu")
        XCTAssertEqual(config.projectorConfiguration.projectorType, "dynamic-c-abs")
        XCTAssertEqual(config.projectorConfiguration.mlpDepth, 2)
        XCTAssertEqual(config.projectorConfiguration.posEmbSize, 4)
    }

    func testTextConfigBuildsLlamaTowerWithConfiguredVocabulary() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()
        let language = LlamaModel(config.textConfiguration)

        XCTAssertEqual(language.vocabularySize, 32)
        XCTAssertEqual(language.kvHeads, [1])
        XCTAssertNotNil(language.lmHead)
    }

    func testTiedEmbeddingsFlagIsReadFromTextConfig() throws {
        let json = SharpAIKanana15VTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"tie_word_embeddings\": false", with: "\"tie_word_embeddings\": true")
        let config = try SharpAIKanana15VTestFixtures.decode(json)

        XCTAssertTrue(config.tieWordEmbeddings)
    }

    func testValidationAcceptsTinyConfig() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let json = SharpAIKanana15VTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"model_type\": \"kanana-1.5-v\",", with: "\"model_type\": \"other\",")
        let config = try SharpAIKanana15VTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsNonQuickGeluVisionActivation() throws {
        let json = SharpAIKanana15VTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"hidden_act\": \"quick_gelu\"", with: "\"hidden_act\": \"gelu\"")
        let config = try SharpAIKanana15VTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIKanana15VError,
                case .unsupportedActivation(let name) = typed
            else { return XCTFail("unexpected error: \(error)") }
            XCTAssertEqual(name, "gelu")
        }
    }
}
