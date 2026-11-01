import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIOpencuaConfigurationTests: XCTestCase {

    func testDecodesNestedTextAndVisionConfig() throws {
        let config = try SharpAIOpencuaTestFixtures.decode()

        XCTAssertEqual(config.modelType, "opencua")
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertEqual(config.visionConfiguration.depth, 1)
        XCTAssertEqual(config.mediaPlaceholderTokenId, 6)
    }

    func testMediaPlaceholderDefaultsWhenAbsent() throws {
        let json = SharpAIOpencuaTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"media_placeholder_token_id\": 6,", with: "")
        let config = try SharpAIOpencuaTestFixtures.decode(json)

        XCTAssertEqual(config.mediaPlaceholderTokenId, 151664)
    }

    func testFlatConfigurationMovesTextFieldsToTopLevel() throws {
        let data = Data(SharpAIOpencuaTestFixtures.tinyJSON.utf8)
        let config = try SharpAIOpencuaTestFixtures.decode()
        let flat = try config.qwen25VLConfiguration(from: data)

        XCTAssertEqual(flat.textConfiguration.hiddenSize, 64)
        XCTAssertEqual(flat.textConfiguration.modelType, "qwen2")
        XCTAssertEqual(flat.baseConfiguration.numHiddenLayers, 2)
        XCTAssertEqual(flat.baseConfiguration.vocabularySize, 128)
        XCTAssertEqual(flat.baseConfiguration.imageTokenId, 6)
        XCTAssertEqual(flat.baseConfiguration.videoTokenId, -1)
        XCTAssertEqual(flat.visionConfiguration.depth, 1)
    }

    func testValidatesModelTypeAndMRoPESection() throws {
        XCTAssertNoThrow(try SharpAIOpencuaTestFixtures.decode().validateModelConfiguration())

        let wrongType = SharpAIOpencuaTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"model_type\": \"opencua\"", with: "\"model_type\": \"qwen2_5_vl\"")
        XCTAssertThrowsError(
            try SharpAIOpencuaTestFixtures.decode(wrongType).validateModelConfiguration())

        let shortSection = SharpAIOpencuaTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"mrope_section\": [2, 1, 1]", with: "\"mrope_section\": [2, 1]")
        XCTAssertThrowsError(
            try SharpAIOpencuaTestFixtures.decode(shortSection).validateModelConfiguration())
    }

    /// The real OpenCUA-7B text_config has no rope_scaling. It must validate, get a
    /// section for Qwen25VL's attention, and build the model without the mrope_section fatalError.
    func testConfigWithoutRopeScalingValidatesAndBuilds() throws {
        let json = try SharpAIOpencuaTestFixtures.withoutRopeScaling()
        let config = try SharpAIOpencuaTestFixtures.decode(json)

        XCTAssertNoThrow(try config.validateModelConfiguration())

        // head_dim 16 gives half 8, split [2, 3, 3]. Qwen2.5-VL's [16, 24, 24] is the head_dim 128 case.
        let flat = try config.qwen25VLConfiguration(from: Data(json.utf8))
        XCTAssertEqual(flat.textConfiguration.ropeScaling?["mrope_section"]?.asInts(), [2, 3, 3])
        if case .string(let type)? = flat.textConfiguration.ropeScaling?["type"] {
            XCTAssertEqual(type, "mrope")
        } else {
            XCTFail("injected rope_scaling has no string type")
        }

        _ = SharpAIOpencuaModel(flat)
    }

    /// A JSON null rope_scaling is the same as an absent one.
    func testNullRopeScalingGetsInjectedSection() throws {
        var root =
            try JSONSerialization.jsonObject(
                with: Data(try SharpAIOpencuaTestFixtures.withoutRopeScaling().utf8))
            as! [String: Any]
        var text = root["text_config"] as! [String: Any]
        text["rope_scaling"] = NSNull()
        root["text_config"] = text
        let data = try JSONSerialization.data(withJSONObject: root)
        let config = try SharpAIOpencuaTestFixtures.decode(String(decoding: data, as: UTF8.self))

        let flat = try config.qwen25VLConfiguration(from: data)
        XCTAssertEqual(flat.textConfiguration.ropeScaling?["mrope_section"]?.asInts(), [2, 3, 3])
    }

    func testRejectsConfigWithoutTextConfig() {
        let json = """
            {
                "model_type": "opencua",
                "vision_config": {}
            }
            """
        XCTAssertThrowsError(try SharpAIOpencuaTestFixtures.decode(json))
    }
}
