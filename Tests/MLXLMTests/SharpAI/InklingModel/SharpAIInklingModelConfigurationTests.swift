// SharpAI fork overlay tests for inkling_model config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIInklingModelConfigurationTests: XCTestCase {

    func testDecodesModelTypeAndArchitectures() throws {
        let json = """
            {
                "model_type": "inkling_model",
                "architectures": ["InklingForCausalLM"]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInklingModelConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "inkling_model")
        XCTAssertEqual(config.architectures, ["InklingForCausalLM"])
    }

    func testMissingFieldsUseDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIInklingModelConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "inkling_model")
        XCTAssertEqual(config.architectures, [])
    }

    func testUnknownFieldsAreIgnored() throws {
        // UNVERIFIED: dimension fields are not decoded yet, so unknown keys must not fail.
        let json = """
            {
                "model_type": "inkling_model",
                "architectures": ["InklingForCausalLM"],
                "hidden_size": 8192,
                "num_hidden_layers": 80
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInklingModelConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "inkling_model")
    }

    func testNestedTextConfigDerivesLayerPatterns() throws {
        // Published checkpoints nest the text backbone under `text_config`. With no
        // `local_layer_ids`, every 6th layer (1-based) is full attention. Layers below
        // `dense_mlp_idx` are dense, and `dense_intermediate_size` sets the dense width.
        let json = """
            {
                "model_type": "inkling_mm_model",
                "text_config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 12,
                    "dense_mlp_idx": 2,
                    "intermediate_size": 8,
                    "dense_intermediate_size": 48
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInklingModelConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "inkling_mm_model")
        XCTAssertEqual(config.text.hiddenSize, 16)
        XCTAssertEqual(config.text.slidingLayers.count, 12)
        XCTAssertTrue(config.text.slidingLayers[4])
        XCTAssertFalse(config.text.slidingLayers[5])
        XCTAssertFalse(config.text.slidingLayers[11])
        XCTAssertEqual(Array(config.text.denseLayers.prefix(3)), [true, true, false])
        XCTAssertEqual(config.text.intermediateSize, 48)
        XCTAssertEqual(config.text.moeIntermediateSize, 8)
    }

    private func assertDecodeFails(_ json: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIInklingModelConfiguration.self, from: Data(json.utf8)),
            file: file, line: line)
    }

    func testShortMlpLayerTypesThrows() {
        assertDecodeFails(
            """
            {"num_hidden_layers": 4, "mlp_layer_types": ["dense", "moe"]}
            """)
    }

    func testZeroKVHeadsThrows() {
        assertDecodeFails("""
            {"num_key_value_heads": 0}
            """)
        assertDecodeFails("""
            {"swa_num_key_value_heads": 0}
            """)
    }

    func testZeroConvKernelThrows() {
        assertDecodeFails("""
            {"sconv_kernel_size": 0}
            """)
    }
}
