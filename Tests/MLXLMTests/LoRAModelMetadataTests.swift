// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLLM

final class LoRAModelMetadataTests: XCTestCase {
    func testVLMMetadataUsesLanguageLayers() async throws {
        let configuration = Data(
            """
            {
              "model_type": "gemma3",
              "mm_tokens_per_image": 4,
              "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "sliding_window": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8
              },
              "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "patch_size": 2,
                "image_size": 4
              }
            }
            """.utf8
        )
        let result = try await VLMModelFactory.shared.loraMetadata(
            configurationData: configuration)
        let metadata = try XCTUnwrap(result)

        XCTAssertEqual(metadata.layerCount, 2)
        XCTAssertEqual(metadata.defaultKeys, Self.defaultKeys)
    }

    func testMetadataPreservesCallerRandomSequence() async throws {
        let registry = ModelTypeRegistry<LanguageModel>()
        await registry.registerModelType("metadata_test") { _ in MetadataTestModel() }
        await registry.registerModelType("throwing_test") { _ in
            _ = MLXRandom.uniform()
            throw TestError.construction
        }
        let factory = LLMModelFactory(
            typeRegistry: registry, modelRegistry: AbstractModelRegistry())
        let expected = MLXRandom.RandomState(seed: 42)

        try await withRandomState(MLXRandom.RandomState(seed: 42)) {
            XCTAssertEqual(resolve().asArray(UInt32.self), expected.next().asArray(UInt32.self))
            for _ in 0 ..< 2 {
                _ = try await factory.loraMetadata(
                    configurationData: Data(#"{"model_type":"metadata_test"}"#.utf8))
                XCTAssertEqual(
                    resolve().asArray(UInt32.self), expected.next().asArray(UInt32.self))
            }

            do {
                _ = try await factory.loraMetadata(
                    configurationData: Data(#"{"model_type":"throwing_test"}"#.utf8))
                XCTFail("Expected the constructor error")
            } catch TestError.construction {
                XCTAssertEqual(
                    resolve().asArray(UInt32.self), expected.next().asArray(UInt32.self))
            }
        }
    }

    func testModelWithoutLoRAConformanceReturnsNil() async throws {
        let registry = ModelTypeRegistry<LanguageModel>()
        await registry.registerModelType("plain_test") { _ in PlainTestModel() }
        let result = try await registry.loraMetadata(
            configurationData: Data(#"{"model_type":"plain_test"}"#.utf8))
        XCTAssertNil(result)
    }

    func testUnknownModelTypeThrows() async throws {
        let registry = ModelTypeRegistry<LanguageModel>()
        do {
            _ = try await registry.loraMetadata(
                configurationData: Data(#"{"model_type":"unknown"}"#.utf8))
            XCTFail("Expected an unsupported model type error")
        } catch ModelFactoryError.unsupportedModelType(let type) {
            XCTAssertEqual(type, "unknown")
        }
    }

    func testInvalidConfigurationThrows() async throws {
        do {
            _ = try await LLMModelFactory.shared.loraMetadata(
                configurationData: Data(#"{"model_type":"qwen3"}"#.utf8))
            XCTFail("Expected a configuration decoding error")
        } catch is DecodingError {
        }
    }

    func testFactoryUsesRegisteredModelLoRAMetadata() async throws {
        let typeRegistry = ModelTypeRegistry<LanguageModel>()
        await typeRegistry.registerModelType("metadata_test") { _ in
            MetadataTestModel()
        }
        let factory = LLMModelFactory(
            typeRegistry: typeRegistry,
            modelRegistry: AbstractModelRegistry()
        )

        let result = try await factory.loraMetadata(
            configurationData: Data(#"{"model_type":"metadata_test"}"#.utf8)
        )
        let metadata = try XCTUnwrap(result)

        XCTAssertEqual(metadata.layerCount, 1)
        XCTAssertEqual(metadata.defaultKeys, ["runtime.a_proj", "runtime.z_proj"])
    }

    func testBuiltInModelMetadataUsesRuntimeModulePaths() async throws {
        let configuration = Data(
            """
            {
              "model_type": "qwen3",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "rms_norm_eps": 1e-5,
              "vocab_size": 64
            }
            """.utf8
        )

        let result = try await LLMModelFactory.shared.loraMetadata(
            configurationData: configuration
        )
        let metadata = try XCTUnwrap(result)

        XCTAssertEqual(metadata.layerCount, 2)
        XCTAssertEqual(metadata.defaultKeys, Self.defaultKeys)
    }

    private static let defaultKeys = [
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    ]

    private enum TestError: Error {
        case construction
    }
}

private class PlainTestModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int] = []

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inputs
    }
}

private final class MetadataTestModel: PlainTestModel, LoRAModel {
    private let layer = Linear(4, 4)

    var loraLayers: [Module] { [layer] }
    var loraDefaultKeys: [String] {
        // Exercise random-state isolation during metadata access as well as construction.
        _ = MLXRandom.uniform()
        return ["runtime.z_proj", "runtime.a_proj"]
    }
}
