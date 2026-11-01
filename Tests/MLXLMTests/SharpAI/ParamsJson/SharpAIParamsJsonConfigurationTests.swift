// SharpAI ParamsJson configuration tests. Decodes inline params.json strings only.
// testDecodesRealEagleParamsJSON decodes the verbatim params.json of the Mistral Large 3 EAGLE
// drafter (https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle).

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIParamsJsonConfigurationTests: XCTestCase {

    func testDecodesRealEagleParamsJSON() throws {
        let config = try SharpAIParamsJsonConfiguration.decode(
            Data(SharpAIParamsJsonTestFixtures.realEagleParamsJSON.utf8))

        XCTAssertEqual(config.dim, 7168)
        XCTAssertEqual(config.nLayers, 1)
        XCTAssertEqual(config.nHeads, 128)
        XCTAssertEqual(config.nKVHeads, 128)
        XCTAssertEqual(config.vocabSize, 131_072)
        XCTAssertEqual(config.normEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertEqual(config.qLoraRank, 1536)
        XCTAssertEqual(config.kvLoraRank, 512)
        XCTAssertEqual(config.qkNopeHeadDim, 128)
        XCTAssertEqual(config.qkRopeHeadDim, 64)
        XCTAssertEqual(config.vHeadDim, 128)
        // head_dim in params.json is 192 = 128 + 64, the MLA query head size.
        XCTAssertEqual(config.qHeadDim, 192)
        XCTAssertFalse(config.tiedEmbeddings)

        XCTAssertEqual(config.moe.numExperts, 128)
        XCTAssertEqual(config.moe.numExpertsPerTok, 4)
        XCTAssertEqual(config.moe.expertHiddenDim, 4096)
        XCTAssertEqual(config.moe.numSharedExperts, 1)
        XCTAssertEqual(config.moe.routedScale, 1.0)
        XCTAssertTrue(config.moe.normTopkProb)

        XCTAssertEqual(config.llama4Scaling?.beta, 0.1)
        XCTAssertEqual(config.llama4Scaling?.originalMaxPositionEmbeddings, 8192)

        // yarn carries factor 36 and apply_scale false; the type is set to "yarn".
        XCTAssertEqual(config.ropeScaling?["type"], .string("yarn"))
        XCTAssertEqual(config.ropeScaling?["factor"], .int(36))
        XCTAssertEqual(config.ropeScaling?["apply_scale"], .bool(false))
        // apply_scale false means no YaRN magnitude correction (vLLM attention_factor 1.0). The
        // mscale pair is equal, so the shared RoPE ratio is 1.0 and not the default 1.358.
        XCTAssertEqual(config.ropeScaling?["mscale"], .int(1))
        XCTAssertEqual(config.ropeScaling?["mscale_all_dim"], .int(1))
        XCTAssertNoThrow(try config.validate())
    }

    func testDecodesNativeParamsJSON() throws {
        let config = try SharpAIParamsJsonConfiguration.decode(
            Data(SharpAIParamsJsonTestFixtures.tinyParamsJSON.utf8))

        XCTAssertEqual(config.dim, 32)
        XCTAssertEqual(config.nLayers, 1)
        XCTAssertEqual(config.nHeads, 4)
        XCTAssertEqual(config.nKVHeads, 4)
        XCTAssertEqual(config.vocabSize, 64)
        XCTAssertEqual(config.qLoraRank, 16)
        XCTAssertEqual(config.kvLoraRank, 12)
        XCTAssertEqual(config.qkNopeHeadDim, 8)
        XCTAssertEqual(config.qkRopeHeadDim, 4)
        XCTAssertEqual(config.vHeadDim, 8)
        XCTAssertEqual(config.qHeadDim, 12)
        XCTAssertFalse(config.tiedEmbeddings)
        XCTAssertNil(config.ropeScaling)
    }

    func testDecodesMoEBlock() throws {
        let config = try SharpAIParamsJsonConfiguration.decode(
            Data(SharpAIParamsJsonTestFixtures.tinyParamsJSON.utf8))

        XCTAssertEqual(config.moe.numExperts, 4)
        XCTAssertEqual(config.moe.numExpertsPerTok, 2)
        XCTAssertEqual(config.moe.expertHiddenDim, 16)
        XCTAssertEqual(config.moe.numSharedExperts, 1)
        XCTAssertEqual(config.moe.routedScale, 1.0)
        XCTAssertTrue(config.moe.normTopkProb)
    }

    func testDecodesLlama4ScalingBlock() throws {
        let config = try SharpAIParamsJsonConfiguration.decode(
            Data(SharpAIParamsJsonTestFixtures.tinyParamsJSON.utf8))

        XCTAssertEqual(config.llama4Scaling?.beta, 0.1)
        XCTAssertEqual(config.llama4Scaling?.originalMaxPositionEmbeddings, 8)
    }

    func testOptionalKeysFallBackToDefaults() throws {
        let config = try SharpAIParamsJsonConfiguration.decode(
            Data(SharpAIParamsJsonTestFixtures.minimalParamsJSON.utf8))

        XCTAssertEqual(config.nKVHeads, 4)
        XCTAssertEqual(config.normEps, 1e-5, accuracy: 1e-9)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertEqual(config.maxPositionEmbeddings, 131_072)
        XCTAssertFalse(config.tiedEmbeddings)
        XCTAssertNil(config.llama4Scaling)
        XCTAssertEqual(config.moe.numSharedExperts, 0)
        XCTAssertEqual(config.moe.routedScale, 1.0)
        XCTAssertTrue(config.moe.normTopkProb)
    }

    func testYarnBlockSetsRopeTypeToYarn() throws {
        let json = SharpAIParamsJsonTestFixtures.minimalParamsJSON.replacingOccurrences(
            of: "\"q_lora_rank\": 16,",
            with: "\"yarn\": {\"factor\": 4.0},\n  \"q_lora_rank\": 16,")
        let config = try SharpAIParamsJsonConfiguration.decode(Data(json.utf8))

        XCTAssertEqual(config.ropeScaling?["type"], .string("yarn"))
        XCTAssertNotNil(config.ropeScaling?["factor"])
    }

    func testValidateAcceptsTinyConfig() throws {
        let config = try SharpAIParamsJsonConfiguration.decode(
            Data(SharpAIParamsJsonTestFixtures.tinyParamsJSON.utf8))

        XCTAssertNoThrow(try config.validate())
    }

    func testValidateRejectsTopKAboveExpertCount() throws {
        let json = SharpAIParamsJsonTestFixtures.tinyParamsJSON.replacingOccurrences(
            of: "\"num_experts_per_tok\": 2", with: "\"num_experts_per_tok\": 5")
        let config = try SharpAIParamsJsonConfiguration.decode(Data(json.utf8))

        XCTAssertThrowsError(try config.validate()) { error in
            guard case SharpAIParamsJsonError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidateRejectsOddRopeDim() throws {
        let json = SharpAIParamsJsonTestFixtures.tinyParamsJSON.replacingOccurrences(
            of: "\"qk_rope_head_dim\": 4", with: "\"qk_rope_head_dim\": 3")
        let config = try SharpAIParamsJsonConfiguration.decode(Data(json.utf8))

        XCTAssertThrowsError(try config.validate())
    }

    func testMissingMLAKeyFailsDecoding() {
        let json = SharpAIParamsJsonTestFixtures.tinyParamsJSON.replacingOccurrences(
            of: "\"q_lora_rank\": 16,", with: "")

        XCTAssertThrowsError(try SharpAIParamsJsonConfiguration.decode(Data(json.utf8)))
    }
}
