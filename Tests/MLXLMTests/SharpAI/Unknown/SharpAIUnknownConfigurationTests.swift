// SharpAI fork overlay tests for the "unknown" entry configs. Not part of upstream.
// Offline only: inline JSON. UNVERIFIED: key names follow vLLM sources, not real checkpoints.
// Written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIUnknownConfigurationTests: XCTestCase {

    private let eagleJSON = """
        {
            "architectures": ["EagleCohereForCausalLM"],
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_norm_eps": 1e-05,
            "vocab_size": 100,
            "logit_scale": 0.0625,
            "tie_word_embeddings": true
        }
        """

    private let panguJSON = """
        {
            "architectures": ["PanguProMoEV2ForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 500,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "n_routed_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 32,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "sandwich_norm": true
        }
        """

    private func decodeEagle(_ text: String) throws -> SharpAIEagleCohereConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIEagleCohereConfiguration.self, from: Data(text.utf8))
    }

    private func decodePangu(_ text: String) throws -> SharpAIPanguProMoEV2Configuration {
        try JSONDecoder.json5().decode(
            SharpAIPanguProMoEV2Configuration.self, from: Data(text.utf8))
    }

    func testEagleDecodesCohereFields() throws {
        let config = try decodeEagle(eagleJSON)

        XCTAssertEqual(config.architectures, ["EagleCohereForCausalLM"])
        XCTAssertEqual(config.base.hiddenSize, 32)
        XCTAssertEqual(config.base.hiddenLayers, 2)
        XCTAssertEqual(config.base.attentionHeads, 4)
        XCTAssertEqual(config.base.kvHeads, 2)
        XCTAssertEqual(config.base.vocabularySize, 100)
        XCTAssertEqual(config.base.logitScale, 0.0625, accuracy: 1e-6)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertNil(config.layerTypes)
        XCTAssertNil(config.slidingWindow)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testEagleRejectsSlidingWindowLayers() throws {
        let text = eagleJSON.replacingOccurrences(
            of: "\"tie_word_embeddings\": true",
            with: "\"tie_word_embeddings\": true, \"sliding_window\": 4, "
                + "\"layer_types\": [\"sliding_attention\", \"full_attention\"]")
        let config = try decodeEagle(text)

        XCTAssertEqual(config.slidingWindow, 4)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testEagleRejectsWrongArchitecture() throws {
        let text = eagleJSON.replacingOccurrences(
            of: "EagleCohereForCausalLM", with: "CohereForCausalLM")
        let config = try decodeEagle(text)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testPanguDecodesMoEFields() throws {
        let config = try decodePangu(panguJSON)

        XCTAssertEqual(config.architectures, ["PanguProMoEV2ForCausalLM"])
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 4)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.routedExperts, 16)
        XCTAssertEqual(config.expertsPerToken, 4)
        XCTAssertEqual(config.moeIntermediateSize, 32)
        XCTAssertEqual(config.sharedExperts, 1)
        XCTAssertEqual(config.firstDenseLayers, 1)
        XCTAssertEqual(config.routedScalingFactor, 2.5, accuracy: 1e-6)
        XCTAssertTrue(config.normTopKProb)
        XCTAssertTrue(config.sandwichNorm)
        XCTAssertEqual(config.paramSinkNumber, 0)
        XCTAssertNil(config.qLoraRank)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testPanguDefaultsWhenOptionalKeysAbsent() throws {
        let text = """
            {
                "architectures": ["PanguProMoEV2ForCausalLM"],
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "vocab_size": 500,
                "n_routed_experts": 16,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 32
            }
            """
        let config = try decodePangu(text)

        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.firstDenseLayers, 0)
        XCTAssertEqual(config.routedScalingFactor, 1, accuracy: 1e-6)
        XCTAssertFalse(config.normTopKProb)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNil(config.sharedExperts)
    }

    func testPanguRejectsTopKAboveExpertCount() throws {
        let text = panguJSON.replacingOccurrences(
            of: "\"num_experts_per_tok\": 4", with: "\"num_experts_per_tok\": 32")
        let config = try decodePangu(text)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testPanguRejectsDenseLayersAboveLayerCount() throws {
        let text = panguJSON.replacingOccurrences(
            of: "\"first_k_dense_replace\": 1", with: "\"first_k_dense_replace\": 9")
        let config = try decodePangu(text)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
