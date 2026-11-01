// SharpAI fork overlay tests for pangu_ultra_moe_mtp. Not part of upstream.
//
// The tiny base dimensions use the key names of SharpAIPanguUltraMoeConfiguration (the
// pangu_ultra_moe overlay in SharpAI/PanguUltraMoe), with small sizes. The forward test checks
// the output shape, that it is finite, and that it is causal. The reference Python was not run,
// so no output is compared with it.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIPanguUltraMoeMtpRegistrationTests: XCTestCase {

    /// Two main layers (layer 0 dense, layer 1 MoE) and one MTP layer at index 2.
    private let tinyJSON = """
        {
          "model_type": "pangu_ultra_moe_mtp",
          "architectures": ["OpenPanguMTPModel"],
          "hidden_size": 32,
          "intermediate_size": 48,
          "moe_intermediate_size": 8,
          "num_hidden_layers": 2,
          "num_nextn_predict_layers": 1,
          "num_dense_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "vocab_size": 50,
          "num_routed_experts": 4,
          "num_shared_experts": 1,
          "num_experts_per_tok": 2,
          "routed_scaling_factor": 2.5,
          "attention_q_lora_dim": 16,
          "attention_kv_lora_dim": 8,
          "attention_qk_dim": 8,
          "attention_qk_rope_dim": 4,
          "attention_v_dim": 8,
          "sandwich_norm": true,
          "rope_theta": 10000,
          "max_position_embeddings": 64
        }
        """

    private var creator: (Data) throws -> any LanguageModel {
        get throws {
            try XCTUnwrap(SharpAIPanguUltraMoeMtpOverlay.creators["pangu_ultra_moe_mtp"])
        }
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIPanguUltraMoeMtpOverlay.creators["pangu_ultra_moe_mtp"])
    }

    func testCreatorBuildsTinyModel() throws {
        let built = try creator(Data(tinyJSON.utf8))
        let model = try XCTUnwrap(built as? SharpAIPanguUltraMoeMtpModel)
        XCTAssertEqual(model.mtpStartLayerIndex, 2)
        XCTAssertEqual(model.numNextnPredictLayers, 1)
        XCTAssertEqual(model.kvHeads, [4])
    }

    func testCreatorRejectsConfigWithoutBaseDimensions() {
        let minimal = Data(
            (
                "{\"model_type\": \"pangu_ultra_moe_mtp\", \"architectures\": [\"OpenPanguMTPModel\"], "
                    + "\"num_hidden_layers\": 2, \"num_nextn_predict_layers\": 1}"
            ).utf8)
        XCTAssertThrowsError(try creator(minimal)) { error in
            if case SharpAIPanguUltraMoeMtpError.unsupported(_) = error {
                XCTFail("creator should build or reject on configuration, not report unsupported")
            }
        }
    }

    func testCreatorRejectsMissingNextnPredictLayers() {
        let json = tinyJSON.replacingOccurrences(
            of: "\"num_nextn_predict_layers\": 1,", with: "")
        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            switch error as? SharpAIPanguUltraMoeMtpError {
            case .invalidConfiguration(_)?: break
            default: XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testCreatorRejectsWrongModelTypeBeforeConstruction() {
        let wrong = Data("{\"model_type\": \"pangu_ultra_moe\"}".utf8)
        XCTAssertThrowsError(try creator(wrong)) { error in
            switch error as? SharpAIPanguUltraMoeMtpError {
            case .invalidConfiguration(_)?: break
            default: XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testErrorDescriptionMentionsModelType() {
        let description =
            SharpAIPanguUltraMoeMtpError.unsupported("test").errorDescription ?? ""
        XCTAssertTrue(description.contains("PanguUltraMoeMtp"))
    }

    func testSanitizeWithEmptyMappingPassesKeysThrough() {
        XCTAssertTrue(SharpAIPanguUltraMoeMtpSanitizer.keyMapping.isEmpty)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": MLXArray(Float(1))
        ]
        let out = SharpAIPanguUltraMoeMtpSanitizer.sanitize(weights)
        XCTAssertEqual(Set(out.keys), Set(weights.keys))
    }

    func testSanitizeWithEmptyInputReturnsEmpty() {
        let out = SharpAIPanguUltraMoeMtpSanitizer.sanitize([:])
        XCTAssertTrue(out.isEmpty)
    }

    func testSanitizeMTPKeepsOnlyMTPNamespaceAndRenames() {
        let zero = MLXArray(Float(0))
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": zero,
            "model.layers.0.self_attn.q_a_proj.weight": zero,
            "lm_head.weight": zero,
            "model.layers.2.embed_tokens.weight": zero,
            "model.layers.2.enorm.weight": zero,
            "model.layers.2.self_attn.q_a_proj.weight": zero,
            "model.layers.2.shared_head.head.weight": zero,
            "model.layers.2.shared_head.norm.weight": zero,
            "model.layers.2.model.rotary_emb.inv_freq": zero,
        ]
        let out = SharpAIPanguUltraMoeMtpSanitizer.sanitizeMTP(
            weights, numHiddenLayers: 2, numNextnPredictLayers: 1)
        XCTAssertEqual(
            Set(out.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.2.enorm.weight",
                "model.layers.2.mtp_block.self_attn.q_a_proj.weight",
                "model.layers.2.shared_head.norm.weight",
            ])
    }
}

// The config key names and MTP key rules come from vLLM's openpangu_mtp.py
// (`OpenPanguMultiTokenPredictor`, `get_spec_layer`, `_rewrite_spec_layer_name`). The class name
// is kept so the test identifiers stay stable.
final class SharpAIPanguUltraMoeMtpKnownIssueTests: XCTestCase {

    // Reads the layer-count keys the MTP layer index depends on. Values are a fixture: 61 hidden
    // layers and one next-token-prediction layer, the counts in the openPangu-Ultra-MoE-718B
    // config.json (FreedomIntelligence repo).
    func testConfigurationFieldsKnownIssue() throws {
        let json =
            "{\"model_type\": \"pangu_ultra_moe_mtp\", \"architectures\": [\"OpenPanguMTPModel\"], "
            + "\"num_hidden_layers\": 61, \"num_nextn_predict_layers\": 1}"
        let config = try JSONDecoder.json5().decode(
            SharpAIPanguUltraMoeMtpConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.hiddenLayers, 61)
        XCTAssertEqual(config.nextnPredictLayers, 1)

        let bare = try JSONDecoder.json5().decode(
            SharpAIPanguUltraMoeMtpConfiguration.self,
            from: Data("{\"model_type\": \"pangu_ultra_moe_mtp\"}".utf8))
        XCTAssertNil(bare.hiddenLayers)
        XCTAssertNil(bare.nextnPredictLayers)
    }

    // Checks the rewrite rules for a 61-layer main model with one MTP layer (layer 61).
    func testSanitizeKeyMappingKnownIssue() {
        let hidden = 61
        let mtp = 1
        func rewrite(_ key: String) -> String {
            SharpAIPanguUltraMoeMtpSanitizer.rewriteKey(
                key, numHiddenLayers: hidden, numNextnPredictLayers: mtp)
        }

        // Shared embedding moves to the top level.
        XCTAssertEqual(
            rewrite("model.layers.61.embed_tokens.weight"), "model.embed_tokens.weight")
        // Spec-layer weights keep their names.
        XCTAssertEqual(rewrite("model.layers.61.enorm.weight"), "model.layers.61.enorm.weight")
        XCTAssertEqual(rewrite("model.layers.61.hnorm.weight"), "model.layers.61.hnorm.weight")
        XCTAssertEqual(
            rewrite("model.layers.61.eh_proj.weight"), "model.layers.61.eh_proj.weight")
        XCTAssertEqual(
            rewrite("model.layers.61.shared_head.norm.weight"),
            "model.layers.61.shared_head.norm.weight")
        // Transformer-block weights of the MTP layer move under mtp_block.
        XCTAssertEqual(
            rewrite("model.layers.61.self_attn.q_a_proj.weight"),
            "model.layers.61.mtp_block.self_attn.q_a_proj.weight")
        // Main-model layers are not MTP keys and pass through unchanged.
        XCTAssertEqual(
            rewrite("model.layers.60.self_attn.q_a_proj.weight"),
            "model.layers.60.self_attn.q_a_proj.weight")
        XCTAssertEqual(
            rewrite("model.layers.0.mlp.gate_proj.weight"), "model.layers.0.mlp.gate_proj.weight")
        XCTAssertEqual(rewrite("model.norm.weight"), "model.norm.weight")
        XCTAssertFalse(
            SharpAIPanguUltraMoeMtpSanitizer.isMtpKey(
                "model.layers.62.enorm.weight", numHiddenLayers: hidden, numNextnPredictLayers: mtp))
    }

    // Tiny-config forward pass of the MTP head: the logits have the expected shape, are finite,
    // and are causal (changing the last input token leaves position 0 unchanged).
    func testTinyForwardPass() throws {
        let json =
            "{\"model_type\": \"pangu_ultra_moe_mtp\", \"architectures\": [\"OpenPanguMTPModel\"], "
            + "\"hidden_size\": 32, \"intermediate_size\": 48, \"moe_intermediate_size\": 8, "
            + "\"num_hidden_layers\": 2, \"num_nextn_predict_layers\": 1, \"num_dense_layers\": 1, "
            + "\"num_attention_heads\": 4, \"num_key_value_heads\": 4, \"vocab_size\": 50, "
            + "\"num_routed_experts\": 4, \"num_shared_experts\": 1, \"num_experts_per_tok\": 2, "
            + "\"routed_scaling_factor\": 2.5, \"attention_q_lora_dim\": 16, "
            + "\"attention_kv_lora_dim\": 8, \"attention_qk_dim\": 8, \"attention_qk_rope_dim\": 4, "
            + "\"attention_v_dim\": 8, \"sandwich_norm\": true, \"rope_theta\": 10000, "
            + "\"max_position_embeddings\": 64}"
        let creator = try XCTUnwrap(
            SharpAIPanguUltraMoeMtpOverlay.creators["pangu_ultra_moe_mtp"])
        let model = try XCTUnwrap(try creator(Data(json.utf8)) as? SharpAIPanguUltraMoeMtpModel)

        let hidden = 32
        let vocab = 50
        let length = 4
        let previous = MLXArray(
            (0 ..< length * hidden).map { Float(sin(Double($0))) }, [1, length, hidden])
        let weight = MLXArray(
            (0 ..< hidden * vocab).map { Float(cos(Double($0))) * 0.1 }, [hidden, vocab])
        let head: (MLXArray) -> MLXArray = { matmul($0, weight) }

        let tokens = MLXArray([Int32(3), 7, 11, 19], [1, length])
        let logits = model.draftLogits(inputIds: tokens, previousHidden: previous, head: head)
        XCTAssertEqual(logits.shape, [1, length, vocab])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })

        let changed = MLXArray([Int32(3), 7, 11, 2], [1, length])
        let changedLogits = model.draftLogits(
            inputIds: changed, previousHidden: previous, head: head)
        let first = logits[0..., 0, 0...].asArray(Float.self)
        let firstChanged = changedLogits[0..., 0, 0...].asArray(Float.self)
        XCTAssertEqual(first.count, firstChanged.count)
        XCTAssertTrue(zip(first, firstChanged).allSatisfy { abs($0 - $1) < 1e-5 })
    }
}
