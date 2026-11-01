// SharpAI fork overlay tests for kimi_k3_mtp. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIKimiK3MtpRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIKimiK3MtpOverlay.creators["kimi_k3_mtp"])
    }

    func testCreatorFailsWithBlockedOnBaseError() {
        guard let creator = SharpAIKimiK3MtpOverlay.creators["kimi_k3_mtp"] else {
            XCTFail("kimi_k3_mtp creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data("{}".utf8))) { error in
            XCTAssertEqual(error as? SharpAIKimiK3MtpError, .blockedOnBase)
        }
    }

    func testErrorDescriptionMentionsModelType() {
        let description = SharpAIKimiK3MtpError.blockedOnBase.errorDescription ?? ""
        XCTAssertTrue(description.contains("kimi_k3_mtp"))
    }

    func testSanitizeWithEmptyMappingIsNoOp() {
        let out = SharpAIKimiK3MtpSanitize.sanitize([:])
        XCTAssertTrue(out.isEmpty)
        XCTAssertTrue(SharpAIKimiK3MtpSanitize.keyMapping.isEmpty)
    }
}

final class SharpAIKimiK3MtpKnownIssueTests: XCTestCase {

    // Real fixture: scripts/conformance/offline/fixtures/kimi_k3/moonshotai__Kimi-K3.json.
    // The base text_config has num_nextn_predict_layers = 0, so the released checkpoint declares
    // no MTP layer; the MTP layer range is therefore empty for the base config.
    // Layer indices follow vLLM KimiK3MultiTokenPredictor.__init__ (mtp.py, raw GitHub).
    func testConfigurationDecodingFromBaseFixture() throws {
        let url = ConformanceBundle.root.appendingPathComponent(
            "fixtures/kimi_k3/moonshotai__Kimi-K3.json")
        let data = try Data(contentsOf: url)
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let text = try XCTUnwrap(root["text_config"] as? [String: Any])
        XCTAssertEqual(text["num_nextn_predict_layers"] as? Int, 0)

        let config = try JSONDecoder().decode(SharpAIKimiK3MtpConfiguration.self, from: data)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 93)
        XCTAssertEqual(config.textConfig.numNextnPredictLayers, 0)
        XCTAssertEqual(config.textConfig.vocabSize, 163_840)
        XCTAssertEqual(config.textConfig.hiddenSize, 7168)
        XCTAssertEqual(config.mtpStartLayerIdx, 93)
        XCTAssertEqual(config.numMtpLayers, 0)
        XCTAssertTrue(config.mtpLayerIndices.isEmpty)
    }

    func testConfigurationMtpLayerIndicesFollowVLLMFormula() throws {
        let json = """
            {"text_config": {"num_hidden_layers": 93, "num_nextn_predict_layers": 2,
                             "vocab_size": 163840, "hidden_size": 7168}}
            """
        let config = try JSONDecoder().decode(
            SharpAIKimiK3MtpConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.mtpStartLayerIdx, 93)
        XCTAssertEqual(config.numMtpLayers, 2)
        XCTAssertEqual(config.mtpLayerIndices, 93 ..< 95)
    }

    // Real source: vLLM nvidia/mtp.py _rewrite_spec_layer_name and load_weights preprocessing.
    // Layer 93 is only a test value for the spec layer index; the caller supplies the real one.
    func testSanitizeKeyMappingKnownIssue() throws {
        let layer = 93
        func rewrite(_ key: String, packed: Bool = false) -> String? {
            SharpAIKimiK3MtpSanitize.specLayerKey(key, specLayer: layer, renamePackedWeights: packed)
        }
        XCTAssertEqual(
            rewrite("language_model.model.layers.93.enorm.weight"),
            "model.layers.93.enorm.weight")
        XCTAssertEqual(
            rewrite("language_model.model.layers.93.hnorm.weight"),
            "model.layers.93.hnorm.weight")
        XCTAssertEqual(
            rewrite("language_model.model.layers.93.eh_proj.weight"),
            "model.layers.93.eh_proj.weight")
        XCTAssertEqual(
            rewrite("language_model.model.layers.93.shared_head.norm.weight"),
            "model.layers.93.shared_head.norm.weight")
        XCTAssertEqual(
            rewrite("language_model.model.layers.93.embed_tokens.weight"),
            "model.embed_tokens.weight")
        XCTAssertEqual(
            rewrite("language_model.model.layers.93.self_attn.q_a_proj.weight"),
            "model.layers.93.mtp_block.self_attn.q_a_proj.weight")
        XCTAssertEqual(
            rewrite("language_model.model.layers.93.mlp.experts.0.w1.weight_packed", packed: true),
            "model.layers.93.mtp_block.mlp.experts.0.w1.weight")
        XCTAssertNil(rewrite("language_model.model.layers.2.self_attn.q_proj.weight"))
        XCTAssertNil(rewrite("vision_tower.patch_embed.proj.weight"))
        XCTAssertNil(rewrite("language_model.model.layers.93.rotary_emb.inv_freq"))
    }

    // No model class exists yet. Verified: the vLLM predictor (raw mtp.py) builds one
    // KimiK3MultiTokenPredictorLayer per MTP index, each reusing the base decoder block.
    // Missing in Swift: (1) the base KimiK3 text backbone (KimiLinear KDA + MoE; only
    // SharpAIKimiK3Configuration/Sanitize exist under Libraries/MLXVLM/SharpAI/KimiK3),
    // (2) the body of KimiK3MultiTokenPredictorLayer (enorm/hnorm/eh_proj/shared_head and
    // mtp_block) - not in the raw mtp.py fetch, (3) a tiny config with num_nextn_predict_layers > 0,
    // because the released base config declares 0 MTP layers and the HF index shows no mtp tensors.
    func testTinyForwardPassKnownIssue() throws {
        XCTExpectFailure(
            "KNOWN GAP: no KimiK3MTPModel class. Missing: KimiK3MultiTokenPredictorLayer body "
                + "(enorm/hnorm/eh_proj/shared_head/mtp_block) from vllm/models/kimi_k3/nvidia/mtp.py "
                + "(predictor __init__/forward/compute_logits verified; layer class not in the raw fetch) "
                + "and the base KimiK3 text backbone (KDA + MoE) absent from MLXLLM/MLXVLM; "
                + "released config has num_nextn_predict_layers = 0"
        ) {
            let creator = SharpAIKimiK3MtpOverlay.creators["kimi_k3_mtp"]
            let model: (any LanguageModel)? = try? creator?(Data("{}".utf8))
            XCTAssertNotNil(model)
        }
    }
}
