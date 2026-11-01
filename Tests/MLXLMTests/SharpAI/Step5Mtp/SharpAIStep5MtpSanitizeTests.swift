// SharpAI fork overlay tests for step5_mtp key sanitization. Not part of upstream.
// These cases pin the sanitizer's rules against vLLM's Step5MTP.load_weights (vllm/models/step5/
// amd/mtp.py, vllm-project/vllm PR #61837, unmerged). They do not prove a real checkpoint layout.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIStep5MtpSanitizeTests: XCTestCase {

    /// Two dense layers plus one MTP layer at index 2.
    private let json = """
        {
            "model_type": "step5_mtp",
            "hidden_size": 64,
            "num_attention_heads": 8,
            "num_attention_groups": 2,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "vocab_size": 512
        }
        """

    private func configuration() throws -> SharpAIStep5MtpConfiguration {
        try SharpAIStep5MtpConfiguration.decode(from: Data(json.utf8))
    }

    private func mapped(_ key: String) throws -> String? {
        SharpAIStep5MtpSanitizer.mappedKey(key, configuration: try configuration())
    }

    func testKeepsNormAndProjectionKeysOnTheMTPLayer() throws {
        XCTAssertEqual(try mapped("model.layers.2.enorm.weight"), "model.layers.2.enorm.weight")
        XCTAssertEqual(try mapped("model.layers.2.hnorm.weight"), "model.layers.2.hnorm.weight")
        XCTAssertEqual(try mapped("model.layers.2.eh_proj.weight"), "model.layers.2.eh_proj.weight")
        XCTAssertEqual(
            try mapped("model.layers.2.shared_head.norm.weight"),
            "model.layers.2.shared_head.norm.weight")
    }

    func testRenamesSharedHeadOutputToHead() throws {
        XCTAssertEqual(
            try mapped("model.layers.2.shared_head.output.weight"),
            "model.layers.2.shared_head.head.weight")
    }

    func testMovesTransformerBlockKeysIntoMTPBlock() throws {
        XCTAssertEqual(
            try mapped("model.layers.2.transformer.self_attn.q_proj.weight"),
            "model.layers.2.mtp_block.self_attn.q_proj.weight")
        XCTAssertEqual(
            try mapped("model.layers.2.mlp.gate_proj.weight"),
            "model.layers.2.mtp_block.mlp.gate_proj.weight")
    }

    func testPromotesBareLayersPrefix() throws {
        XCTAssertEqual(try mapped("layers.2.enorm.weight"), "model.layers.2.enorm.weight")
    }

    func testMapsEmbedTokensToModelEmbedTokens() throws {
        XCTAssertEqual(try mapped("model.embed_tokens.weight"), "model.embed_tokens.weight")
        XCTAssertEqual(try mapped("embed_tokens.weight"), "model.embed_tokens.weight")
    }

    func testDropsDenseStackAndTopLevelHead() throws {
        XCTAssertNil(try mapped("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertNil(try mapped("model.layers.1.mlp.down_proj.weight"))
        XCTAssertNil(try mapped("lm_head.weight"))
    }

    func testDropsLayersPastTheMTPRange() throws {
        XCTAssertNil(try mapped("model.layers.3.enorm.weight"))
    }

    func testDropsRotaryBufferAndTokEmbeddings() throws {
        XCTAssertNil(try mapped("model.layers.2.rotary_emb.inv_freq"))
        XCTAssertNil(try mapped("model.layers.2.tok_embeddings.weight"))
    }

    func testSanitizeKeepsOnlyMappedKeysWithValues() throws {
        let config = try configuration()
        let weights: [String: MLXArray] = [
            "model.layers.2.shared_head.output.weight": MLXArray([1, 2] as [Int32]),
            "model.layers.2.transformer.mlp.gate_proj.weight": MLXArray([3, 4] as [Int32]),
            "model.embed_tokens.weight": MLXArray([5, 6] as [Int32]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([7, 8] as [Int32]),
            "lm_head.weight": MLXArray([9, 10] as [Int32]),
        ]

        let result = SharpAIStep5MtpSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(
            Set(result.keys),
            [
                "model.layers.2.shared_head.head.weight",
                "model.layers.2.mtp_block.mlp.gate_proj.weight",
                "model.embed_tokens.weight",
            ])
    }

    /// Stacking follows vLLM's stacked_params_mapping (q/k/v to qkv_proj, gate/up to gate_up_proj).
    /// The fused layout is checked here; the built module's parameter names are not.
    func testFusedProjectionStackingFusesQKVAndGateUp() throws {
        let q = MLXArray(Array(repeating: Float(1), count: 8)).reshaped([2, 4])
        let k = MLXArray(Array(repeating: Float(2), count: 8)).reshaped([2, 4])
        let v = MLXArray(Array(repeating: Float(3), count: 8)).reshaped([2, 4])
        let gate = MLXArray(Array(repeating: Float(4), count: 8)).reshaped([2, 4])
        let up = MLXArray(Array(repeating: Float(5), count: 8)).reshaped([2, 4])
        let weights: [String: MLXArray] = [
            "model.layers.2.mtp_block.self_attn.q_proj.weight": q,
            "model.layers.2.mtp_block.self_attn.k_proj.weight": k,
            "model.layers.2.mtp_block.self_attn.v_proj.weight": v,
            "model.layers.2.mtp_block.mlp.gate_proj.weight": gate,
            "model.layers.2.mtp_block.mlp.up_proj.weight": up,
        ]

        let result = SharpAIStep5MtpSanitizer.stackFusedProjections(weights)

        XCTAssertEqual(
            Set(result.keys),
            [
                "model.layers.2.mtp_block.self_attn.qkv_proj.weight",
                "model.layers.2.mtp_block.mlp.gate_up_proj.weight",
            ])
        let qkv = try XCTUnwrap(result["model.layers.2.mtp_block.self_attn.qkv_proj.weight"])
        XCTAssertEqual(qkv.shape, [6, 4])
        // Rows in shard order: q (1s), then k (2s), then v (3s).
        let expectedQKV = Array(repeating: Float(1), count: 8)
            + Array(repeating: Float(2), count: 8)
            + Array(repeating: Float(3), count: 8)
        XCTAssertEqual(qkv.asArray(Float.self), expectedQKV)
        let gateUp = try XCTUnwrap(result["model.layers.2.mtp_block.mlp.gate_up_proj.weight"])
        XCTAssertEqual(gateUp.shape, [4, 4])
        // Rows in shard order: gate (4s), then up (5s).
        let expectedGateUp = Array(repeating: Float(4), count: 8)
            + Array(repeating: Float(5), count: 8)
        XCTAssertEqual(gateUp.asArray(Float.self), expectedGateUp)
    }

    /// A partial group is left alone, and routed expert weights are never stacked.
    func testFusedStackingSkipsPartialGroupsAndExperts() throws {
        let part = MLXArray(Array(repeating: Float(1), count: 8)).reshaped([2, 4])
        let weights: [String: MLXArray] = [
            "model.layers.2.mtp_block.self_attn.q_proj.weight": part,
            "model.layers.2.mtp_block.self_attn.k_proj.weight": part,
            "model.layers.2.mtp_block.moe.experts.0.gate_proj.weight": part,
            "model.layers.2.mtp_block.moe.experts.0.up_proj.weight": part,
        ]

        let result = SharpAIStep5MtpSanitizer.stackFusedProjections(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }
}
