// SharpAI fork overlay tests for bailing_hybrid checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIBailingHybridSanitizeTests: XCTestCase {

    private func configuration(tied: Bool, nextn: Int = 0) throws
        -> SharpAIBailingHybridConfiguration
    {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "tie_word_embeddings": \(tied),
                "num_nextn_predict_layers": \(nextn)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(json.utf8))
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    func testDropsLmHeadWhenEmbeddingsAreTied() throws {
        let config = try configuration(tied: true)
        let input = weights([
            "model.word_embeddings.weight",
            "lm_head.weight",
        ])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["model.word_embeddings.weight"])
        XCTAssertNil(out["lm_head.weight"])
    }

    func testKeepsLmHeadWhenEmbeddingsAreNotTied() throws {
        let config = try configuration(tied: false)
        let input = weights(["lm_head.weight"])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["lm_head.weight"])
    }

    func testDropsNextnLayerKeysPastHiddenLayers() throws {
        // UNVERIFIED: MTP layer assumed at index num_hidden_layers (here 2), one nextn layer.
        let config = try configuration(tied: false, nextn: 1)
        let input = weights([
            "model.layers.1.attention.query_key_value.weight",
            "model.layers.2.attention.query_key_value.weight",
            "model.layers.2.eh_proj.weight",
        ])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["model.layers.1.attention.query_key_value.weight"])
        XCTAssertNil(out["model.layers.2.attention.query_key_value.weight"])
        XCTAssertNil(out["model.layers.2.eh_proj.weight"])
    }

    func testKeepsLayerKeysPastNextnRange() throws {
        // With no nextn layers, an index past num_hidden_layers is kept, not dropped.
        let config = try configuration(tied: false, nextn: 0)
        let input = weights(["model.layers.2.attention.query_key_value.weight"])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["model.layers.2.attention.query_key_value.weight"])
    }

    func testMainLayerKeysKeepHFNames() throws {
        let config = try configuration(tied: false)
        let keys = [
            "model.word_embeddings.weight",
            "model.layers.0.attention.dense.weight",
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.norm.weight",
        ]

        let out = SharpAIBailingHybridSanitizer.sanitize(weights(keys), configuration: config)

        XCTAssertEqual(Set(out.keys), Set(keys))
    }

    func testLayerIndexParsing() {
        XCTAssertEqual(
            SharpAIBailingHybridSanitizer.layerIndex(of: "model.layers.7.mlp.weight"), 7)
        XCTAssertNil(SharpAIBailingHybridSanitizer.layerIndex(of: "model.word_embeddings.weight"))
        XCTAssertNil(SharpAIBailingHybridSanitizer.layerIndex(of: "model.layers"))
    }

    private func mlaConfiguration() throws -> SharpAIBailingHybridConfiguration {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 128,
                "vocab_size": 512,
                "num_experts": 3,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 2,
                "kv_lora_rank": 5,
                "v_head_dim": 3
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(json.utf8))
    }

    func testStacksExpertWeightsIntoSwitchMLP() throws {
        let config = try mlaConfiguration()
        var input = [String: MLXArray]()
        for e in 0 ..< 3 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                input["model.layers.0.mlp.experts.\(e).\(proj).weight"] = MLXArray.ones([8, 4]) * MLXArray(Float(e))
            }
        }

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertEqual(out.count, 3)
        let stackedGate = try XCTUnwrap(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(stackedGate.shape, [3, 8, 4])
        XCTAssertEqual(stackedGate[2].sum().item(Float.self), 64)
    }

    func testSplitsKvBProjIntoEmbedQAndUnembedOut() throws {
        let config = try mlaConfiguration()
        // heads 2, nope 4, v 3, kvLora 5: [2 * 7, 5]
        let kvB = MLXArray(0 ..< 70).reshaped(14, 5).asType(.float32)

        let out = SharpAIBailingHybridSanitizer.sanitize(
            ["model.layers.0.attention.kv_b_proj.weight": kvB], configuration: config)

        XCTAssertNil(out["model.layers.0.attention.kv_b_proj.weight"])
        let embedQ = try XCTUnwrap(out["model.layers.0.attention.embed_q.weight"])
        let unembedOut = try XCTUnwrap(out["model.layers.0.attention.unembed_out.weight"])
        XCTAssertEqual(embedQ.shape, [2, 5, 4])
        XCTAssertEqual(unembedOut.shape, [2, 3, 5])
        // Head 1 key row 0 is kv_b row 7; head 1 value row 0 is kv_b row 11.
        XCTAssertEqual(embedQ[1, 0..., 0].asArray(Float.self), [35, 36, 37, 38, 39])
        XCTAssertEqual(unembedOut[1, 0].asArray(Float.self), [55, 56, 57, 58, 59])
    }

    func testLightningSlopeIsNotAParameter() throws {
        let config = try mlaConfiguration()
        let layer = SharpAIBailingHybridLightningAttention(config, layerIdx: 0)
        let keys = layer.parameters().flattened().map { $0.0 }
        XCTAssertFalse(keys.contains { $0.hasSuffix("slope") })
    }

    func testExpertBiasIsOptionalInGate() throws {
        let off = try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self,
            from: Data(
                """
                {"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                 "intermediate_size": 128, "vocab_size": 512, "num_experts": 4,
                 "num_experts_per_tok": 2, "moe_router_enable_expert_bias": false}
                """.utf8))
        let gateOff = SharpAIBailingHybridGate(off)
        XCTAssertFalse(gateOff.parameters().flattened().contains { $0.0.hasSuffix("expert_bias") })

        let gateOn = SharpAIBailingHybridGate(try mlaConfiguration())
        XCTAssertTrue(gateOn.parameters().flattened().contains { $0.0.hasSuffix("expert_bias") })
    }
}
