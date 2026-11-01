// SharpAI fork overlay tests for the bailing_hybrid creator and forward pass. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIBailingHybridRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "bailing_hybrid",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 512
        }
        """

    /// A tiny complete config: 4 layers with 2 lightning layers (0, 2) and 2 MLA layers (1, 3),
    /// dense MLP in layer 0 and MoE with group-limited routing in the rest.
    private let tinyJSON = """
        {
            "model_type": "bailing_hybrid",
            "hidden_size": 32,
            "intermediate_size": 48,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "n_group": 4,
            "topk_group": 2,
            "first_k_dense_replace": 1,
            "layer_group_size": 2,
            "group_norm_size": 2,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "routed_scaling_factor": 2.5,
            "head_dim": 8,
            "kv_lora_rank": 8,
            "qk_rope_head_dim": 4,
            "qk_nope_head_dim": 4,
            "v_head_dim": 8,
            "q_lora_rank": 12,
            "partial_rotary_factor": 0.5,
            "tie_word_embeddings": false
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIBailingHybridOverlay.creators["bailing_hybrid"])
    }

    func testConfigWithoutMLAFieldsIsRejectedBeforeConstruction() throws {
        guard let creator = SharpAIBailingHybridOverlay.creators["bailing_hybrid"] else {
            XCTFail("bailing_hybrid creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertTrue(message.contains("kv_lora_rank"), message)
        }
    }

    func testCreatorBuildsModelForCompleteConfig() throws {
        guard let creator = SharpAIBailingHybridOverlay.creators["bailing_hybrid"] else {
            XCTFail("bailing_hybrid creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIBailingHybridModel)
    }

    func testCreatorModelRunsForwardPass() throws {
        MLXRandom.seed(5)
        guard let creator = SharpAIBailingHybridOverlay.creators["bailing_hybrid"] else {
            XCTFail("bailing_hybrid creator missing")
            return
        }

        let built = try creator(Data(tinyJSON.utf8))
        guard let model = built as? SharpAIBailingHybridModel else {
            XCTFail("creator did not return SharpAIBailingHybridModel")
            return
        }
        let tokens = MLXArray([Int32(1), 5, 9]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64])
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIBailingHybridOverlay.creators["bailing_hybrid"] else {
            XCTFail("bailing_hybrid creator missing")
            return
        }
        let json = """
            { "hidden_size": 64 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testLightningRecurrenceMatchesStepwiseDefinition() throws {
        MLXRandom.seed(7)
        let B = 1
        let H = 2
        let L = 7
        let K = 3
        let V = 4
        let scale: Float = 0.5
        let q = MLXRandom.normal([B, H, L, K])
        let k = MLXRandom.normal([B, H, L, K])
        let v = MLXRandom.normal([B, H, L, V])
        let g = MLXArray([Float(-0.1), Float(-0.05)])

        // chunk 3 over L = 7 exercises several chunks and the padded tail.
        let (out, state) = SharpAIBailingHybridRecurrence.recurrentGLA(
            q: q, k: k, v: v, g: g, scale: scale, chunk: 3)

        // Definition from bailing_hybrid.recurrent_gla: h_t = exp(g) h_{t-1} + k_t^T v_t,
        // y_t = (scale q_t) h_t.
        var h = MLXArray.zeros([B, H, K, V], dtype: .float32)
        let decay = MLX.exp(g).reshaped(1, H, 1, 1)
        var ys: [MLXArray] = []
        for t in 0 ..< L {
            let kt = k[0..., 0..., t, 0...].expandedDimensions(axis: -1)
            let vt = v[0..., 0..., t, 0...].expandedDimensions(axis: -2)
            let qt = q[0..., 0..., t, 0...].expandedDimensions(axis: -2)
            h = h * decay + kt * vt
            ys.append((qt * scale).matmul(h))
        }
        let reference = concatenated(ys, axis: 2)

        XCTAssertEqual(out.shape, [B, H, L, V])
        XCTAssertTrue(allClose(out, reference, rtol: 1e-4, atol: 1e-5).item(Bool.self))
        XCTAssertTrue(allClose(state, h, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    func testLightningRecurrenceMaskedTailIsIgnored() throws {
        MLXRandom.seed(11)
        let B = 1
        let H = 2
        let L = 7
        let K = 3
        let V = 4
        let q = MLXRandom.normal([B, H, L, K])
        let k = MLXRandom.normal([B, H, L, K])
        let v = MLXRandom.normal([B, H, L, V])
        let g = MLXArray([Float(-0.1), Float(-0.05)])
        // The last two positions are padding: no state update and no output.
        let mask = MLXArray([Float(1), 1, 1, 1, 1, 0, 0]).reshaped(B, L)

        let masked = SharpAIBailingHybridRecurrence.recurrentGLA(
            q: q, k: k, v: v, g: g, scale: 0.5, mask: mask, chunk: 3)
        let prefix = SharpAIBailingHybridRecurrence.recurrentGLA(
            q: q[0..., 0..., ..<5, 0...], k: k[0..., 0..., ..<5, 0...],
            v: v[0..., 0..., ..<5, 0...], g: g, scale: 0.5, chunk: 3)

        XCTAssertTrue(
            allClose(
                masked.output[0..., 0..., ..<5, 0...], prefix.output, rtol: 1e-4, atol: 1e-5
            ).item(Bool.self))
        XCTAssertTrue(
            allClose(masked.state, prefix.state, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    func testTinyForwardPass() throws {
        MLXRandom.seed(3)
        let config = try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(tinyJSON.utf8))
        let model = try SharpAIBailingHybridModel(config)
        let tokens = MLXArray([Int32(1), 5, 9, 2, 7]).reshaped(1, 5)

        let full = model(tokens, cache: nil)
        XCTAssertEqual(full.shape, [1, 5, 64])
        XCTAssertTrue(all(isFinite(full)).item(Bool.self))

        // Prefill four tokens into the hybrid cache, then decode the fifth. The result must match
        // the single full forward, which exercises the MLA cache and the recurrent state together.
        let cache = try model.newCache(parameters: nil)
        let prefix = model(tokens[0..., ..<4], cache: cache)
        let step = model(tokens[0..., 4...], cache: cache)
        XCTAssertTrue(
            allClose(prefix, full[0..., ..<4], rtol: 1e-3, atol: 1e-3).item(Bool.self))
        XCTAssertTrue(
            allClose(step, full[0..., 4...], rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }
}
