// SharpAI fork overlay tests for the interns1_pro text decoder (FoPE, grouped router, MoE,
// KV-cache offsets, checkpoint key layout). Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// Tiny interns1_pro configs for unit tests. The full model (94 layers, 512 experts) is too large
/// to instantiate here. The fixture keeps the reference structure: FoPE with per-KV-head
/// coefficients, a grouped router with 2 groups, and an MoE in every layer.
enum SharpAIInterns1ProTinyFixture {
    static let vocabularySize = 64

    static let fope = """
        {
            "architectures": ["InternS1ProForConditionalGeneration"],
            "model_type": "interns1_pro",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "interns1_pro_text",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 48,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 64,
                "rms_norm_eps": 1e-06,
                "rope_theta": 10.0,
                "hidden_act": "silu",
                "attention_bias": false,
                "num_experts": 8,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 16,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "norm_topk_prob": true,
                "router_n_groups": 2,
                "rope_scaling": {
                    "rope_type": "default",
                    "fope_init_factor": 0.5,
                    "fope_sep_head": true,
                    "num_inv_freq": null
                }
            }
        }
        """

    /// Same model without rope_scaling, so plain RoPE is used.
    static let plain = """
        {
            "architectures": ["InternS1ProForConditionalGeneration"],
            "model_type": "interns1_pro",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "interns1_pro_text",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 48,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 64,
                "rms_norm_eps": 1e-06,
                "rope_theta": 10.0,
                "hidden_act": "silu",
                "attention_bias": false,
                "num_experts": 8,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 16,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "norm_topk_prob": true,
                "router_n_groups": 2
            }
        }
        """
}

final class SharpAIInterns1ProTextModelTests: XCTestCase {

    private func textConfig(_ json: String) throws -> SharpAIInterns1ProTextConfiguration {
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns1ProConfiguration.self, from: Data(json.utf8))
        return try XCTUnwrap(config.textConfig)
    }

    /// With identity coefficients the FoPE output reduces to the plain RoPE formula:
    /// cos/sin of (position * inv_freq[d mod D/2]), duplicated over the head dimension.
    /// Verified against: InternS1ProMoeTextFourierEmbedding.forward and the default rope init.
    func testFoPEWithIdentityCoefficientsMatchesPlainRoPE() throws {
        let config = try textConfig(SharpAIInterns1ProTinyFixture.fope)
        let embedding = SharpAIInterns1ProFourierEmbedding(config)

        // Offset 2 with length 3 covers positions 2, 3 and 4.
        let rotation = embedding.rotation(length: 3, offset: .scalar(2), dtype: .float32)
        XCTAssertTrue(rotation.sepHead)
        XCTAssertEqual(rotation.cos.shape, [1, 2, 3, 8])
        XCTAssertEqual(rotation.sin.shape, [1, 2, 3, 8])

        let cosValues = rotation.cos.asArray(Float.self)
        let sinValues = rotation.sin.asArray(Float.self)
        let half = 4
        let theta = Double(config.ropeTheta)
        for h in 0 ..< 2 {
            for p in 0 ..< 3 {
                for d in 0 ..< 8 {
                    let i = d % half
                    let invFreq = 1.0 / pow(theta, Double(2 * i) / 8.0)
                    let angle = Double(2 + p) * invFreq
                    let idx = (h * 3 + p) * 8 + d
                    XCTAssertEqual(Double(cosValues[idx]), cos(angle), accuracy: 1e-4)
                    XCTAssertEqual(Double(sinValues[idx]), sin(angle), accuracy: 1e-4)
                }
            }
        }
    }

    /// Router with zero weights gives uniform scores. Two groups of four experts with top-4
    /// overall must pick exactly two experts from each group, with equal normalised scores.
    /// Verified against: InternS1ProMoeTextGroupedRouter.forward.
    func testGroupedRouterPicksTopKPerGroup() throws {
        let config = try textConfig(SharpAIInterns1ProTinyFixture.fope)
        let router = SharpAIInterns1ProRouter(config)

        let (inds, scores) = router(MLXArray.ones([1, 3, 32]))

        XCTAssertEqual(inds.shape, [1, 3, 4])
        XCTAssertEqual(scores.shape, [1, 3, 4])
        let indexValues = inds.asArray(Int32.self)
        let scoreValues = scores.asArray(Float.self)
        for row in 0 ..< 3 {
            let picked = (0 ..< 4).map { Int(indexValues[row * 4 + $0]) }
            XCTAssertEqual(picked.filter { $0 < 4 }.count, 2, "row \(row) group 0: \(picked)")
            XCTAssertEqual(picked.filter { $0 >= 4 && $0 < 8 }.count, 2, "row \(row) group 1")
            for s in 0 ..< 4 {
                XCTAssertEqual(scoreValues[row * 4 + s], 0.25, accuracy: 1e-5)
            }
        }
    }

    /// Prefill of three tokens must match two tokens, then one more token through the KV cache.
    /// This checks that FoPE and RoPE use the cache offset for the new position.
    func testKVCacheStepMatchesFullForward() throws {
        guard let creator = SharpAIInterns1ProOverlay.creators["interns1_pro"] else {
            XCTFail("interns1_pro creator missing")
            return
        }
        let model = try creator(Data(SharpAIInterns1ProTinyFixture.fope.utf8))

        let full = model(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: nil)
        let cache = try model.newCache(parameters: nil)
        _ = model(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: cache)
        let step = model(MLXArray([Int32(3)]).reshaped(1, 1), cache: cache)

        let expected = full[0..., 2, 0...].asArray(Float.self)
        let actual = step[0..., 0, 0...].asArray(Float.self)
        XCTAssertEqual(expected.count, actual.count)
        for (e, a) in zip(expected, actual) {
            XCTAssertEqual(e, a, accuracy: 1e-3)
        }
    }

    /// The model's own parameter layout must accept the sanitized checkpoint keys with
    /// verify: .all. The checkpoint names come from model.safetensors.index.json, with per-expert
    /// tensors and the FoPE coefficients under model.language_model.rotary_emb.
    func testCheckpointKeysLoadWithVerifyAll() throws {
        let config = try textConfig(SharpAIInterns1ProTinyFixture.fope)
        let model = SharpAIInterns1ProModel(config, tieWordEmbeddings: false)

        let h = config.hiddenSize
        let hk = config.headDim
        let lm = "model.language_model"
        var weights: [String: MLXArray] = [
            "\(lm).embed_tokens.weight": MLXArray.zeros([64, h]),
            "\(lm).norm.weight": MLXArray.zeros([h]),
            "\(lm).rotary_emb.sin_coef": MLXArray.zeros([config.kvHeads, hk / 2, hk / 2]),
            "\(lm).rotary_emb.cos_coef": MLXArray.zeros([config.kvHeads, hk / 2, hk / 2]),
            "lm_head.weight": MLXArray.zeros([64, h]),
            // Vision keys are present in the full checkpoint. The text model must drop them.
            "model.visual.blocks.0.attn.qkv.weight": MLXArray.zeros([4, 4]),
        ]
        for l in 0 ..< config.hiddenLayers {
            let p = "\(lm).layers.\(l)"
            weights["\(p).input_layernorm.weight"] = MLXArray.zeros([h])
            weights["\(p).post_attention_layernorm.weight"] = MLXArray.zeros([h])
            weights["\(p).self_attn.q_proj.weight"] = MLXArray.zeros([4 * hk, h])
            weights["\(p).self_attn.k_proj.weight"] = MLXArray.zeros([2 * hk, h])
            weights["\(p).self_attn.v_proj.weight"] = MLXArray.zeros([2 * hk, h])
            weights["\(p).self_attn.o_proj.weight"] = MLXArray.zeros([h, 4 * hk])
            weights["\(p).self_attn.q_norm.weight"] = MLXArray.zeros([hk])
            weights["\(p).self_attn.k_norm.weight"] = MLXArray.zeros([hk])
            weights["\(p).mlp.gate.weight"] = MLXArray.zeros([config.numExperts, h])
            for e in 0 ..< config.numExperts {
                weights["\(p).mlp.experts.\(e).gate_proj.weight"] = MLXArray.zeros([16, h])
                weights["\(p).mlp.experts.\(e).up_proj.weight"] = MLXArray.zeros([16, h])
                weights["\(p).mlp.experts.\(e).down_proj.weight"] = MLXArray.zeros([h, 16])
            }
        }

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNil(sanitized["model.visual.blocks.0.attn.qkv.weight"])
        XCTAssertNil(sanitized["\(lm).layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertNotNil(sanitized["\(lm).rotary_emb.sin_coef"])
        XCTAssertNil(sanitized["\(lm).rotary_emb.inv_freq"])
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
    }
}
