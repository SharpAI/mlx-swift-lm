import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIBailingMoeV3Tests: XCTestCase {

    /// Tiny MLA-only config. With layer_group_size (4) > num_hidden_layers (2), no layer is KDA.
    private func makeConfig(
        hiddenLayers: Int = 2,
        layerGroupSize: Int = 4,
        numExperts: Int = 4,
        extraJSON: String = ""
    ) throws -> SharpAIBailingMoeV3Configuration {
        let json = """
            {
                "model_type": "bailing_moe_v3",
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": \(hiddenLayers),
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "q_lora_rank": null,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4,
                "v_head_dim": 4,
                "num_experts": \(numExperts),
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "num_shared_experts": 1,
                "moe_shared_expert_intermediate_size": 8,
                "first_k_dense_replace": 1,
                "n_group": 2,
                "topk_group": 1,
                "routed_scaling_factor": 2.5,
                "layer_group_size": \(layerGroupSize),
                "gated_attention_proj_granularity_type": "head_wise",
                "tie_word_embeddings": false,
                "use_bias": false\(extraJSON)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIBailingMoeV3Configuration.self, from: Data(json.utf8))
    }

    // MARK: Configuration

    func testConfigurationDecodesInlineJSON() throws {
        let config = try makeConfig()
        XCTAssertEqual(config.vocabularySize, 32)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertNil(config.qLoraRank)
        XCTAssertEqual(config.kvLoraRank, 8)
        XCTAssertEqual(config.numExperts, 4)
        XCTAssertEqual(config.routedScalingFactor, 2.5)
        XCTAssertNil(config.mropeSection)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testConfigurationDecodesMropeSection() throws {
        let config = try makeConfig(extraJSON: ", \"mrope_section\": [0, 2, 2]")
        XCTAssertEqual(config.mropeSection, [0, 2, 2])
    }

    func testValidationAcceptsKDALayers() throws {
        // 8 layers with group size 4: layers 0-2 and 4-6 are KDA. KDA is implemented now.
        let config = try makeConfig(hiddenLayers: 8, layerGroupSize: 4)
        XCTAssertTrue(config.isKDALayer(0))
        XCTAssertFalse(config.isKDALayer(3))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsLowRankKDA() throws {
        // no_kda_lora = false needs f_a/f_b and g_a/g_b projections, which are not implemented.
        let config = try makeConfig(
            hiddenLayers: 8, layerGroupSize: 4, extraJSON: ", \"no_kda_lora\": false")
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsSwigluClamp() throws {
        let config = try makeConfig(
            extraJSON: ", \"expert_swiglu_limit_list\": [0, 4]")
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsBadMropeSection() throws {
        let config = try makeConfig(extraJSON: ", \"mrope_section\": [8, 12, 12]")
        // qk_rope_head_dim / 2 is 2 here, so [8, 12, 12] cannot match.
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    // MARK: Sanitize

    func testSanitizeStacksRoutedExpertsAndDropsMTPLayers() throws {
        let model = SharpAIBailingMoeV3Model(try makeConfig())
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 4 {
            weights["model.layers.1.mlp.experts.\(e).gate_proj.weight"] = MLXArray.zeros([8, 16])
        }
        weights["model.layers.2.self_attn.q_proj.weight"] = MLXArray.zeros([1])
        weights["model.layers.0.attention.rotary_emb.inv_freq"] = MLXArray.zeros([2])
        weights["model.word_embeddings.weight"] = MLXArray.zeros([32, 16])

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.2.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["model.layers.0.attention.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.word_embeddings.weight"])
    }

    func testSanitizeFlattensKDADecayParameters() throws {
        let model = SharpAIBailingMoeV3Model(try makeConfig())
        let weights: [String: MLXArray] = [
            "model.layers.0.attention.A_log": MLXArray.zeros([1, 1, 4, 1]),
            "model.layers.0.attention.dt_bias": MLXArray.zeros([1, 32]),
            "model.layers.1.attention.A_log": MLXArray.zeros([4]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized["model.layers.0.attention.A_log"]?.shape, [4])
        XCTAssertEqual(sanitized["model.layers.0.attention.dt_bias"]?.shape, [32])
        XCTAssertEqual(sanitized["model.layers.1.attention.A_log"]?.shape, [4])
    }

    // MARK: mRoPE

    func testMRoPEMatchesInterleavedRoPEForTextPositions() throws {
        let x = MLXArray((0 ..< 32).map { Float($0) * 0.1 }).reshaped(1, 1, 4, 8)
        let rope = SharpAIBailingMoeV3RoPE(dims: 8, base: 10000, mropeSection: nil)
        let ours = rope(x, offset: nil)
        let reference = MLXFast.RoPE(
            x, dimensions: 8, traditional: true, base: 10000, scale: 1.0, offset: 0)
        eval(ours, reference)
        XCTAssertTrue(allClose(ours, reference, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    func testMRoPEWithSectionsMatchesTextPositionsWhenAxesAgree() throws {
        // With T = H = W, any section split must reduce to plain RoPE.
        let x = MLXArray((0 ..< 32).map { Float($0) * 0.1 }).reshaped(1, 1, 4, 8)
        let plain = SharpAIBailingMoeV3RoPE(dims: 8, base: 10000, mropeSection: nil)
        let sectioned = SharpAIBailingMoeV3RoPE(dims: 8, base: 10000, mropeSection: [0, 2, 2])
        let positions = MLXArray([0, 1, 2, 3] as [Float]).reshaped(1, 4)
        let a = plain.apply(x, temporal: positions, height: positions, width: positions)
        let b = sectioned.apply(x, temporal: positions, height: positions, width: positions)
        eval(a, b)
        XCTAssertTrue(allClose(a, b, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    // MARK: Forward

    func testForwardPassProducesLogitsShape() throws {
        let model = SharpAIBailingMoeV3Model(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testForwardPassUpdatesCacheExactlyOnce() throws {
        let model = SharpAIBailingMoeV3Model(try makeConfig())
        let cache = try model.newCache(parameters: nil)
        let tokenCount = 5
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)
        let logits = model(inputs, cache: cache)
        eval(logits)
        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(
                layerCache.offset, tokenCount,
                "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens")
        }
    }

    // MARK: KDA (linear attention)

    func testKDALogDecayMatchesReferenceGateFormulas() throws {
        // f: [1, 1, H, D] with H = 2, D = 3. A_log gives exp(A) = 1 and 2 per head.
        let heads = 2
        let dim = 3
        let fValues: [Float] = [0.5, -1.0, 2.0, 0.0, 0.25, -0.75]
        let dtValues: [Float] = [0.1, 0.0, -0.2, 0.3, 0.0, 0.5]
        let aExp: [Float] = [1, 2]
        let f = MLXArray(fValues).reshaped(1, 1, heads, dim)
        let aLog = MLXArray([Float(0), Float(log(2.0))])
        let dtBias = MLXArray(dtValues)

        let safe = SharpAIBailingMoeV3KDA.logDecay(
            f: f, aLog: aLog, dtBias: dtBias, numHeads: heads, headDim: dim,
            safeGate: true, lowerBound: -5)
        let plain = SharpAIBailingMoeV3KDA.logDecay(
            f: f, aLog: aLog, dtBias: dtBias, numHeads: heads, headDim: dim,
            safeGate: false, lowerBound: -5)
        eval(safe, plain)
        let safeValues = safe.reshaped(-1).asArray(Float.self)
        let plainValues = plain.reshaped(-1).asArray(Float.self)

        for h in 0 ..< heads {
            for d in 0 ..< dim {
                let i = h * dim + d
                let biased = Double(fValues[i]) + Double(dtValues[i])
                let z = Double(aExp[h]) * biased
                // Safe gate: lower_bound * sigmoid(exp(A) * (f + dt_bias)).
                let expectedSafe = Float(-5 * (1 / (1 + exp(-z))))
                // Plain gate: -exp(A) * softplus(f + dt_bias).
                let expectedPlain = Float(-Double(aExp[h]) * log(1 + exp(biased)))
                XCTAssertEqual(safeValues[i], expectedSafe, accuracy: 1e-5)
                XCTAssertEqual(plainValues[i], expectedPlain, accuracy: 1e-5)
            }
        }
    }

    func testGatedDeltaRecurrenceMatchesNaiveReference() throws {
        // Naive fla recurrence per step: S = S * exp(g); S += outer(beta * k, v - k^T S);
        // o = q^T S. State layout here is [B, H, K, V].
        MLXRandom.seed(7)
        let (B, T, H, D) = (1, 5, 2, 4)
        let q = MLXRandom.normal([B, T, H, D])
        let k = MLXRandom.normal([B, T, H, D])
        let v = MLXRandom.normal([B, T, H, D])
        let decay = MLX.exp(-MLXRandom.uniform(low: 0, high: 2, [B, T, H, D]))
        let beta = sigmoid(MLXRandom.normal([B, T, H]))

        var state = MLXArray.zeros([B, H, D, D])
        var outputs: [MLXArray] = []
        for t in 0 ..< T {
            let qt = q[0..., t, 0..., 0...].reshaped(B, H, D).expandedDimensions(axis: -1)
            let kt = k[0..., t, 0..., 0...].reshaped(B, H, D)
            let vt = v[0..., t, 0..., 0...].reshaped(B, H, D)
            let gt = decay[0..., t, 0..., 0...].reshaped(B, H, D).expandedDimensions(axis: -1)
            let bt = beta[0..., t, 0...].reshaped(B, H, 1)
            state = state * gt
            let kTimesS = (kt.expandedDimensions(axis: -1) * state).sum(axis: -2)
            let residual = vt - kTimesS
            let kb = (kt * bt).expandedDimensions(axis: -1)
            state = state + kb * residual.expandedDimensions(axis: -2)
            let o = (qt * state).sum(axis: -2)
            outputs.append(o)
        }
        let naive = MLX.stacked(outputs, axis: 1)  // [B, T, H, D]

        let (actual, _) = SharpAIOverlayGatedDeltaKDA.update(
            q: q, k: k, v: v, g: decay, beta: beta, state: nil, mask: nil, useKernel: false)
        eval(naive, actual)
        XCTAssertEqual(actual.shape, [B, T, H, D])
        XCTAssertTrue(allClose(actual, naive, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    func testKDAForwardPrefillMatchesTokenByTokenDecode() throws {
        // hidden_layers 4 with group size 4: layers 0-2 are KDA, layer 3 is MLA.
        let config = try makeConfig(hiddenLayers: 4, layerGroupSize: 4)
        XCTAssertTrue(config.isKDALayer(0))
        XCTAssertFalse(config.isKDALayer(3))
        let model = SharpAIBailingMoeV3Model(config)
        let tokens = MLXArray([1, 5, 9, 3, 7] as [Int32]).reshaped(1, 5)

        let prefillCache = try model.newCache(parameters: nil)
        let prefill = model(tokens, cache: prefillCache)

        let stepCache = try model.newCache(parameters: nil)
        var steps: [MLXArray] = []
        for t in 0 ..< 5 {
            steps.append(model(tokens[0..., t ..< (t + 1)], cache: stepCache))
        }
        let incremental = concatenated(steps, axis: 1)
        eval(prefill, incremental)

        XCTAssertEqual(prefill.shape, [1, 5, 32])
        XCTAssertTrue(allClose(prefill, incremental, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        // Every layer advanced by exactly the five tokens, in both runs.
        for cache in [prefillCache, stepCache] {
            for (i, layerCache) in cache.enumerated() {
                XCTAssertEqual(layerCache.offset, 5, "layer \(i) offset")
            }
        }
        // KDA slots: conv states [B, K-1, H*d] and recurrent state [B, H, d, d].
        let kda = prefillCache[0] as? ArraysCache
        XCTAssertEqual(kda?[0]?.shape, [1, 3, 16])
        XCTAssertEqual(kda?[3]?.shape, [1, 2, 8, 8])
    }
}
