import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("DeepseekV32 / glm_moe_dsa")
struct DeepseekV32Tests {

    /// GLM-5.2's real config shape: no top-level `rope_theta`, everything RoPE inside
    /// `rope_parameters`. Values are the shipped ones where they matter to decoding and
    /// shrunk where they only affect size.
    private func glmStyleConfigData(
        numHiddenLayers: Int = 4, indexTopk: Int = 2048
    ) -> Data {
        let json = """
        {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": \(numHiddenLayers),
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "routed_scaling_factor": 2.5,
            "kv_lora_rank": 16,
            "q_lora_rank": 32,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "qk_nope_head_dim": 16,
            "norm_topk_prob": true,
            "n_group": 1,
            "topk_group": 1,
            "num_experts_per_tok": 2,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
            "attention_bias": false,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "index_topk": \(indexTopk),
            "index_n_heads": 32,
            "index_head_dim": 128,
            "indexer_rope_interleave": true,
            "num_nextn_predict_layers": 1
        }
        """
        return json.data(using: .utf8)!
    }

    @Test("glm_moe_dsa config decodes rope_theta out of rope_parameters")
    func testRopeParametersUnpacking() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData())

        #expect(cfg.ropeTheta == 8_000_000)
        // rope_type "default" means plain RoPE. Forwarding the dict as rope_scaling
        // would send the V3 stack looking for yarn fields that are not there.
        #expect(cfg.ropeScaling == nil)
        #expect(cfg.indexTopk == 2048)
        #expect(cfg.numNextnPredictLayers == 1)
    }

    @Test("glm_moe_dsa maps to a model through the factory registry")
    func testRegistryEntry() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: glmStyleConfigData(), modelType: "glm_moe_dsa")
        #expect(model is DeepseekV32Model)

        let dsv32 = try await LLMTypeRegistry.shared.createModel(
            configuration: glmStyleConfigData(), modelType: "deepseek_v3_2")
        #expect(dsv32 is DeepseekV32Model)
    }

    @Test("DeepseekV32 forward pass produces finite logits of the right shape")
    func testForwardPass() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData())
        let model = DeepseekV32Model(cfg)

        let input = MLXArray(0 ..< 6).reshaped(1, 6)
        let logits = model(input, cache: nil)

        #expect(logits.shape == [1, 6, 128])
        let sum = logits.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    @Test("Sparse attention engages once cached length exceeds index_topk, and stays finite")
    func testSparseAttentionEngagesPastTopkAndStaysFinite() throws {
        // 6 tokens through a single forward pass against index_topk: 4 — the
        // indexer's cache reaches length 6 > 4 within that one call, so every
        // query position in this batch routes through the sparse (masked) path,
        // not just the dense fallback below-threshold regime the other tests
        // exercise.
        let seqLen = 6
        let sparseCfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(indexTopk: 4))
        let sparseModel = DeepseekV32Model(sparseCfg)

        let input = MLXArray(0 ..< seqLen).reshaped(1, seqLen)
        let sparseLogits = sparseModel(input, cache: nil)

        #expect(sparseLogits.shape == [1, seqLen, 128])
        let sum = sparseLogits.sum().item(Float.self)
        #expect(!sum.isNaN, "sparse-path logits contain NaN")
        #expect(!sum.isInfinite, "sparse-path logits contain Inf")

        // Finite and shape-correct isn't enough on its own to prove the mask is
        // doing anything — a broken mask that ends up all-true would pass that
        // check too. Compare against the same weights with index_topk raised
        // past seqLen (dense, indexer never engages) and confirm the two
        // genuinely diverge.
        let denseCfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(indexTopk: 1_000_000))
        let denseModel = DeepseekV32Model(denseCfg)
        denseModel.update(parameters: sparseModel.parameters())
        let denseLogits = denseModel(input, cache: nil)
        eval(sparseLogits, denseLogits)

        let maxAbsDiff = (sparseLogits - denseLogits).abs().max().item(Float.self)
        #expect(
            maxAbsDiff > 0,
            "sparse and dense attention produced identical output — the top-k mask had no effect"
        )
    }

    /// Every other test in this file passes `cache: nil`, which trivially
    /// side-steps any bug that only manifests once a real KVCache mutates its
    /// own `.offset` — exactly what happened here: the causal mask was built
    /// from `mainCache.offset` *after* `mainCache.update(...)` had already
    /// incremented it, sizing the mask one batch too wide and crashing the
    /// boolean AND/broadcast the moment this ran with a real cache. This test
    /// exercises that path directly: a real cache from `newCache`, prefilled
    /// past `index_topk` in one call (sparse engages), then one more decode
    /// step (`L == 1`, exercising the other offset-read site).
    @Test("Sparse attention with a real (non-nil) cache: prefill past index_topk, then decode")
    func testSparseAttentionWithRealCache() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(indexTopk: 4))
        let model = DeepseekV32Model(cfg)
        let cache = model.newCache(parameters: nil)

        let prefillInput = MLXArray(0 ..< 6).reshaped(1, 6)
        let prefillLogits = model(prefillInput, cache: cache)
        eval(prefillLogits)

        #expect(prefillLogits.shape == [1, 6, 128])
        let prefillSum = prefillLogits.sum().item(Float.self)
        #expect(!prefillSum.isNaN, "prefill logits contain NaN with a real cache")
        #expect(!prefillSum.isInfinite, "prefill logits contain Inf with a real cache")

        // Cache offset is now 6 > index_topk (4): every subsequent decode step
        // routes through the sparse path, exercising the L == 1 offset-read site.
        let decodeInput = MLXArray([6]).reshaped(1, 1)
        let decodeLogits = model(decodeInput, cache: cache)
        eval(decodeLogits)

        #expect(decodeLogits.shape == [1, 1, 128])
        let decodeSum = decodeLogits.sum().item(Float.self)
        #expect(!decodeSum.isNaN, "decode logits contain NaN with a real cache")
        #expect(!decodeSum.isInfinite, "decode logits contain Inf with a real cache")
    }

    /// The inertness property from #139: the reference indexer returns no
    /// selection at all until the cache is longer than `index_topk` — so a
    /// context at or under that length must be numerically indistinguishable
    /// from a model whose indexer never engages (`index_topk` effectively
    /// infinite). This is the self-check that needs no real checkpoint: build
    /// two identically-shaped, identically-seeded models that differ only in
    /// `index_topk`, and confirm they agree exactly below the smaller threshold.
    @Test("Below index_topk, output matches a model where the indexer never engages")
    func testDenseRegimeMatchesAcrossIndexTopkValues() throws {
        let seqLen = 6

        // `MLXRandom.seed()` is global, process-wide mutable state, and
        // swift-testing runs tests concurrently by default — reseeding then
        // constructing two models is not reliably reproducible when other
        // tests' own (unseeded) random construction can race with it. Instead,
        // build one model normally and copy its parameters onto the second —
        // deterministic regardless of any concurrent RNG activity elsewhere in
        // the process. `index_topk` isn't a stored/loaded parameter (it only
        // gates runtime control flow), so this leaves the two models identical
        // except for the one field under test.
        let smallTopkCfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(indexTopk: seqLen + 1))
        let smallTopkModel = DeepseekV32Model(smallTopkCfg)

        let largeTopkCfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(indexTopk: 1_000_000))
        let largeTopkModel = DeepseekV32Model(largeTopkCfg)
        largeTopkModel.update(parameters: smallTopkModel.parameters())
        eval(largeTopkModel)

        let input = MLXArray(0 ..< seqLen).reshaped(1, seqLen)
        let smallTopkLogits = smallTopkModel(input, cache: nil)
        let largeTopkLogits = largeTopkModel(input, cache: nil)
        eval(smallTopkLogits, largeTopkLogits)

        // seqLen (6) <= smallTopkCfg.indexTopk (7): the indexer's own "not longer
        // than index_topk yet" check means it returns nil for both models here —
        // this is exercising that stage 2's own inertness holds, not comparing
        // against a since-removed stage-1 implementation.
        let maxAbsDiff = (smallTopkLogits - largeTopkLogits).abs().max().item(Float.self)
        #expect(
            maxAbsDiff == 0,
            "identically-seeded models must match exactly below index_topk, diff=\(maxAbsDiff)"
        )
    }

    /// `DeepseekV32Model` doesn't get `KVCacheDimensionProvider`'s default
    /// `newCache` (its cache shape is a `CacheList` pair, not that protocol's
    /// flat array), so honoring `parameters.maxKVSize` — bounded/rotating cache
    /// for constant-memory long context — has to be done explicitly rather than
    /// inherited for free. Confirms both cache slots in the pair are actually
    /// `RotatingKVCache` when requested, and stay the default `KVCacheSimple`
    /// when not (so the main/indexer caches keep growing in lockstep either way).
    @Test("newCache honors maxKVSize with a RotatingKVCache pair")
    func testNewCacheHonorsMaxKVSize() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData())
        let model = DeepseekV32Model(cfg)

        let bounded = model.newCache(parameters: GenerateParameters(maxKVSize: 32))
        guard let firstBounded = bounded.first as? CacheList else {
            Issue.record("newCache(maxKVSize:) did not return CacheList")
            return
        }
        #expect(firstBounded[0] is RotatingKVCache)
        #expect(firstBounded[1] is RotatingKVCache)

        let unbounded = model.newCache(parameters: nil)
        guard let firstUnbounded = unbounded.first as? CacheList else {
            Issue.record("newCache(nil) did not return CacheList")
            return
        }
        #expect(firstUnbounded[0] is KVCacheSimple)
        #expect(firstUnbounded[1] is KVCacheSimple)
    }

    @Test("DeepseekV32 sanitize keeps indexer weights, still drops compressor weights")
    func testSanitizeKeepsIndexerDropsCompressor() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData())
        let model = DeepseekV32Model(cfg)

        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.0.self_attn.indexer.wk.weight": MLXArray.zeros([128, 64]),
            "model.layers.0.self_attn.indexer.weights_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.0.self_attn.indexer.k_norm.weight": MLXArray.zeros([128]),
            "model.layers.0.attn.compressor.wkv.weight": MLXArray.zeros([1, 1]),
        ]
        let sanitized = model.sanitize(weights: weights)

        // Stage 2 has a real Indexer module to load these into — unlike stage 1,
        // they must now survive sanitize or the checkpoint's indexer weights are
        // silently discarded and the module stays randomly initialised.
        #expect(sanitized.keys.contains("model.layers.0.self_attn.q_a_proj.weight"))
        #expect(sanitized.keys.contains("model.layers.0.self_attn.indexer.wk.weight"))
        #expect(sanitized.keys.contains("model.layers.0.self_attn.indexer.weights_proj.weight"))
        #expect(sanitized.keys.contains("model.layers.0.self_attn.indexer.k_norm.weight"))
        // .attn.compressor. is an unrelated, still-unimplemented feature — still dropped.
        #expect(!sanitized.keys.contains { $0.contains(".attn.compressor.") })
    }

    /// The trap this port had to avoid. DeepSeek-V3's sanitize dropped
    /// `model.layers.61` by string, because V3 has 61 layers and its multi-token
    /// prediction block sits at index 61. GLM-5.2 has 78 layers, so layer 61 is an
    /// ordinary load-bearing layer — the literal would have silently deleted it and
    /// left a model that loads and quietly computes the wrong thing.
    @Test("MTP-layer filtering follows num_hidden_layers, not a hardcoded index")
    func testLayer61IsKeptWhenItIsARealLayer() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(numHiddenLayers: 78))
        let model = DeepseekV32Model(cfg)

        let weights: [String: MLXArray] = [
            "model.layers.61.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.77.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.78.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(
            sanitized.keys.contains("model.layers.61.self_attn.q_a_proj.weight"),
            "layer 61 is a real layer in a 78-layer model and must survive")
        #expect(sanitized.keys.contains("model.layers.77.self_attn.q_a_proj.weight"))
        #expect(
            !sanitized.keys.contains("model.layers.78.self_attn.q_a_proj.weight"),
            "layer 78 is past the main stack (MTP) and must be dropped")
    }

    /// The same filter has to keep behaving for DeepSeek-V3 itself, where the literal
    /// 61 was correct.
    @Test("DeepSeek-V3 still drops its MTP layer at index 61")
    func testDeepseekV3MTPLayerStillDropped() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(numHiddenLayers: 61))
        let model = DeepseekV32Model(cfg)

        let weights: [String: MLXArray] = [
            "model.layers.60.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.61.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized.keys.contains("model.layers.60.self_attn.q_a_proj.weight"))
        #expect(!sanitized.keys.contains("model.layers.61.self_attn.q_a_proj.weight"))
    }
}
