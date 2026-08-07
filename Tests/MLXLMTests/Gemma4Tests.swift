import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct Gemma4Tests {

    /// Create a minimal test configuration for Gemma 4 using upstream's JSON-based init
    private func makeTinyConfigData() -> Data {
        let json = """
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "global_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 2,
                "rope_traditional": false,
                "sliding_window": 128,
                "sliding_window_pattern": 1,
                "max_position_embeddings": 512,
                "num_kv_shared_layers": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "hidden_size_per_layer_input": 32,
                "vocab_size_per_layer_input": 10,
                "final_logit_softcapping": 30.0,
                "enable_moe_block": false,
                "attention_k_eq_v": false
            },
            "vocab_size": 100
        }
        """
        return json.data(using: .utf8)!
    }

    /// A config with KV-shared layers, mirroring gemma-4-e4b-it-4bit's shape
    /// (num_hidden_layers 42 / num_kv_shared_layers 18) at tiny scale.
    private func makeKVSharedConfigData(layers: Int = 4, shared: Int = 2) -> Data {
        let json = """
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": \(layers),
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "global_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 2,
                "rope_traditional": false,
                "sliding_window": 128,
                "sliding_window_pattern": 1,
                "max_position_embeddings": 512,
                "num_kv_shared_layers": \(shared),
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "hidden_size_per_layer_input": 32,
                "vocab_size_per_layer_input": 10,
                "final_logit_softcapping": 30.0,
                "enable_moe_block": false,
                "attention_k_eq_v": false
            },
            "vocab_size": 100
        }
        """
        return json.data(using: .utf8)!
    }

    /// KV-shared layers ship no k_proj/v_proj/k_norm weights, so the model must not
    /// create those parameters — building them made update(verify: .all) demand tensors
    /// the checkpoint does not contain, and gemma-4-e4b-it-4bit failed to load at the
    /// first shared layer (SwiftLM #120).
    @Test("Gemma 4 KV-shared layers build no K/V projections")
    func testKVSharedLayersOmitKVProjections() throws {
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: makeKVSharedConfigData(layers: 4, shared: 2))
        let model = Gemma4Model(config)
        let keys = Set(model.parameters().flattened().map { $0.0 })

        // Boundary is 4 - 2 = 2: layers 0-1 own their K/V, layers 2-3 share.
        for owning in 0..<2 {
            #expect(keys.contains { $0.contains("layers.\(owning).self_attn.k_proj") },
                    "layer \(owning) owns its K/V and must have k_proj")
        }
        for shared in 2..<4 {
            for absent in ["k_proj", "v_proj", "k_norm"] {
                #expect(!keys.contains { $0.contains("layers.\(shared).self_attn.\(absent)") },
                        "KV-shared layer \(shared) must not build \(absent)")
            }
        }
    }

    /// A forward pass with no explicit cache must not trap on a shared layer: those
    /// layers have no projections to fall back on, so shared state has to reach them
    /// regardless of caching (review follow-up on mlx-swift-lm#44).
    @Test("Gemma 4 KV-shared forward without a cache")
    func testKVSharedForwardWithoutCache() throws {
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: makeKVSharedConfigData(layers: 4, shared: 2))
        let model = Gemma4Model(config)
        let tokens = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let logits = model(tokens, cache: nil)
        eval(logits)
        #expect(logits.shape.last == 100)
    }

    /// A checkpoint may ship k_proj/v_proj/k_norm for KV-shared layers even though the
    /// model does not build them — gemma-4-e2b-it-4bit does, gemma-4-e4b-it-4bit does
    /// not, with the same num_kv_shared_layers semantics. sanitize must discard the
    /// vestigial ones, or update(verify: .all) fails on the checkpoints that include
    /// them (regression from #44, caught in review of #45).
    @Test("Gemma 4 sanitize drops vestigial KV-shared projections")
    func testSanitizeDropsVestigialSharedKVWeights() throws {
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: makeKVSharedConfigData(layers: 4, shared: 2))
        let model = Gemma4Model(config)

        // A checkpoint that ships K/V for every layer, e2b-style.
        var weights: [String: MLXArray] = [:]
        for layer in 0..<4 {
            for name in ["k_proj", "v_proj", "k_norm", "q_proj", "o_proj"] {
                weights["language_model.model.layers.\(layer).self_attn.\(name).weight"] =
                    MLXArray.zeros([4, 4])
            }
        }

        let sanitized = model.sanitize(weights: weights)

        // Boundary is 4 - 2 = 2: layers 0-1 keep their K/V, layers 2-3 lose it.
        for owning in 0..<2 {
            #expect(sanitized.keys.contains { $0.contains("layers.\(owning).self_attn.k_proj") })
        }
        for shared in 2..<4 {
            for dropped in ["k_proj", "v_proj", "k_norm"] {
                #expect(!sanitized.keys.contains { $0.contains("layers.\(shared).self_attn.\(dropped)") },
                        "layer \(shared) is KV-shared, so \(dropped) must be discarded")
            }
            // q_proj and o_proj belong to every layer and must survive.
            #expect(sanitized.keys.contains { $0.contains("layers.\(shared).self_attn.q_proj") })
        }
    }

    @Test("Gemma 4 Configuration Decoding")
    func testGemma4ConfigDecoding() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        // vocabSize is internal, verify via model
        let model = Gemma4Model(config)
        #expect(model.vocabularySize == 100)
    }

    @Test("Gemma 4 Model Instantiation")
    func testGemma4ModelInstantiation() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)
        #expect(model.vocabularySize == 100)
    }

    @Test("Gemma 4 Forward Pass - Shape")
    func testGemma4ForwardPass() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)

        let input = MLXArray(0..<8).reshaped(1, 8)
        let output = model(input, cache: nil)

        #expect(output.shape == [1, 8, model.vocabularySize])
    }

    @Test("Forward Pass Determinism")
    func testDeterminism() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)

        let input = MLXArray(0..<8).reshaped(1, 8)
        let output1 = model(input, cache: nil)
        let output2 = model(input, cache: nil)
        #expect(allClose(output1, output2).item(Bool.self))
    }

    @Test("No NaN/Inf in Output")
    func testNoNaNInf() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)

        let input = MLXArray(Int32(0)..<Int32(5)).reshaped(1, 5)
        let output = model(input, cache: nil)
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    /// Create a minimal test configuration for Gemma 4 Text MoE
    private func makeTinyTextMoEConfigData() -> Data {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "num_key_value_heads": 2,
            "rope_traditional": false,
            "sliding_window": 128,
            "sliding_window_pattern": 1,
            "max_position_embeddings": 512,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true,
            "hidden_size_per_layer_input": 32,
            "vocab_size_per_layer_input": 10,
            "final_logit_softcapping": 30.0,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 128,
            "attention_k_eq_v": false
        }
        """
        return json.data(using: .utf8)!
    }

    @Test("Gemma 4 Text MoE Instantiation & Forward Pass")
    func testGemma4TextMoEInstantiationAndForward() throws {
        let data = makeTinyTextMoEConfigData()
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
        let model = Gemma4TextModel(config)
        #expect(model.vocabularySize == 100)
        
        // This validates that the conditional MoE logic and SwitchGLU layer initialize properly
        // without crashing, proving we correctly load gemma4_text active MoEs.
        let input = MLXArray(0..<8).reshaped(1, 8)
        let output = model(input, cache: nil)

        // Ensure dimensionality is correct
        #expect(output.shape == [1, 8, model.vocabularySize])
        
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    // -------------------------------------------------------------------------
    // MARK: - K-eq-V regression tests (Issue #59)
    //
    // Prior to the fix, attention_k_eq_v=true caused a double-transpose of v:
    //   k → kNorm → transpose → [B, nKvH, L, D]
    //   v = k  (already transposed)
    //   v = vNorm(v).transposed(0,2,1,3)  → [B, L, nKvH, D]  ← WRONG!
    // Leading to: [broadcast_shapes] (1,512,2,512) vs (1,2,512,512)
    // These tests guard against the regression re-entering main.
    // -------------------------------------------------------------------------

    private func makeTinyKeqVConfigData() -> Data {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 64,
            "num_global_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "num_key_value_heads": 2,
            "rope_traditional": false,
            "sliding_window": 8,
            "sliding_window_pattern": 1,
            "max_position_embeddings": 512,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 100,
            "final_logit_softcapping": 30.0,
            "enable_moe_block": false,
            "attention_k_eq_v": true
        }
        """
        return json.data(using: .utf8)!
    }

    private func makeTinyKeqVMoEConfigData() -> Data {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 64,
            "num_global_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "num_key_value_heads": 2,
            "rope_traditional": false,
            "sliding_window": 8,
            "sliding_window_pattern": 1,
            "max_position_embeddings": 512,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 100,
            "final_logit_softcapping": 30.0,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 64,
            "attention_k_eq_v": true
        }
        """
        return json.data(using: .utf8)!
    }

    @Test("K-eq-V Forward Pass — no double-transpose regression (Issue #59)")
    func testGemma4KeqVForwardPass() throws {
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: makeTinyKeqVConfigData())
        let model = Gemma4TextModel(config)
        let input = MLXArray(0..<6).reshaped(1, 6)
        let output = model(input, cache: nil)
        #expect(output.shape == [1, 6, model.vocabularySize])
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    @Test("K-eq-V + MoE Forward Pass — gemma-4-26b-a4b shape regression (Issue #59)")
    func testGemma4KeqVMoEForwardPass() throws {
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: makeTinyKeqVMoEConfigData())
        let model = Gemma4TextModel(config)
        let input = MLXArray(0..<6).reshaped(1, 6)
        let output = model(input, cache: nil)
        #expect(output.shape == [1, 6, model.vocabularySize])
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    // -------------------------------------------------------------------------
    // MARK: - MTP Speculative Decoding Tests
    //
    // Exercises the full Gemma4 two-stage MTP pipeline:
    //   1. Gemma4AssistantModel.callMTP() wired with mainModelRef
    //   2. MTPTokenIterator fallback path (first step, no prior mtpLogits)
    //   3. MTPTokenIterator draft+verify round
    //   4. Greedy determinism and NaN-free output
    //
    // No real weights needed — tiny random-init models validate shape/flow.
    // Design reference: llama.cpp PR #22673 (Qwen3.6 MTP, 72% accept rate)
    // Key insight from llama.cpp: hidden state BEFORE final norm (t_h_pre_norm)
    // must be passed to the MTP head, not the post-norm output.
    // -------------------------------------------------------------------------

    private func makeTinyAssistantConfigData() -> Data {
        // Same dims as makeTinyConfigData() so no projection weight is needed
        let json = """
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "global_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 2,
                "rope_traditional": false,
                "sliding_window": 128,
                "sliding_window_pattern": 1,
                "max_position_embeddings": 512,
                "num_kv_shared_layers": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "hidden_size_per_layer_input": 0,
                "vocab_size_per_layer_input": 100,
                "final_logit_softcapping": 30.0,
                "enable_moe_block": false,
                "attention_k_eq_v": false
            },
            "vocab_size": 100
        }
        """
        return json.data(using: .utf8)!
    }

    @Test("Gemma4 MTP — callMTP returns main logits with correct shape")
    func testGemma4AssistantCallMTPShape() throws {
        let mainCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyConfigData())
        let asstCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyAssistantConfigData())
        let mainModel = Gemma4Model(mainCfg)
        let asstModel = Gemma4AssistantModel(asstCfg)
        asstModel.mainModelRef = mainModel

        let cache = asstModel.newCache(parameters: nil)
        let input = MLXArray(0..<5).reshaped(1, 5)
        let results = asstModel.callMTP(input, cache: cache, mtpCaches: nil)

        #expect(results.count >= 1)
        let mainLogits = results[0]
        #expect(mainLogits.shape == [1, 5, 100])
        let sum = mainLogits.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    @Test("Gemma4 MTP — assistant logits have correct vocab dimension")
    func testGemma4AssistantMTPVocabDim() throws {
        let mainCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyConfigData())
        let asstCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyAssistantConfigData())
        let mainModel = Gemma4Model(mainCfg)
        let asstModel = Gemma4AssistantModel(asstCfg)
        asstModel.mainModelRef = mainModel

        let cache = asstModel.newCache(parameters: nil)
        let input = MLXArray(0..<3).reshaped(1, 3)
        let results = asstModel.callMTP(input, cache: cache, mtpCaches: nil)

        for logits in results {
            #expect(logits.dim(-1) == 100)
        }
    }

    @Test("Gemma4 MTP — MTPTokenIterator fallback produces a valid token")
    func testMTPTokenIteratorFallbackStep() throws {
        let mainCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyConfigData())
        let asstCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyAssistantConfigData())
        let mainModel = Gemma4Model(mainCfg)
        let asstModel = Gemma4AssistantModel(asstCfg)
        asstModel.mainModelRef = mainModel

        let params = GenerateParameters(maxTokens: 3, temperature: 0.0)
        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))

        var iterator = try MTPTokenIterator(
            input: input, model: asstModel, parameters: params, numMTPTokens: 1)

        let token = iterator.next()
        #expect(token != nil)
        if let t = token {
            #expect(t >= 0 && t < 100)
        }
    }

    @Test("Gemma4 MTP — iterator generates exactly maxTokens then stops")
    func testMTPTokenIteratorMaxTokens() throws {
        let mainCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyConfigData())
        let asstCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyAssistantConfigData())
        let mainModel = Gemma4Model(mainCfg)
        let asstModel = Gemma4AssistantModel(asstCfg)
        asstModel.mainModelRef = mainModel

        let params = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let input = LMInput(tokens: MLXArray([1, 2, 3]))

        var iterator = try MTPTokenIterator(
            input: input, model: asstModel, parameters: params, numMTPTokens: 1)

        var tokens = [Int]()
        while let t = iterator.next() { tokens.append(t) }

        #expect(tokens.count == 4)
        for t in tokens { #expect(t >= 0 && t < 100) }
    }

    @Test("Gemma4 MTP — greedy decoding is deterministic")
    func testMTPTokenIteratorDeterminism() throws {
        let mainCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyConfigData())
        let asstCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyAssistantConfigData())
        let mainModel = Gemma4Model(mainCfg)

        let params = GenerateParameters(maxTokens: 5, temperature: 0.0)
        let input = LMInput(tokens: MLXArray([7, 3, 1]))

        func run() throws -> [Int] {
            let asst = Gemma4AssistantModel(asstCfg)
            asst.mainModelRef = mainModel
            var it = try MTPTokenIterator(
                input: input, model: asst, parameters: params, numMTPTokens: 1)
            var out = [Int]()
            while let t = it.next() { out.append(t) }
            return out
        }

        let run1 = try run()
        let run2 = try run()
        #expect(run1 == run2)
    }

    @Test("Gemma4 MTP — no NaN/Inf in generated token stream")
    func testMTPTokenIteratorNoNaNInf() throws {
        let mainCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyConfigData())
        let asstCfg = try JSONDecoder().decode(Gemma4Configuration.self, from: makeTinyAssistantConfigData())
        let mainModel = Gemma4Model(mainCfg)
        let asstModel = Gemma4AssistantModel(asstCfg)
        asstModel.mainModelRef = mainModel

        let params = GenerateParameters(maxTokens: 8, temperature: 0.0)
        let input = LMInput(tokens: MLXArray([1, 5, 9, 2]))

        var iterator = try MTPTokenIterator(
            input: input, model: asstModel, parameters: params, numMTPTokens: 1)

        var count = 0
        while let t = iterator.next() {
            #expect(t >= 0 && t < 100)
            count += 1
        }
        #expect(count == 8)
    }
    }
}


