// MTPSpeculativeDecodingTests.swift
// Unit tests for Phase 1 and Phase 2 of MTP Speculative Decoding.
//
// Phase 1: MTPConfig gating, MTPLanguageModel protocol structural checks
// Phase 2: Qwen35TextConfiguration MTP field, callMTP output shape & correctness,
//          MTPTokenIterator end-to-end, generateMTP graceful fallback
//
// All tests run model-free (tiny synthetic configs) and download nothing.
// Design follows the existing SpeculativeDecodingTests / Qwen35Tests patterns.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// MARK: - Tiny model factory

/// Builds a minimal Qwen35TextConfiguration that can be instantiated without
/// downloading weights.  Dimension sizes are kept tiny (64-D) so that
/// forward-pass tests run in milliseconds.
private func makeQwen35TextConfig(
    numMTPLayers: Int = 0,
    numHiddenLayers: Int = 4,
    hiddenSize: Int = 64,
    vocabSize: Int = 100
) throws -> Qwen35TextConfiguration {
    let json = """
    {
        "model_type": "qwen3_5",
        "hidden_size": \(hiddenSize),
        "num_hidden_layers": \(numHiddenLayers),
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": \(vocabSize),
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "full_attention_interval": 4,
        "num_nextn_predict_layers": \(numMTPLayers)
    }
    """
    return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
}

/// Builds a minimal DeepseekV4Configuration
private func makeDeepseekV4Config(
    numMTPLayers: Int = 0,
    numHiddenLayers: Int = 4,
    hiddenSize: Int = 64,
    vocabSize: Int = 100
) throws -> DeepseekV4Configuration {
    let json = """
    {
        "model_type": "deepseek_v4",
        "hidden_size": \(hiddenSize),
        "num_hidden_layers": \(numHiddenLayers),
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": \(vocabSize),
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "num_nextn_predict_layers": \(numMTPLayers),
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "hc_mult": 2,
        "moe_intermediate_size": 64
    }
    """
    return try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
}

// MARK: - Phase 1: MTPConfig & protocol

extension MLXTestingSuite {
    @Suite
    struct MTPPhase1ConfigTests {

        // 1.1 — SWIFTLM_MTP_ENABLE env var gate
        @Test("MTPConfig.retainMTPWeights reflects SWIFTLM_MTP_ENABLE env var")
        func testRetainMTPWeightsEnvGate() {
            let envSet = ProcessInfo.processInfo.environment["SWIFTLM_MTP_ENABLE"] == "1"
            // In CI the env var is never set, so retainMTPWeights should be false.
            // If someone runs with the env var, the value should flip to true.
            if envSet {
                #expect(MTPConfig.retainMTPWeights == true)
            } else {
                #expect(MTPConfig.retainMTPWeights == false)
            }
        }

        // 1.2 — Compile-time protocol hierarchy check
        @Test("MTPLanguageModel is a refinement of LanguageModel (type system check)")
        func testMTPProtocolIsSubprotocol() throws {
            // We verify the protocol hierarchy is correct by checking that
            // Qwen35TextModel (an MTPLanguageModel) satisfies LanguageModel.
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)

            // This assignment only compiles if MTPLanguageModel refines LanguageModel.
            let _: any LanguageModel = model
            let _: any MTPLanguageModel = model
            // If we reach here, the protocol hierarchy is correct.
            #expect(Bool(true))
        }

        // 1.3 — Qwen35TextConfiguration decodes num_nextn_predict_layers
        @Test("Qwen35TextConfiguration decodes num_nextn_predict_layers correctly")
        func testConfigDecodesNumNextnPredictLayers() throws {
            let configWith3 = try makeQwen35TextConfig(numMTPLayers: 3)
            #expect(configWith3.numNextnPredictLayers == 3)

            let configWith0 = try makeQwen35TextConfig(numMTPLayers: 0)
            #expect(configWith0.numNextnPredictLayers == 0)
        }

        // 1.4 — mtp array respects the SWIFTLM_MTP_ENABLE gate
        @Test("Qwen35TextModel.mtp array is empty when MTP env var is unset")
        func testMTPArrayEmptyWithoutEnvVar() throws {
            guard ProcessInfo.processInfo.environment["SWIFTLM_MTP_ENABLE"] != "1" else {
                return  // env var is set — skip this guard check
            }
            // Even if the config declares numNextnPredictLayers = 2,
            // the array should be empty when the env var is not set.
            let config = try makeQwen35TextConfig(numMTPLayers: 2)
            let model = Qwen35TextModel(config)
            #expect(model.mtp.isEmpty,
                    "mtp array must be empty when SWIFTLM_MTP_ENABLE is not set")
        }
    }
}

// MARK: - Phase 2: callMTP output correctness

extension MLXTestingSuite {
    @Suite
    struct MTPPhase2ConformanceTests {

        // 2.1 — callMTP without MTP heads returns exactly main logits
        @Test("callMTP with no MTP heads returns [main_logits] (fallback)")
        func testCallMTPFallbackReturnsSingleTensor() throws {
            let vocabSize = 100
            let config = try makeQwen35TextConfig(numMTPLayers: 0, vocabSize: vocabSize)
            let model = Qwen35TextModel(config)

            let inputs = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
            let results = model.callMTP(inputs, cache: nil)
            eval(results[0])

            #expect(results.count == 1, "Expected exactly 1 tensor (no MTP heads)")
            let logits = results[0]
            #expect(logits.shape[0] == 1, "Batch dimension must be 1")
            #expect(logits.shape[1] == 4, "Sequence dimension must match input length")
            #expect(logits.shape[2] == vocabSize, "Vocab dimension must match config")
        }

        // 2.2 — callMTP main logits match direct callAsFunction (determinism)
        @Test("callMTP main logits match callAsFunction exactly")
        func testCallMTPMainLogitsMatchCallAsFunction() throws {
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)

            let inputs = MLXArray([1, 2, 3, 4]).reshaped(1, 4)

            // Run both paths
            let directLogits = model(inputs, cache: nil)
            let mtpResults = model.callMTP(inputs, cache: nil)
            eval(directLogits, mtpResults[0])

            // Both should produce identical results (same graph, no randomness)
            let maxAbsDiff = (directLogits - mtpResults[0]).abs().max(keepDims: false)
                .item(Float.self)
            #expect(maxAbsDiff < 1e-4,
                    "callMTP main logits must be bit-identical to callAsFunction logits, diff=\(maxAbsDiff)")
        }

        // 2.3 — callMTP logit shape with batch size > 1
        @Test("callMTP produces correct logit shapes for B=2 S=6")
        func testCallMTPShapeMultiBatch() throws {
            let vocabSize = 100
            let config = try makeQwen35TextConfig(vocabSize: vocabSize)
            let model = Qwen35TextModel(config)

            let B = 2
            let S = 6
            // Create a 2D input [B, S] filled with token id 1
            let inputs = MLXArray(Array(repeating: 1, count: B * S)).reshaped(B, S)
            let results = model.callMTP(inputs, cache: nil)
            eval(results[0])

            let logits = results[0]
            #expect(logits.ndim == 3)
            #expect(logits.shape[0] == B)
            #expect(logits.shape[1] == S)
            #expect(logits.shape[2] == vocabSize)
        }

        // 2.4 — Qwen35TextModel conforms to MTPLanguageModel at runtime
        @Test("Qwen35TextModel dynamically casts to MTPLanguageModel")
        func testQwen35TextModelConformsAtRuntime() throws {
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)

            // Upcast to erasure type that InferenceEngine actually casts against
            let asLanguageModel: any LanguageModel = model
            let castedOpt = asLanguageModel as? (any MTPLanguageModel)
            #expect(castedOpt != nil, "Qwen35TextModel must satisfy MTPLanguageModel at runtime")
        }

        // 2.5 — DeepseekV4Model MTP array conditionally allocated
        @Test("DeepseekV4Model.mtpLayers is empty without MTP env var")
        func testDeepseekMTPArrayEmptyWithoutEnvVar() throws {
            guard ProcessInfo.processInfo.environment["SWIFTLM_MTP_ENABLE"] != "1" else {
                return
            }
            let config = try makeDeepseekV4Config(numMTPLayers: 2)
            let model = DeepseekV4Model(config)
            #expect(model.model.mtpLayers.isEmpty,
                    "DeepseekV4Model.mtpLayers must be empty when SWIFTLM_MTP_ENABLE is not set")
        }

        // 2.6 — DeepseekV4 callMTP fallback returns single tensor
        @Test("DeepseekV4 callMTP with no heads returns exactly main logits")
        func testDeepseekCallMTPFallback() throws {
            let vocabSize = 100
            let config = try makeDeepseekV4Config(numMTPLayers: 0, vocabSize: vocabSize)
            let model = DeepseekV4Model(config)

            let inputs = MLXArray([1, 2]).reshaped(1, 2)
            let results = model.callMTP(inputs, cache: nil as [KVCache]?)

            #expect(results.count == 1, "Expected exactly 1 tensor")
            let logits = results[0]
            #expect(logits.shape[0] == 1)
            #expect(logits.shape[1] == 2)
            #expect(logits.shape[2] == vocabSize)
        }
    }
}

// MARK: - Phase 2: MTPTokenIterator end-to-end

extension MLXTestingSuite {
    @Suite
    struct MTPPhase2IteratorTests {

        // 2.5 — MTPTokenIterator initialises (no cache trimming requirement failure)
        // Note: MTPTokenIterator requires canTrimPromptCache. With a Qwen35 model
        // (KVCacheSimple + MambaCache), the default cache IS trimmable.
        @Test("MTPTokenIterator initialises without throwing for Qwen35TextModel")
        func testMTPIteratorInit() throws {
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)
            let input = LMInput(tokens: MLXArray([1, 2, 3]))
            let params = GenerateParameters(maxTokens: 4, temperature: 0.0)

            // Should not throw
            let _ = try MTPTokenIterator(
                input: input,
                model: model,
                parameters: params,
                numMTPTokens: 1
            )
        }

        // 2.6 — MTPTokenIterator respects maxTokens exactly
        @Test("MTPTokenIterator produces exactly maxTokens tokens")
        func testMTPIteratorExactTokenCount() throws {
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)
            let maxTokens = 8
            let input = LMInput(tokens: MLXArray([1, 2, 3]))
            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)

            var iter = try MTPTokenIterator(
                input: input,
                model: model,
                parameters: params,
                numMTPTokens: 1
            )
            var count = 0
            while let _ = iter.next() { count += 1 }
            #expect(count == maxTokens,
                    "Expected exactly \(maxTokens) tokens, got \(count)")
        }

        // 2.7 — At temperature 0, MTPTokenIterator must equal standard TokenIterator
        //
        // This is the critical correctness guarantee from the MTPLX analysis:
        // "Probability-ratio acceptance with residual correction" must collapse to
        // identity (all accepted) at temperature 0 since draft and main distributions
        // are identical (same model head).
        @Test("MTPTokenIterator at temperature=0 matches TokenIterator output")
        func testMTPIteratorGreedyEqualsStandard() throws {
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)
            let maxTokens = 10
            let promptTokens = MLXArray([1, 2, 3, 4])
            let input = LMInput(tokens: promptTokens)
            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)

            // Standard iterator
            var stdIter = try TokenIterator(input: input, model: model, parameters: params)
            var standardTokens = [Int]()
            while let t = stdIter.next() { standardTokens.append(t) }

            // MTP iterator (depth=1, greedy)
            var mtpIter = try MTPTokenIterator(
                input: input,
                model: model,
                parameters: params,
                numMTPTokens: 1
            )
            var mtpTokens = [Int]()
            while let t = mtpIter.next() { mtpTokens.append(t) }

            #expect(!standardTokens.isEmpty)
            #expect(!mtpTokens.isEmpty)
            // At temperature 0, every draft should be accepted — output must be identical
            #expect(standardTokens == mtpTokens,
                    "MTPTokenIterator at temperature=0 must produce identical output to standard TokenIterator")
        }

        // 2.8 — maxTokens is respected even with deep drafting (numMTPTokens=3)
        @Test("MTPTokenIterator respects maxTokens with deep draft depth")
        func testMTPIteratorMaxTokensWithDeepDraft() throws {
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)
            let maxTokens = 5
            let input = LMInput(tokens: MLXArray([1, 2]))
            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)

            var iter = try MTPTokenIterator(
                input: input,
                model: model,
                parameters: params,
                numMTPTokens: 3  // draft 3 at a time
            )
            var count = 0
            while let _ = iter.next() { count += 1 }
            #expect(count == maxTokens,
                    "Must emit exactly maxTokens=\(maxTokens) even when drafting 3 at a time; got \(count)")
        }

        // 2.9 — KV cache offset advances after MTPTokenIterator run
        @Test("KVCache offset advances after MTPTokenIterator completes")
        func testMTPIteratorCacheAdvances() throws {
            let config = try makeQwen35TextConfig()
            let model = Qwen35TextModel(config)
            let maxTokens = 6
            let input = LMInput(tokens: MLXArray([1, 2, 3]))
            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)

            let cache = model.newCache(parameters: params)
            var iter = try MTPTokenIterator(
                input: input,
                model: model,
                cache: cache,
                parameters: params,
                numMTPTokens: 1
            )
            while let _ = iter.next() {}

            // At least one layer must have advanced its cache offset
            let advanced = cache.filter { $0.offset > 0 }
            #expect(!advanced.isEmpty,
                    "At least one KVCache layer must have offset > 0 after generation")
        }

        // 2.10 — generateMTP gracefully handles an MTPLanguageModel with no heads
        @Test("generateMTP produces tokens even when MTP heads are absent (fallback path)")
        func testGenerateMTPFallbackWithNoHeads() async throws {
            let config = try makeQwen35TextConfig(numMTPLayers: 0)
            let model = Qwen35TextModel(config)
            let processor = TestInputProcessor()
            let ctx = ModelContext(
                configuration: processor.configuration,
                model: model,
                processor: processor,
                tokenizer: processor.tokenizer
            )
            let input = LMInput(tokens: MLXArray([1, 2]))
            let params = GenerateParameters(maxTokens: 4, temperature: 0.0)

            var tokenCount = 0
            for await generation in try generateMTP(
                input: input,
                parameters: params,
                context: ctx,
                numMTPTokens: 1
            ) {
                if case .chunk(_, _) = generation { tokenCount += 1 }
            }
            #expect(tokenCount > 0,
                    "generateMTP must produce output tokens even with no MTP heads")
        }
    }
}
