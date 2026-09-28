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
@testable import MLXLLM
@testable import MLXLMCommon
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
    vocabSize: Int = 100,
    headDim: Int? = nil,
    fullAttentionInterval: Int = 4
) throws -> Qwen35TextConfiguration {
    let headDimLine = headDim.map { "\"head_dim\": \($0)," } ?? ""
    let json = """
    {
        \(headDimLine)
        "model_type": "qwen3_5",
        "hidden_size": \(hiddenSize),
        "num_hidden_layers": \(numHiddenLayers),
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": \(vocabSize),
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "full_attention_interval": \(fullAttentionInterval),
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
        "head_dim": 16,
        "q_lora_rank": 16,
        "kv_lora_rank": 16,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "o_groups": 2,
        "o_lora_rank": 16,
        "sliding_window": 512,
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
        "hc_sinkhorn_iters": 2,
        "hc_eps": 1e-6,
        "moe_intermediate_size": 64,
        "compress_ratios": [1, 1, 1, 1],
        "compress_rope_theta": 10000.0,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "num_hash_layers": 1,
        "norm_topk_prob": false
    }
    """
    return try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
}

/// A hybrid Qwen3.5 trunk with stand-in MTP heads, so drafts are made without the
/// SWIFTLM_MTP_ENABLE weights. Head 0 repeats the main prediction and head 1 shifts
/// it, which gives a mix of accepted and rejected drafts.
private final class HybridDraftingModel: Module, MTPLanguageModel {
    let inner: Qwen35TextModel

    init(_ inner: Qwen35TextModel) {
        self.inner = inner
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try inner.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        inner(input, cache: cache, state: state)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inner(inputs, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try inner.newCache(parameters: parameters)
    }

    func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?) -> [MLXArray] {
        let main = inner(inputs, cache: cache)
        return [main, main, MLX.roll(main, shift: 1, axis: -1)]
    }
}

/// A hybrid trunk whose logits always pick one token, so every draft is accepted.
private final class ConstantDraftingModel: Module, MTPLanguageModel {
    let inner: Qwen35TextModel

    init(_ inner: Qwen35TextModel) {
        self.inner = inner
        super.init()
    }

    private func constant(like logits: MLXArray) -> MLXArray {
        let vocab = logits.dim(-1)
        let row = MLXArray((0 ..< vocab).map { $0 == 7 ? Float(10) : Float(0) })
        return broadcast(row, to: logits.shape)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        LMOutput(logits: callAsFunction(input.tokens, cache: cache))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        constant(like: inner(inputs, cache: cache))
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try inner.newCache(parameters: parameters)
    }

    func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?) -> [MLXArray] {
        let main = callAsFunction(inputs, cache: cache)
        return [main, main, main]
    }
}

/// A trunk whose logits count up: after token t the main head picks t + 1 and MTP head i
/// picks t + 2 + i, so every draft is right unless its head is in `wrongHeads`. Each
/// callMTP writes one row per input token to every MTP cache.
private final class CountingDraftingModel: Module, MTPLanguageModel {
    let inner: Qwen35TextModel
    let numHeads: Int
    let wrongHeads: Set<Int>
    let mtpCacheCount: Int

    init(_ inner: Qwen35TextModel, numHeads: Int, wrongHeads: Set<Int> = [], mtpCacheCount: Int = 0)
    {
        self.inner = inner
        self.numHeads = numHeads
        self.wrongHeads = wrongHeads
        self.mtpCacheCount = mtpCacheCount
        super.init()
    }

    private func counting(_ inputs: MLXArray, _ trunk: MLXArray, shift: Int) -> MLXArray {
        let vocab = trunk.dim(-1)
        let target = (inputs.asType(.int32) + Int32(shift)) % Int32(vocab)
        let ids = MLXArray(Int32(0) ..< Int32(vocab))
        return (ids .== target[.ellipsis, .newAxis]).asType(trunk.dtype) * 10 + 0 * trunk
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        LMOutput(logits: callAsFunction(input.tokens, cache: cache))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        counting(inputs, inner(inputs, cache: cache), shift: 1)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try inner.newCache(parameters: parameters)
    }

    func makeMTPCaches(parameters: GenerateParameters?) -> [[KVCache]] {
        (0 ..< mtpCacheCount).map { _ in [KVCacheSimple()] }
    }

    func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?) -> [MLXArray] {
        let trunk = inner(inputs, cache: cache)
        for mtpCache in mtpCaches ?? [] {
            let rows = MLXArray.zeros([1, 1, inputs.dim(1), 4])
            _ = mtpCache[0].update(keys: rows, values: rows)
        }
        let main = counting(inputs, trunk, shift: 1)
        let heads = (0 ..< numHeads).map { i in
            counting(inputs, trunk, shift: wrongHeads.contains(i) ? 50 : 2 + i)
        }
        return [main] + heads
    }
}

/// The attention offset past the prompt.
private func consumedTokens(_ cache: [KVCache], promptLength: Int) -> Int {
    cache.first { !($0 is MambaCache) }!.offset - promptLength
}

/// A main model and a draft that is a slightly perturbed copy of it, so drafts are
/// partly accepted.
private func makePartlyAgreeingPair(
    seed: UInt64, noise: Float = 0.02, headDim: Int? = nil
) throws -> (Qwen35TextModel, Qwen35TextModel) {
    let config = try makeQwen35TextConfig(headDim: headDim)
    return withRandomState(MLXRandom.RandomState(seed: seed)) {
        let main = Qwen35TextModel(config)
        let draft = Qwen35TextModel(config)
        draft.update(parameters: main.parameters())
        let perturbed = draft.parameters().flattened().map { key, value in
            (key, value + noise * MLXRandom.normal(value.shape).asType(value.dtype))
        }
        draft.update(parameters: ModuleParameters.unflattened(perturbed))
        eval(main, draft)
        return (main, draft)
    }
}

/// A cache fed `prompt` and then each of `tokens` one at a time.
private func referenceCache(
    _ model: Qwen35TextModel, prompt: MLXArray, tokens: some Sequence<Int>,
    parameters: GenerateParameters
) throws -> [KVCache] {
    let cache = try model.newCache(parameters: parameters)
    eval(model(prompt[.newAxis], cache: cache))
    for t in tokens { eval(model(MLXArray([Int32(t)])[.newAxis], cache: cache)) }
    return cache
}

/// `actual` must hold the same attention offsets and recurrent state as `reference`.
private func expectSameCache(
    _ reference: [KVCache], _ actual: [KVCache], _ label: String = "", compareAttention: Bool = true
) {
    for (i, (ref, act)) in zip(reference, actual).enumerated() {
        #expect(ref.offset == act.offset, "\(label) \(type(of: ref)): \(ref.offset) vs \(act.offset)")
        if compareAttention, !(ref is MambaCache) {
            #expect(ref.state.count == act.state.count, "\(label) layer \(i): attention state missing")
            for (a, b) in zip(ref.state, act.state) where a.shape == b.shape {
                let diff = abs(a - b).max().item(Float.self)
                #expect(diff < 1e-4, "\(label) layer \(i): attention max diff \(diff)")
            }
        }
        guard ref is MambaCache else { continue }
        #expect(ref.state.count == act.state.count, "\(label) layer \(i): recurrent state missing")
        for (a, b) in zip(ref.state, act.state) {
            let diff = abs(a - b).max().item(Float.self)
            #expect(diff < 1e-5, "\(label) layer \(i): max diff \(diff), max |ref| \(abs(a).max().item(Float.self))")
        }
    }
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
            #expect(model.model.layers.count == config.numHiddenLayers - config.numNextnPredictLayers,
                    "DeepseekV4Model.layers count should exclude MTP layers when SWIFTLM_MTP_ENABLE is not set")
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

        // 2.7b — Rejected drafts on a hybrid (attention + recurrent) cache. A Mamba layer
        // can't drop only the rejected tail, so the iterator restores its checkpoint and
        // re-feeds the kept tokens. Its cache must equal one fed exactly the tokens it
        // consumed (compared directly: a tiny random model's near-tie argmax can differ
        // between batched verify and single-step decoding).
        @Test("MTPTokenIterator rollback keeps a hybrid cache equal to the tokens it consumed")
        func testMTPIteratorHybridRejectionsMatchGreedy() throws {
            let config = try makeQwen35TextConfig()
            let trunk = withRandomState(MLXRandom.RandomState(seed: 72)) {
                let model = Qwen35TextModel(config)
                eval(model)
                return model
            }
            let model = HybridDraftingModel(trunk)
            let prompt = MLXArray([1, 2, 3, 4, 5, 6])
            let params = GenerateParameters(maxTokens: 16, temperature: 0.0)

            let mtpCache = try trunk.newCache(parameters: params)
            var mtpIter = try MTPTokenIterator(
                input: LMInput(tokens: prompt), model: model, cache: mtpCache,
                parameters: params, numMTPTokens: 2)
            var mtpTokens = [Int]()
            while let t = mtpIter.next() { mtpTokens.append(t) }
            #expect(mtpIter.totalDraftTokens > mtpIter.acceptedDraftTokens)
            #expect(mtpCache.contains { $0 is MambaCache })

            let consumed = mtpCache.first { !($0 is MambaCache) }!.offset - prompt.size
            let refCache = try trunk.newCache(parameters: params)
            eval(trunk(prompt[.newAxis], cache: refCache))
            for t in mtpTokens.prefix(consumed) {
                eval(trunk(MLXArray([Int32(t)])[.newAxis], cache: refCache))
            }
            for (ref, mtp) in zip(refCache, mtpCache) {
                #expect(ref.offset == mtp.offset, "\(type(of: ref)): \(ref.offset) vs \(mtp.offset)")
                guard ref is MambaCache else { continue }
                #expect(ref.state.count == mtp.state.count)
                for (a, b) in zip(ref.state, mtp.state) {
                    #expect(abs(a - b).max().item(Float.self) < 1e-5)
                }
            }
        }

        // 2.7c — After a fully accepted round the recurrent checkpoint must be dropped
        // (trim(0)); a stale one would later restore an old state.
        @Test("MTPTokenIterator drops the recurrent checkpoint after fully accepted rounds")
        func testMTPIteratorClearsCheckpointWhenAllDraftsAccepted() throws {
            let trunk = withRandomState(MLXRandom.RandomState(seed: 74)) {
                let model = Qwen35TextModel(try! makeQwen35TextConfig())
                eval(model)
                return model
            }
            let model = ConstantDraftingModel(trunk)
            let params = GenerateParameters(maxTokens: 12, temperature: 0.0)
            let cache = try trunk.newCache(parameters: params)
            var iter = try MTPTokenIterator(
                input: LMInput(tokens: MLXArray([1, 2, 3, 4])), model: model, cache: cache,
                parameters: params, numMTPTokens: 2)
            while iter.next() != nil {}
            #expect(iter.totalDraftTokens > 0)
            #expect(iter.acceptedDraftTokens == iter.totalDraftTokens)
            let mambas = cache.compactMap { $0 as? MambaCache }
            #expect(!mambas.isEmpty)
            #expect(mambas.allSatisfy { !$0.hasRollbackCheckpoint })
        }

        // 2.7d — Draft-model speculative decoding with hybrid main and draft models, drafts
        // partly accepted: after each round both caches must equal caches fed exactly the
        // tokens they consumed.
        @Test("SpeculativeTokenIterator rollback keeps hybrid main and draft caches exact")
        func testSpeculativeIteratorHybridRollback() throws {
            let (main, draft) = try makePartlyAgreeingPair(seed: 75)
            let prompt = MLXArray([1, 2, 3, 4, 5, 6])
            let params = GenerateParameters(maxTokens: 24, temperature: 0.0)
            let mainCache = try main.newCache(parameters: params)
            let draftCache = try draft.newCache(parameters: params)
            var iter = try SpeculativeTokenIterator(
                input: LMInput(tokens: prompt), mainModel: main, draftModel: draft,
                mainCache: mainCache, draftCache: draftCache, parameters: params,
                numDraftTokens: 3)
            var tokens = [Int]()
            while let t = iter.next() { tokens.append(t) }
            #expect(iter.acceptedDraftTokens > 0)
            #expect(iter.totalDraftTokens > iter.acceptedDraftTokens)
            expectConsumedTokens(main, draft, prompt, tokens, mainCache, draftCache, params)
        }

        /// Both caches equal caches fed exactly the tokens each one consumed.
        private func expectConsumedTokens(
            _ main: Qwen35TextModel, _ draft: Qwen35TextModel, _ prompt: MLXArray,
            _ tokens: [Int], _ mainCache: [KVCache], _ draftCache: [KVCache],
            _ params: GenerateParameters, compareAttention: Bool = true
        ) {
            func consumed(_ cache: [KVCache]) -> Int {
                cache.first { !($0 is MambaCache) }!.offset - prompt.size
            }
            expectSameCache(
                try! referenceCache(
                    main, prompt: prompt, tokens: tokens.prefix(consumed(mainCache)),
                    parameters: params),
                mainCache, "main", compareAttention: compareAttention)
            expectSameCache(
                try! referenceCache(
                    draft, prompt: prompt, tokens: tokens.prefix(consumed(draftCache)),
                    parameters: params),
                draftCache, "draft", compareAttention: compareAttention)
        }

        // 2.7e — A long first-round y (prefill left the prompt unprocessed) with TurboKV,
        // whose caches can only rewind their uncompressed tail. y's head is fed as a
        // plain step, so the rewind never reaches the compressed history.
        @Test("SpeculativeTokenIterator with TurboKV and a long first-round y")
        func testSpeculativeIteratorTurboKVLongFirstRound() throws {
            let (main, draft) = try makePartlyAgreeingPair(seed: 76, headDim: 128)
            let prompt = MLXArray((0 ..< 600).map { Int32($0 % 97 + 1) })
            let params = GenerateParameters(maxTokens: 12, temperature: 0.0, prefillStepSize: 1024)
            let mainCache = try main.newCache(parameters: params)
            for case let simple as KVCacheSimple in mainCache {
                simple.turboQuantEnabled = true
                simple.turboMinActivationTokens = 300
            }
            let draftCache = try draft.newCache(parameters: params)
            for case let simple as KVCacheSimple in draftCache {
                simple.turboQuantEnabled = true
                simple.turboMinActivationTokens = 300
            }
            var iter = try SpeculativeTokenIterator(
                input: LMInput(tokens: prompt), mainModel: main, draftModel: draft,
                mainCache: mainCache, draftCache: draftCache, parameters: params,
                numDraftTokens: 3)
            var tokens = [Int]()
            while let t = iter.next() { tokens.append(t) }
            #expect(tokens.count == 12)
            #expect(iter.totalDraftTokens > iter.acceptedDraftTokens)
            // Compressed attention can't be compared row by row; offsets and recurrent can.
            expectConsumedTokens(
                main, draft, prompt, tokens, mainCache, draftCache, params,
                compareAttention: false)
        }

        // 2.7f — A restored cache whose sliding window has wrapped isn't trimmable at that
        // moment but still rewinds per layer, so init must accept it (a ChatSession turn).
        @Test("SpeculativeTokenIterator accepts a restored cache whose window has wrapped")
        func testSpeculativeIteratorAcceptsWrappedRestoredCache() throws {
            let (main, draft) = try makePartlyAgreeingPair(seed: 77)
            let params = GenerateParameters(maxTokens: 6, maxKVSize: 8, temperature: 0.0)
            let history = MLXArray((0 ..< 12).map { Int32($0 + 1) })
            let mainCache = try main.newCache(parameters: params)
            let draftCache = try draft.newCache(parameters: params)
            eval(main(history[.newAxis], cache: mainCache), draft(history[.newAxis], cache: draftCache))
            #expect(mainCache.contains { $0 is RotatingKVCache && !$0.isTrimmable })

            var iter = try SpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray([Int32(3)])), mainModel: main, draftModel: draft,
                mainCache: mainCache, draftCache: draftCache, parameters: params,
                numDraftTokens: 2)
            var count = 0
            while iter.next() != nil { count += 1 }
            #expect(count == 6)
        }

        // 2.7g — Stopping early leaves accepted drafts committed but unemitted. Finalize
        // must take them out of a hybrid cache's recurrent state too, not just attention,
        // keep the storage timeline in step, and do nothing when called again.
        @Test("SpeculativeTokenIterator finalize rewinds unemitted tokens from a hybrid cache")
        func testSpeculativeIteratorHybridFinalize() throws {
            let (main, draft) = try makePartlyAgreeingPair(seed: 75)
            let prompt = MLXArray([1, 2, 3, 4, 5, 6])
            let params = GenerateParameters(maxTokens: 24, temperature: 0.0)
            var maxLookahead = 0
            var maxDraftLookahead = 0
            for stop in 2 ... 12 {
                let mainCache = try main.newCache(parameters: params)
                let draftCache = try draft.newCache(parameters: params)
                var iter = try SpeculativeTokenIterator(
                    input: LMInput(tokens: prompt), mainModel: main, draftModel: draft,
                    mainCache: mainCache, draftCache: draftCache, parameters: params,
                    numDraftTokens: 3)
                var tokens = [Int]()
                while tokens.count < stop, let t = iter.next() { tokens.append(t) }
                let consumed = { consumedTokens(mainCache, promptLength: prompt.size) }
                let draftConsumed = { consumedTokens(draftCache, promptLength: prompt.size) }
                let before = consumed()
                maxLookahead = Swift.max(maxLookahead, before - tokens.count)
                maxDraftLookahead = Swift.max(maxDraftLookahead, draftConsumed() - tokens.count)
                iter.finalizeGeneration()
                #expect(consumed() == Swift.min(before, tokens.count), "stop \(stop)")
                #expect(draftConsumed() <= tokens.count, "stop \(stop): draft")
                #expect(iter.mainCacheStorage.processedTokenCount == consumed() + prompt.size)
                #expect(iter.draftCacheStorage.processedTokenCount == draftConsumed() + prompt.size)
                expectSameCache(
                    try referenceCache(
                        main, prompt: prompt, tokens: tokens.prefix(consumed()), parameters: params),
                    mainCache, "main stop \(stop)")
                expectSameCache(
                    try referenceCache(
                        draft, prompt: prompt, tokens: tokens.prefix(draftConsumed()),
                        parameters: params),
                    draftCache, "draft stop \(stop)")

                let after = (consumed(), draftConsumed())
                let processed = (
                    iter.mainCacheStorage.processedTokenCount,
                    iter.draftCacheStorage.processedTokenCount
                )
                iter.finalizeGeneration()
                #expect(consumed() == after.0, "a second finalize must not trim main again")
                #expect(
                    draftConsumed() == after.1, "a second finalize must not trim the draft again")
                #expect(iter.mainCacheStorage.processedTokenCount == processed.0)
                #expect(iter.draftCacheStorage.processedTokenCount == processed.1)
            }
            // Main can hold two unemitted tokens; the draft, one behind, holds one.
            #expect(maxLookahead >= 2)
            #expect(maxDraftLookahead >= 1)
        }

        // 2.7n — A hybrid draft rewinds a whole round of single-token writes, so its
        // sliding window must hold numDraftTokens + 1 rows past the 4 pinned ones.
        @Test("SpeculativeTokenIterator rejects a hybrid draft window too small to rewind")
        func testSpeculativeIteratorRejectsSmallHybridDraftWindow() throws {
            let (main, draft) = try makePartlyAgreeingPair(seed: 75)
            let prompt = MLXArray([1, 2, 3, 4, 5, 6])
            let numDraft = 3
            func run(maxKVSize: Int) throws -> [Int] {
                let params = GenerateParameters(
                    maxTokens: 24, maxKVSize: maxKVSize, temperature: 0.0)
                var iter = try SpeculativeTokenIterator(
                    input: LMInput(tokens: prompt), mainModel: main, draftModel: draft,
                    parameters: params, numDraftTokens: numDraft)
                var tokens = [Int]()
                while let t = iter.next() { tokens.append(t) }
                return tokens
            }
            #expect(throws: (any Error).self) { try run(maxKVSize: numDraft + 4) }
            #expect(try run(maxKVSize: numDraft + 5).count == 24)
        }

        // 2.7h — MTP: stopping early leaves verified drafts in the cache; finalize must
        // rewind them, from the recurrent state too.
        @Test("MTPTokenIterator finalize rewinds unemitted drafts from a hybrid cache")
        func testMTPIteratorHybridFinalize() throws {
            let trunk = withRandomState(MLXRandom.RandomState(seed: 78)) {
                let model = Qwen35TextModel(try! makeQwen35TextConfig())
                eval(model)
                return model
            }
            let model = ConstantDraftingModel(trunk)
            let prompt = MLXArray([1, 2, 3, 4])
            let params = GenerateParameters(maxTokens: 16, temperature: 0.0)
            var sawLookahead = false
            for stop in 2 ... 8 {
                let cache = try trunk.newCache(parameters: params)
                var iter = try MTPTokenIterator(
                    input: LMInput(tokens: prompt), model: model, cache: cache,
                    parameters: params, numMTPTokens: 2)
                var tokens = [Int]()
                while tokens.count < stop, let t = iter.next() { tokens.append(t) }
                let consumed = { cache.first { !($0 is MambaCache) }!.offset - prompt.size }
                let before = consumed()
                if before > tokens.count { sawLookahead = true }
                iter.finalizeGeneration()
                #expect(consumed() == Swift.min(before, tokens.count), "stop \(stop)")
                expectSameCache(
                    try referenceCache(
                        trunk, prompt: prompt, tokens: tokens.prefix(consumed()), parameters: params),
                    cache, "mtp stop \(stop)")
            }
            #expect(sawLookahead)
        }

        /// Stops MTP generation after each of `stops` tokens, finalizes, and checks the cache
        /// holds exactly the tokens emitted.
        private func expectMTPFinalizeExact(
            _ trunk: Qwen35TextModel, _ model: CountingDraftingModel, stops: ClosedRange<Int>,
            minLookahead: Int = 1
        ) throws {
            let prompt = MLXArray([1, 2, 3, 4])
            let params = GenerateParameters(maxTokens: 24, temperature: 0.0)
            var maxLookahead = 0
            for stop in stops {
                let cache = try trunk.newCache(parameters: params)
                var iter = try MTPTokenIterator(
                    input: LMInput(tokens: prompt), model: model, cache: cache,
                    parameters: params, numMTPTokens: model.numHeads)
                var tokens = [Int]()
                while tokens.count < stop, let t = iter.next() { tokens.append(t) }
                let before = consumedTokens(cache, promptLength: prompt.size)
                maxLookahead = Swift.max(maxLookahead, before - tokens.count)
                iter.finalizeGeneration()
                let after = consumedTokens(cache, promptLength: prompt.size)
                #expect(after == Swift.min(before, tokens.count), "stop \(stop)")
                for mtpCache in iter.mtpCaches {
                    #expect(mtpCache[0].offset == after + prompt.size, "stop \(stop): MTP cache")
                }
                expectSameCache(
                    try referenceCache(
                        trunk, prompt: prompt, tokens: tokens.prefix(after), parameters: params),
                    cache, "stop \(stop)")
            }
            #expect(maxLookahead >= minLookahead)
        }

        private func makeTrunk(seed: UInt64, fullAttentionInterval: Int = 4) -> Qwen35TextModel {
            withRandomState(MLXRandom.RandomState(seed: seed)) {
                let model = Qwen35TextModel(
                    try! makeQwen35TextConfig(fullAttentionInterval: fullAttentionInterval))
                eval(model)
                return model
            }
        }

        // 2.7i — The last head is always wrong, so each verify round accepts two drafts,
        // rolls back and re-feeds, and the next round falls back to a plain step. Stopping
        // mid-round then leaves lookahead from a partly accepted round.
        @Test("MTPTokenIterator finalize after partly accepted and fallback rounds")
        func testMTPIteratorFinalizeAfterPartialAndFallbackRounds() throws {
            let trunk = makeTrunk(seed: 79)
            let model = CountingDraftingModel(trunk, numHeads: 3, wrongHeads: [2], mtpCacheCount: 2)
            try expectMTPFinalizeExact(trunk, model, stops: 2 ... 10)
        }

        // 2.7j — Every draft accepted: finalize can drop two drafts at once, and must trim
        // the MTP caches with the main cache.
        @Test("MTPTokenIterator finalize drops several drafts and trims the MTP caches")
        func testMTPIteratorFinalizeDeepLookahead() throws {
            let trunk = makeTrunk(seed: 80)
            let model = CountingDraftingModel(trunk, numHeads: 3, mtpCacheCount: 2)
            try expectMTPFinalizeExact(trunk, model, stops: 2 ... 10, minLookahead: 2)
        }

        // 2.7k — A pure-attention trunk has no recurrent round start; finalize trims only.
        @Test("MTPTokenIterator finalize on a non-hybrid cache")
        func testMTPIteratorFinalizeNonHybrid() throws {
            let trunk = makeTrunk(seed: 81, fullAttentionInterval: 1)
            #expect(try trunk.newCache(parameters: nil).allSatisfy { !($0 is MambaCache) })
            let model = CountingDraftingModel(trunk, numHeads: 3, mtpCacheCount: 1)
            try expectMTPFinalizeExact(trunk, model, stops: 2 ... 8, minLookahead: 2)
        }

        // 2.7l — A round that returns before verifying commits nothing, so a later finalize
        // must not rewind the previous round's drafts again.
        @Test("MTPTokenIterator round without output leaves nothing to finalize")
        func testMTPIteratorEmptyRoundCommitsNothing() throws {
            final class FailingAfter: Module, MTPLanguageModel {
                let wrapped: CountingDraftingModel
                var calls = 0
                let limit: Int
                init(_ wrapped: CountingDraftingModel, limit: Int) {
                    self.wrapped = wrapped
                    self.limit = limit
                    super.init()
                }
                func prepare(
                    _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
                    prefill: PrefillParameters
                ) throws -> PrepareResult {
                    try wrapped.prepare(input, cache: cache, state: state, prefill: prefill)
                }
                func callAsFunction(
                    _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
                ) -> LMOutput {
                    wrapped(input, cache: cache, state: state)
                }
                func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
                    wrapped(inputs, cache: cache)
                }
                func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
                    try wrapped.newCache(parameters: parameters)
                }
                func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?)
                    -> [MLXArray]
                {
                    calls += 1
                    return calls > limit
                        ? [] : wrapped.callMTP(inputs, cache: cache, mtpCaches: mtpCaches)
                }
            }
            let trunk = makeTrunk(seed: 82)
            let model = FailingAfter(CountingDraftingModel(trunk, numHeads: 2), limit: 2)
            let prompt = MLXArray([1, 2, 3, 4])
            let params = GenerateParameters(maxTokens: 24, temperature: 0.0)
            let cache = try trunk.newCache(parameters: params)
            var iter = try MTPTokenIterator(
                input: LMInput(tokens: prompt), model: model, cache: cache,
                parameters: params, numMTPTokens: 2)
            var tokens = [Int]()
            while let t = iter.next() { tokens.append(t) }
            // Fallback round (1 token), then a verify round with both drafts (3 tokens).
            #expect(tokens.count == 4)
            let before = consumedTokens(cache, promptLength: prompt.size)
            iter.finalizeGeneration()
            #expect(consumedTokens(cache, promptLength: prompt.size) == before)
        }

        // 2.7m — generateMTP stops on an EOS token in the middle of a verified round; the
        // loop's finalize must leave the caller's cache holding only the emitted tokens.
        @Test("generateMTP finalizes the cache when a stop token ends generation early")
        func testGenerateMTPFinalizesOnStopToken() async throws {
            let trunk = makeTrunk(seed: 83)
            let model = CountingDraftingModel(trunk, numHeads: 3)
            let prompt = MLXArray([1, 2, 3, 4])
            // Fallback emits 5; the next round verifies 6, 7, 8 and ends with 9. Stopping
            // at 6 leaves 7 and 8 committed but unemitted.
            let stop = 6
            let processor = TestInputProcessor()
            let context = ModelContext(
                configuration: ModelConfiguration(id: "test", eosTokenIds: [stop]),
                model: model, processor: processor, tokenizer: processor.tokenizer)
            let params = GenerateParameters(maxTokens: 24, temperature: 0.0)
            let cache = try trunk.newCache(parameters: params)
            var sawInfo = false
            for await generation in try generateMTP(
                input: LMInput(tokens: prompt), cache: cache, parameters: params,
                context: context, numMTPTokens: 3)
            {
                if case .info = generation { sawInfo = true }
            }
            #expect(sawInfo)
            #expect(consumedTokens(cache, promptLength: prompt.size) == 2)
            expectSameCache(
                try referenceCache(trunk, prompt: prompt, tokens: [5, stop], parameters: params),
                cache, "generateMTP")
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

            let cache = try model.newCache(parameters: params)
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
