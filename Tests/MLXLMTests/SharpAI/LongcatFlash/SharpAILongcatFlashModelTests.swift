// SharpAI fork overlay tests for longcat_flash model structure, cache and forward pass.
// Not part of upstream. These tests were written without running them.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILongcatFlashModelTests: XCTestCase {

    private func shapeOf(_ model: SharpAILongcatFlashModel, _ key: String) -> [Int]? {
        model.parameters().flattened().first { $0.0 == key }?.1.shape
    }

    func testParameterNamesMatchCheckpointLayout() throws {
        let model = try SharpAILongcatFlashTestFixtures.model()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.ngram_embeddings.word_embeddings.weight"))
        XCTAssertTrue(keys.contains("model.ngram_embeddings.embedders.0.weight"))
        XCTAssertTrue(keys.contains("model.ngram_embeddings.post_projs.3.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.1.kv_b_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.0.q_a_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlps.1.down_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.input_layernorm.0.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.post_attention_layernorm.1.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.router.classifier.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.router.e_score_correction_bias"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.norm.weight"))
        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testRouterClassifierCoversRoutedPlusZeroExperts() throws {
        let model = try SharpAILongcatFlashTestFixtures.model()

        // 4 routed experts + 2 zero experts.
        XCTAssertEqual(
            shapeOf(model, "model.layers.0.mlp.router.classifier.weight"), [6, 16])
        XCTAssertEqual(
            shapeOf(model, "model.layers.0.mlp.switch_mlp.gate_proj.weight"), [4, 8, 16])
    }

    func testVocabModsFollowMlxLmRecurrence() throws {
        let model = try SharpAILongcatFlashTestFixtures.model()
        let ngram = model.model.ngramEmbeddings

        // Index 0 is order 2 (dim 129): [32 mod 129].
        XCTAssertEqual(ngram.vocabMods[0], [32])
        // Index 2 is order 3 (dim 128 + 4 + 1 = 133): [32, 32 * 32 mod 133] = [32, 93].
        XCTAssertEqual(ngram.vocabMods[2], [32, 93])
        XCTAssertEqual(ngram.embedders.count, 4)
        XCTAssertEqual(ngram.embedders[0].weight.shape, [129, 4])
    }

    func testNewCacheHasNgramCacheThenOneCacheListPerLayer() throws {
        let model = try SharpAILongcatFlashTestFixtures.model(numLayers: 2)
        let cache = try model.newCache(parameters: nil)

        XCTAssertEqual(cache.count, 3)
        XCTAssertTrue(cache[0] is ArraysCache)
        XCTAssertTrue(cache[1] is CacheList)
        XCTAssertTrue(cache[2] is CacheList)
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try SharpAILongcatFlashTestFixtures.model()
        let inputs = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 5, 32])
    }

    func testForwardPassAdvancesEveryAttentionCacheOnce() throws {
        let model = try SharpAILongcatFlashTestFixtures.model()
        let cache = try model.newCache(parameters: nil)
        let inputs = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let logits = model(inputs, cache: cache)
        eval(logits)

        for entry in cache.dropFirst() {
            let list = try XCTUnwrap(entry as? CacheList)
            XCTAssertEqual(list[0].offset, 5)
            XCTAssertEqual(list[1].offset, 5)
        }
    }

    func testNgramHistoryKeepsLastNMinusOneTokens() throws {
        let model = try SharpAILongcatFlashTestFixtures.model()
        let cache = try model.newCache(parameters: nil)
        let inputs = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let logits = model(inputs, cache: cache)
        eval(logits)

        // emb_neighbor_num = 3, so the history holds the last two token ids.
        let history = try XCTUnwrap(cache[0] as? ArraysCache)
        let stored = try XCTUnwrap(history[0])
        XCTAssertEqual(stored.shape, [1, 2])
        XCTAssertEqual(stored.asArray(Int.self), [4, 5])
    }

    func testIncrementalDecodeMatchesFullForward() throws {
        // UNVERIFIED: tolerance chosen by hand. Not run. Checks that the n-gram history
        // cache gives the same last-position logits as a single full-sequence pass.
        let model = try SharpAILongcatFlashTestFixtures.model()
        let tokens = MLXArray([1, 2, 3, 4] as [Int32]).reshaped(1, 4)
        let full = model(tokens, cache: nil)

        let cache = try model.newCache(parameters: nil)
        let prefix = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        _ = model(prefix, cache: cache)
        let last = MLXArray([4] as [Int32]).reshaped(1, 1)
        let step = model(last, cache: cache)
        eval(full, step)

        let expected = full[0 ..< 1, 3 ..< 4, 0...]
        XCTAssertTrue(
            allClose(step, expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    func testRegistrationCreatesLongcatFlashModelFromJSON() throws {
        let creator = try XCTUnwrap(SharpAILongcatFlashOverlay.creators["longcat_flash"])
        let model = try creator(Data(SharpAILongcatFlashTestFixtures.json().utf8))

        XCTAssertTrue(model is SharpAILongcatFlashModel)
    }
}
