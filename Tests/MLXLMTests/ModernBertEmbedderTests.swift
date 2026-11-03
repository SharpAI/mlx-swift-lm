// SL-67: the `modernbert` embedder is reachable through the real EmbedderTypeRegistry.
//
// Air-gapped: random-init tiny config, no checkpoint, no network. "Registered and runs" only:
// these tests do NOT claim numeric parity with the HF ModernBERT reference.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXEmbedders

final class ModernBertEmbedderTests: XCTestCase {

    /// 4 layers: 0 and 3 global, 1 and 2 local with a window of 4.
    private static let tinyConfig = """
        {
            "model_type": "modernbert",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 4,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "norm_eps": 1e-5,
            "local_attention": 4,
            "global_attn_every_n_layers": 3,
            "max_position_embeddings": 64
        }
        """

    private func makeModel(_ json: String = tinyConfig) async throws -> any EmbeddingModel {
        try await EmbedderTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), modelType: "modernbert")
    }

    func testFactoryCreatesModernBertEncoder() async throws {
        let model = try await makeModel()
        XCTAssertEqual(model.vocabularySize, 32)
        XCTAssertEqual(model.maxPositionEmbeddings, 64)
    }

    func testForwardShapeAndFinite() async throws {
        let model = try await makeModel()
        let inputs = MLXArray([1, 2, 3, 4, 5, 6, 7] as [Int32]).reshaped(1, 7)
        let out = model(inputs, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
        let hidden = try XCTUnwrap(out.hiddenStates)
        eval(hidden)
        XCTAssertEqual(hidden.shape, [1, 7, 8])
        XCTAssertTrue(all(isFinite(hidden)).item(Bool.self))
    }

    func testPooledOutputIsNormalised() async throws {
        let model = try await makeModel()
        let inputs = MLXArray([1, 2, 3, 4, 5, 6, 7, 8] as [Int32]).reshaped(2, 4)
        let mask = MLXArray.ones([2, 4], dtype: .int32)
        let out = model(inputs, positionIds: nil, tokenTypeIds: nil, attentionMask: mask)
        let strategy = try XCTUnwrap(model.poolingStrategy)
        let pooled = Pooling(strategy: strategy)(out, mask: mask, normalize: true)
        eval(pooled)
        XCTAssertEqual(pooled.shape, [2, 8])
        let norms = sqrt(sum(pooled * pooled, axis: -1))
        eval(norms)
        for n in norms.asArray(Float.self) { XCTAssertEqual(n, 1.0, accuracy: 1e-4) }
    }

    /// Attention is bidirectional: changing the last token must move position 0.
    func testAttentionIsBidirectional() async throws {
        let model = try await makeModel()
        let a = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let b = MLXArray([1, 2, 7] as [Int32]).reshaped(1, 3)
        let ha = try XCTUnwrap(model(a, positionIds: nil, tokenTypeIds: nil, attentionMask: nil).hiddenStates)
        let hb = try XCTUnwrap(model(b, positionIds: nil, tokenTypeIds: nil, attentionMask: nil).hiddenStates)
        let delta = abs(ha[0..., 0, 0...] - hb[0..., 0, 0...]).max().item(Float.self)
        XCTAssertGreaterThan(delta, 1e-6)
    }

    /// A masked pad key must not influence real positions.
    func testPaddingMaskHidesPaddedKeys() async throws {
        let model = try await makeModel()
        let a = MLXArray([9, 11, 13, 17] as [Int32]).reshaped(1, 4)
        let b = MLXArray([23, 11, 13, 17] as [Int32]).reshaped(1, 4)
        let mask = MLXArray([0, 1, 1, 1] as [Int32]).reshaped(1, 4)
        let ha = try XCTUnwrap(model(a, positionIds: nil, tokenTypeIds: nil, attentionMask: mask).hiddenStates)
        let hb = try XCTUnwrap(model(b, positionIds: nil, tokenTypeIds: nil, attentionMask: mask).hiddenStates)
        let delta = abs(ha[0..., 1..., 0...] - hb[0..., 1..., 0...]).max().item(Float.self)
        XCTAssertLessThanOrEqual(delta, 1e-5)
    }

    /// Bare-checkpoint keys plus a masked-LM head load strictly after sanitize.
    func testSanitizedBareCheckpointLoadsAllParameters() async throws {
        let model = try await makeModel()
        var bare: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            bare[key.hasPrefix("model.") ? String(key.dropFirst(6)) : key] = value
        }
        bare["head.dense.weight"] = MLXArray.zeros([8, 8])
        bare["decoder.weight"] = MLXArray.zeros([32, 8])
        bare["layers.0.attn.rotary_emb.inv_freq"] = MLXArray.zeros([2])
        let sanitized = try model.sanitize(weights: bare)
        XCTAssertEqual(
            Set(sanitized.keys), Set(model.parameters().flattened().map { $0.0 }))
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }

    func testRejectsInvalidConfigurationBeforeBuilding() async throws {
        let bad = Self.tinyConfig.replacingOccurrences(
            of: "\"num_attention_heads\": 2", with: "\"num_attention_heads\": 3")
        do {
            _ = try await makeModel(bad)
            XCTFail("expected a decoding error")
        } catch is DecodingError {
        }
    }

    func testUpstreamKeysStillRegistered() async {
        for key in ["bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert", "qwen3", "lfm2"] {
            let present = await EmbedderTypeRegistry.shared.contains(key)
            XCTAssertTrue(present, key)
        }
    }
}
