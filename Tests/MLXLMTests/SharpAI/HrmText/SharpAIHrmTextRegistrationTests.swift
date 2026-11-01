// SharpAI fork overlay tests for the hrm_text creator and forward pass. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHrmTextRegistrationTests: XCTestCase {

    // Test sizes, not the HRM-Text-1B fixture: 2 layers per stack, H_cycles 2, L_cycles 3.
    // That gives 2 * 2 * (3 + 1) = 16 cache slots.
    private let validJSON = """
        {
            "model_type": "hrm_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 512,
            "H_cycles": 2,
            "L_cycles": 3,
            "L_bp_cycles": [0]
        }
        """

    private func configuration() throws -> SharpAIHrmTextConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIHrmTextConfiguration.self, from: Data(validJSON.utf8))
    }

    private func tokens(_ ids: [Int32]) -> MLXArray {
        MLXArray(ids, [1, ids.count])
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIHrmTextOverlay.creators["hrm_text"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIHrmTextOverlay.creators["hrm_text"] else {
            XCTFail("hrm_text creator missing")
            return
        }

        // The forward pass is implemented now, so a valid config builds the model.
        let model = try creator(Data(validJSON.utf8))
        XCTAssertTrue(model is SharpAIHrmTextForCausalLM)
        XCTAssertEqual(try model.newCache(parameters: nil).count, 16)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIHrmTextOverlay.creators["hrm_text"] else {
            XCTFail("hrm_text creator missing")
            return
        }
        let json = """
            { "hidden_size": 64 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testHierarchicalLoopsKnownIssueSkipped() throws {
        // Reference: HrmTextModel.forward (transformers v5.22.1) written out as its documented loop:
        //   for h in range(H_cycles):
        //       for l in range(L_cycles): z_L = L(z_L + z_H)
        //       z_H = H(z_H + z_L)
        // The input embeddings enter only as the initial z_H.
        let config = try configuration()
        let model = SharpAIHrmTextForCausalLM(config)
        let inner = model.model
        let ids = tokens([1, 5, 9, 2])

        let actual = inner(ids, cache: nil, tokenTypeIds: nil)

        var zH = inner.embedTokens(ids) * inner.embeddingScale
        var zL = broadcast(inner.zLInit, to: zH.shape)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode =
            createAttentionMask(h: zH, cache: nil as KVCache?)
        for _ in 0 ..< config.hCycles {
            for _ in 0 ..< config.lCycles {
                zL = inner.lModule(zL + zH, mask: mask, cache: nil, cycleOffset: 0)
            }
            zH = inner.hModule(zH + zL, mask: mask, cache: nil, cycleOffset: 0)
        }

        XCTAssertEqual(actual.shape, zH.shape)
        XCTAssertLessThan(maxAbsDiff(actual, zH), 1e-4)
    }

    func testPerStepCacheKnownIssueSkipped() throws {
        let config = try configuration()
        let model = SharpAIHrmTextForCausalLM(config)
        XCTAssertEqual(config.cacheSlotCount, 16)

        // Full causal forward without a cache, as the reference.
        let full = model(tokens([1, 5, 9, 2]), cache: nil)

        // Prefill 3 tokens, then decode the 4th. Each stack call owns its own slots.
        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, config.cacheSlotCount)

        let prefill = model(tokens([1, 5, 9]), cache: caches)
        XCTAssertEqual(prefill.shape, [1, 3, 512])
        // Every slot was used exactly once. A shared or skipped slot would show a wrong offset.
        for cache in caches {
            XCTAssertEqual(cache.offset, 3)
        }

        let decode = model(tokens([2]), cache: caches)
        XCTAssertEqual(decode.shape, [1, 1, 512])
        for cache in caches {
            XCTAssertEqual(cache.offset, 4)
        }

        // The cached decode step matches the last position of the full forward.
        let fullLast = full[0..., 3, 0...]
        let decodeLast = decode[0..., 0, 0...]
        XCTAssertLessThan(maxAbsDiff(fullLast, decodeLast), 1e-3)
    }

    func testTinyForwardPassSkipped() throws {
        guard let creator = SharpAIHrmTextOverlay.creators["hrm_text"] else {
            XCTFail("hrm_text creator missing")
            return
        }
        let model = try creator(Data(validJSON.utf8))

        let logits = model(tokens([1, 5, 9]), cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 512])
        let flat = logits.asArray(Float.self)
        XCTAssertTrue(flat.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(flat.map { abs($0) }.max() ?? 0, 0)
    }
}
