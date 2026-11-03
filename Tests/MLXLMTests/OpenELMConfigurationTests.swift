// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLLM

/// Replaces assert at OpenELM.swift:16; Release builds compile out `assert(modelDim % headDim == 0)` in `computeHeads`.
/// These tests state the same invariant explicitly via the real config-decoding caller.
struct OpenELMConfigurationTests {
    @Test func computeHeadsIsExactForDivisibleDims() {
        #expect(computeHeads(modelDim: 64, headDim: 16) == 4)
        #expect(computeHeads(modelDim: 128, headDim: 64) == 2)
    }

    @Test func makeDivisibleReturnsMultipleOfDivisor() {
        for v in stride(from: Float(1), through: 2000, by: 7.3) {
            for divisor in [8, 16, 64, 256] {
                let r = makeDivisible(v, divisor: divisor)
                #expect(r % divisor == 0, "v=\(v) divisor=\(divisor) -> \(r)")
                #expect(Float(r) >= 0.9 * v)
            }
        }
    }

    @Test(arguments: [
        (16, 128, 4), (64, 1280, 8), (64, 2048, 16), (32, 512, 12), (48, 768, 12),
    ] as [(Int, Int, Int)])
    func decodedConfigHasDivisibleQueryDims(headDim: Int, modelDim: Int, layers: Int) throws {
        let json = """
            {"model_type": "openelm", "head_dim": \(headDim), "num_transformer_layers": \(layers),
             "model_dim": \(modelDim), "vocab_size": 32000, "ffn_dim_divisor": 256}
            """
        let config = try JSONDecoder().decode(
            OpenElmConfiguration.self, from: Data(json.utf8))

        let groups = config.numGqaGroups
        #expect(config.numQueryHeads.count == layers)
        #expect(config.kvHeads.count == layers)
        for i in 0 ..< layers {
            let heads = config.numQueryHeads[i]
            let kv = config.kvHeads[i]
            #expect(heads > 0, "layer \(i): heads=\(heads)")
            // The per-layer query dimension (heads * headDim) must be a multiple of
            // headDim * groups, i.e. computeHeads never truncated and GQA splits evenly.
            #expect(
                (heads * headDim) % (headDim * groups) == 0,
                "layer \(i): heads=\(heads) headDim=\(headDim) groups=\(groups)")
            #expect(heads % groups == 0, "layer \(i): heads=\(heads) groups=\(groups)")
            #expect(kv >= 1, "layer \(i): kv=\(kv)")
            #expect(kv * groups == heads, "layer \(i): kv=\(kv) heads=\(heads)")
        }
    }
}
