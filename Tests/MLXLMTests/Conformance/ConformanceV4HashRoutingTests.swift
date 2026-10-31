// SL-35 step 4.3: DeepSeek-V4 hash routing (gate.tid2eid) for MoE layers with index < num_hash_layers.
//
// A hash layer picks its experts from the lookup table `tid2eid[token_id]` ([vocab, top-k] int32) instead of a
// score top-k, and has no e_score_correction_bias (vLLM nvidia/model.py:286-310). The routing WEIGHTS still
// come from the learned gate scores gathered at those experts.
//
// Verified: indices equal the table rows exactly; weights equal an independent recomputation
// (sqrtsoftplus(x W^T) gathered, top-k normalisation, routed_scaling_factor).
// INFERRED (no V4 source offline; support-kernels.json notes "normalize/scale order unverified"): the
// normalisation and scale are applied exactly as on the learned path (normalise only when norm_topk_prob,
// then multiply by routed_scaling_factor). Token ids are the model's input ids of the same forward call.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

struct ConformanceV4HashRoutingTests {
    static func config() throws -> SharpAIDeepseekV4Configuration {
        try ConformanceV4CompressorTests.attentionConfig(ratios: [0, 4, 128, 0])
    }

    /// A valid table: every row a distinct set of `topk` experts, deterministic.
    static func table(vocab: Int, experts: Int, topk: Int) -> MLXArray {
        var rows: [Int32] = []
        for v in 0 ..< vocab {
            for j in 0 ..< topk { rows.append(Int32((v * 3 + j * 5 + v / 7) % experts)) }
        }
        // distinct per row requires gcd-friendly strides; verify below in the test.
        return MLXArray(rows, [vocab, topk])
    }

    static func sqrtSoftplus(_ l: MLXArray) -> MLXArray { sqrt(log1p(exp(l))) }

    @Test("hash gate: indices come from tid2eid[token ids], weights from the gathered learned scores")
    func hashGateMatchesRecomputation() throws {
        let cfg = try Self.config()
        let gate = SharpAIDeepseekV4Gate(config: cfg, layerIndex: 0)  // num_hash_layers = 1 -> hash
        let tab = Self.table(vocab: cfg.vocabSize, experts: cfg.nRoutedExperts, topk: cfg.numExpertsPerTok)
        gate.update(parameters: ModuleParameters.unflattened([
            ("weight", MLXRandom.normal([cfg.nRoutedExperts, cfg.hiddenSize], key: MLXRandom.key(2))),
            ("tid2eid", tab),
        ]))
        let ids = MLXArray([5, 17, 33, 63, 0, 5], [2, 3]).asType(.int32)
        let x = MLXRandom.normal([2, 3, cfg.hiddenSize], key: MLXRandom.key(3))
        let (inds, w) = gate(x, tokenIds: ids)
        eval(inds, w)
        let wantInds = take(tab, ids, axis: 0)  // [2, 3, topk]
        #expect(inds.shape == [2, 3, cfg.numExpertsPerTok])
        #expect((inds.asType(.int32) .== wantInds).all().item(Bool.self), "indices differ from the table rows")
        let scores = Self.sqrtSoftplus(matmul(x, gate.weight.T))
        var want = takeAlong(scores, wantInds.asType(.int32), axis: -1)
        if cfg.normTopkProb { want = want / (want.sum(axis: -1, keepDims: true) + 1e-20) }
        want = want * cfg.routedScalingFactor
        eval(want)
        #expect(abs(w - want).max().item(Float.self) < 1e-5)
        // routing indices depend on the token only, not on the hidden state.
        let (inds2, _) = gate(MLXRandom.normal([2, 3, cfg.hiddenSize], key: MLXRandom.key(4)), tokenIds: ids)
        #expect((inds2.asType(.int32) .== wantInds).all().item(Bool.self))
    }

    @Test("the test table really has distinct experts per row (non-vacuity)")
    func tableIsValid() throws {
        let cfg = try Self.config()
        let t = Self.table(vocab: cfg.vocabSize, experts: cfg.nRoutedExperts, topk: cfg.numExpertsPerTok)
        let flat = t.asArray(Int32.self), k = cfg.numExpertsPerTok
        for v in 0 ..< cfg.vocabSize { #expect(Set(flat[(v * k) ..< ((v + 1) * k)]).count == k, "row \(v)") }
        #expect(Set(flat).count == cfg.nRoutedExperts, "table should use every expert")
    }

    @Test("hash layers own tid2eid and no e_score_correction_bias; learned layers the opposite")
    func parameterLayout() throws {
        let cfg = try Self.config()
        let hash = Set(SharpAIDeepseekV4Gate(config: cfg, layerIndex: 0).parameters().flattened().map { $0.0 })
        let learned = Set(SharpAIDeepseekV4Gate(config: cfg, layerIndex: 1).parameters().flattened().map { $0.0 })
        #expect(hash == ["weight", "tid2eid"])
        #expect(learned == ["weight", "e_score_correction_bias"])
        let t = SharpAIDeepseekV4Gate(config: cfg, layerIndex: 0).tid2eid
        #expect(t?.shape == [cfg.vocabSize, cfg.numExpertsPerTok] && t?.dtype == .int32)
    }

    @Test("learned layers ignore token ids")
    func learnedLayerIgnoresIds() throws {
        let cfg = try Self.config()
        let gate = SharpAIDeepseekV4Gate(config: cfg, layerIndex: 1)
        gate.update(parameters: ModuleParameters.unflattened([
            ("weight", MLXRandom.normal([cfg.nRoutedExperts, cfg.hiddenSize], key: MLXRandom.key(2))),
        ]))
        let x = MLXRandom.normal([1, 4, cfg.hiddenSize], key: MLXRandom.key(3))
        let (a, wa) = gate(x)
        let (b, wb) = gate(x, tokenIds: MLXArray([1, 2, 3, 4], [1, 4]).asType(.int32))
        eval(a, wa, b, wb)
        #expect((a .== b).all().item(Bool.self) && abs(wa - wb).max().item(Float.self) == 0)
    }

    @Test("model forward routes hash layer 0 by token id: changing the table changes the logits")
    func modelUsesTable() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        eval(m)
        let layer0 = try #require(m.model.layers.first)
        // the freshly built routed experts are all-zero; give them distinct weights so the choice matters.
        ConformanceV4CompressorTests.randomize(layer0.ffn, seed: 9, scale: 0.5, skip: ["gate.tid2eid", "gate.e_score_correction_bias"])
        let ids = MLXArray([1, 2, 3, 4, 5, 6], [1, 6]).asType(.int32)
        func run(_ shift: Int32) -> MLXArray {
            let base = (0 ..< 64 * 2).map { Int32($0 % 4) }
            let tab = MLXArray(base.map { ($0 + shift) % 4 }, [64, 2])
            layer0.ffn.gate.update(parameters: ModuleParameters.unflattened([("tid2eid", tab)]))
            let out = m(ids, cache: nil)
            eval(out)
            return out
        }
        let a = run(0), b = run(1)
        #expect(a.shape == [1, 6, 64])
        #expect(a.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(abs(a - b).max().item(Float.self) > 1e-6, "hash table has no effect on the forward pass")
    }
}
