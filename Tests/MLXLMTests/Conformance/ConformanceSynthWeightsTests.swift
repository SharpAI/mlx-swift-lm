// Conformance harness, step B2: config-driven synthetic checkpoints.
//
// From a (reduced) config JSON this builds the model through the public registry, then does what a
// real load does, minus the download:
//   1. fabricate a *checkpoint-style* key list (the names an HF/mlx-community safetensors file
//      would use, which can differ from the module's own names, e.g. per-expert MoE tensors),
//   2. fill it with deterministic random weights,
//   3. run it through the model's own `sanitize(weights:)`,
//   4. require the sanitized key set (and shapes) to equal the module's parameter set exactly
//      (no missing, no unexpected), load it with verification, and run a forward pass.
//
// The checkpoint key list is derived from the module's native parameters by an explicit,
// per-architecture inverse mapping (`SynthSpec.checkpointEntries`), so a sanitize() that forgets to
// stack experts / drop a stray tensor is caught. No network, no real weights, tiny GPU use.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

enum ConformanceSynth {

    struct Entry {
        var key: String
        var shape: [Int]
        var dtype: DType
    }

    /// One architecture under test.
    struct SynthSpec: Sendable {
        var name: String
        var modelType: String
        var json: String
        /// Maps the module's native `(key, shape, dtype)` list to the list a checkpoint file would
        /// contain. Includes tensors that a loader is expected to drop (the "stray" keys).
        var checkpointEntries: @Sendable ([Entry]) -> [Entry]
        /// Checkpoint keys that sanitize() must drop (so they must not survive into the result).
        var strayKeys: @Sendable ([Entry]) -> Set<String>
    }

    struct Report: Equatable {
        var missing: [String] = []
        var unexpected: [String] = []
        var shapeMismatch: [String] = []
        var isClean: Bool { missing.isEmpty && unexpected.isEmpty && shapeMismatch.isEmpty }
    }

    static func nativeEntries(_ model: Module) -> [Entry] {
        model.parameters().flattened().map {
            Entry(key: $0.0, shape: $0.1.shape, dtype: $0.1.dtype)
        }.sorted { $0.key < $1.key }
    }

    /// Deterministic random weights: one independent PRNG key per tensor, derived from `seed` and
    /// the tensor's position in the sorted key list. Small scale keeps activations finite.
    static func randomWeights(_ entries: [Entry], seed: UInt64) -> [String: MLXArray] {
        let sorted = entries.sorted { $0.key < $1.key }
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: max(sorted.count, 2))
        var out: [String: MLXArray] = [:]
        for (i, e) in sorted.enumerated() {
            out[e.key] = (MLXRandom.normal(e.shape, key: keys[i]) * 0.02).asType(e.dtype)
        }
        return out
    }

    /// Compare a sanitized checkpoint against the module's native parameter set.
    static func compare(sanitized: [String: MLXArray], native: [Entry]) -> Report {
        var r = Report()
        let nativeByKey = Dictionary(uniqueKeysWithValues: native.map { ($0.key, $0) })
        r.missing = nativeByKey.keys.filter { sanitized[$0] == nil }.sorted()
        r.unexpected = sanitized.keys.filter { nativeByKey[$0] == nil }.sorted()
        r.shapeMismatch = sanitized.compactMap { k, v in
            guard let n = nativeByKey[k], n.shape != v.shape else { return nil }
            return "\(k): checkpoint \(v.shape) vs module \(n.shape)"
        }.sorted()
        return r
    }

    // MARK: inverse mappings

    /// Checkpoint names equal module names (llama, gemma3_text).
    static func identity(extra: @escaping @Sendable ([Entry]) -> [Entry] = { _ in [] })
        -> @Sendable ([Entry]) -> [Entry]
    {
        { native in native + extra(native) }
    }

    /// HF MoE layout: `mlp.switch_mlp.<p>.weight [E, a, b]` is stored as E tensors
    /// `mlp.experts.<e>.<p>.weight [a, b]`.
    static func unstackExperts(_ native: [Entry]) -> [Entry] {
        var out: [Entry] = []
        for e in native {
            if e.key.contains(".mlp.switch_mlp."), e.shape.count == 3 {
                for x in 0 ..< e.shape[0] {
                    out.append(
                        Entry(
                            key: e.key.replacingOccurrences(
                                of: ".mlp.switch_mlp.", with: ".mlp.experts.\(x)."),
                            shape: Array(e.shape.dropFirst()), dtype: e.dtype))
                }
            } else {
                out.append(e)
            }
        }
        return out
    }

    // MARK: specs

    static let llamaTied = SynthSpec(
        name: "llama (tied embeddings)", modelType: "llama",
        json: """
            {"model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2,
             "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-5, "vocab_size": 64, "tie_word_embeddings": true}
            """,
        checkpointEntries: identity(extra: { native in
            // Real llama checkpoints carry precomputed rotary tables and (when tied) an lm_head.
            let layers = Set(native.compactMap { $0.key.split(separator: ".").dropFirst(2).first })
            let rotary = layers.sorted().map {
                Entry(key: "model.layers.\($0).self_attn.rotary_emb.inv_freq", shape: [4], dtype: .float32)
            }
            return rotary + [Entry(key: "lm_head.weight", shape: [64, 32], dtype: .float32)]
        }),
        strayKeys: { native in
            let layers = Set(native.compactMap { $0.key.split(separator: ".").dropFirst(2).first })
            return Set(layers.map { "model.layers.\($0).self_attn.rotary_emb.inv_freq" })
                .union(["lm_head.weight"])
        })

    static let qwen3MoE = SynthSpec(
        name: "qwen3_moe (per-expert checkpoint)", modelType: "qwen3_moe",
        json: """
            {"model_type": "qwen3_moe", "hidden_size": 32, "num_hidden_layers": 2,
             "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
             "mlp_only_layers": [], "moe_intermediate_size": 16, "rms_norm_eps": 1e-6,
             "vocab_size": 64, "tie_word_embeddings": false}
            """,
        checkpointEntries: unstackExperts,
        strayKeys: { _ in [] })

    static let gemma3Text = SynthSpec(
        name: "gemma3_text", modelType: "gemma3_text",
        json: ConformanceForwardTests.gemma3TextJSON(softcap: nil),
        checkpointEntries: identity(), strayKeys: { _ in [] })

    static let all: [SynthSpec] = [llamaTied, qwen3MoE, gemma3Text]

    // MARK: driver

    struct Outcome {
        var report: Report
        var logits: MLXArray
        var checkpointKeyCount: Int
    }

    static func run(
        _ spec: SynthSpec, seed: UInt64 = 7,
        mutateCheckpoint: (inout [Entry]) -> Void = { _ in }
    ) async throws -> Outcome {
        let model = try await ConformanceBuild.llm(spec.json, modelType: spec.modelType)
        let module = try #require(model as? Module)
        let native = nativeEntries(module)
        var ckpt = spec.checkpointEntries(native)
        mutateCheckpoint(&ckpt)
        let weights = randomWeights(ckpt, seed: seed)
        let sanitized = try model.sanitize(weights: weights)

        for stray in spec.strayKeys(native) {
            #expect(sanitized[stray] == nil, "\(spec.name): sanitize kept stray key \(stray)")
        }
        let report = compare(sanitized: sanitized, native: native)
        if report.isClean {
            try module.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
            eval(module)
        }
        let logits = try ConformanceBuild.logits(model, tokens: [1, 2, 3, 4])
        return Outcome(report: report, logits: logits, checkpointKeyCount: ckpt.count)
    }
}

struct ConformanceSynthWeightsTests {

    @Test("synthetic checkpoint: sanitize accepts exactly the module's keys, forward is finite",
        arguments: ConformanceSynth.all)
    func synthCheckpointRoundTrip(spec: ConformanceSynth.SynthSpec) async throws {
        let out = try await ConformanceSynth.run(spec)
        #expect(out.report.missing.isEmpty, "\(spec.name) missing: \(out.report.missing.prefix(5))")
        #expect(out.report.unexpected.isEmpty, "\(spec.name) unexpected: \(out.report.unexpected.prefix(5))")
        #expect(out.report.shapeMismatch.isEmpty, "\(spec.name) shapes: \(out.report.shapeMismatch.prefix(5))")
        #expect(out.logits.shape == [1, 4, 64], "\(spec.name) logits shape \(out.logits.shape)")
        #expect(out.logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test("the MoE spec really exercises the per-expert path")
    func moeCheckpointIsPerExpert() async throws {
        let out = try await ConformanceSynth.run(ConformanceSynth.qwen3MoE)
        let model = try await ConformanceBuild.llm(
            ConformanceSynth.qwen3MoE.json, modelType: "qwen3_moe")
        let native = ConformanceSynth.nativeEntries(try #require(model as? Module))
        // 4 experts x 3 projections x 2 layers replace 3 stacked tensors x 2 layers.
        #expect(out.checkpointKeyCount == native.count - 6 + 4 * 3 * 2)
    }

    @Test("same seed gives identical weights, different seed does not")
    func determinism() {
        let entries = [ConformanceSynth.Entry(key: "a.weight", shape: [4, 4], dtype: .float32)]
        let a = ConformanceSynth.randomWeights(entries, seed: 3)["a.weight"]!
        let b = ConformanceSynth.randomWeights(entries, seed: 3)["a.weight"]!
        let c = ConformanceSynth.randomWeights(entries, seed: 4)["a.weight"]!
        #expect(allClose(a, b).item(Bool.self))
        #expect(!allClose(a, c).item(Bool.self))
    }

    // The checker must be able to fail: otherwise "clean" proves nothing.
    @Test("detects a missing checkpoint tensor")
    func detectsMissing() async throws {
        let out = try await ConformanceSynth.run(ConformanceSynth.llamaTied) { ckpt in
            ckpt.removeAll { $0.key == "model.layers.1.mlp.up_proj.weight" }
        }
        #expect(out.report.missing == ["model.layers.1.mlp.up_proj.weight"])
    }

    @Test("detects an unexpected checkpoint tensor")
    func detectsUnexpected() async throws {
        let out = try await ConformanceSynth.run(ConformanceSynth.gemma3Text) { ckpt in
            ckpt.append(.init(key: "model.layers.0.bogus.weight", shape: [2], dtype: .float32))
        }
        #expect(out.report.unexpected == ["model.layers.0.bogus.weight"])
    }
}
