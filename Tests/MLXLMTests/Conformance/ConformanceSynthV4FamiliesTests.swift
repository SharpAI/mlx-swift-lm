// Conformance harness, round 3 (SL-29 phase 2): synthetic-checkpoint coverage for more registered families.
//
// Same pattern as ConformanceSynthWeightsTests (checkpoint-style keys -> random weights -> sanitize ->
// parameter parity -> load -> forward), but the config of each family is a real offline fixture shrunk to
// tiny dimensions with its FULL key set kept (`SynthFixture.tiny`). Families chosen from the registered
// model types that have a fixture: nemotron_h (hybrid Mamba2 + attention + MoE), glm4_moe (MoE with a
// trailing MTP layer in the checkpoint), lfm2 (hybrid short-conv + attention).
// upstream-watch: NemotronH.swift, GLM4MOE.swift, LFM2.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

enum SynthFamilies {
    typealias Entry = ConformanceSynth.Entry

    /// Depthwise conv weights are stored `[C, 1, K]` in HF checkpoints and `[C, K, 1]` in the module.
    static func swapConv(_ native: [Entry]) -> [Entry] {
        native.map { e in
            guard e.key.hasSuffix("conv1d.weight") || e.key.hasSuffix("conv.conv.weight"), e.shape.count == 3,
                e.shape[2] == 1
            else { return e }
            return Entry(key: e.key, shape: [e.shape[0], 1, e.shape[1]], dtype: e.dtype)
        }
    }

    /// `mixer.switch_mlp.<fc>.weight [E, a, b]` -> `mixer.experts.<e>.<name>.weight [a, b]`.
    static func unstack(_ native: [Entry], rename: [String: String]) -> [Entry] {
        var out: [Entry] = []
        for e in native {
            if let r = e.key.range(of: ".switch_mlp."), e.shape.count == 3 {
                let rest = e.key[r.upperBound...]  // "fc1.weight"
                let proj = String(rest.prefix { $0 != "." })
                let suffix = rest.dropFirst(proj.count)
                for x in 0 ..< e.shape[0] {
                    out.append(
                        Entry(
                            key: "\(e.key[..<r.lowerBound]).experts.\(x).\(rename[proj] ?? proj)\(suffix)",
                            shape: Array(e.shape.dropFirst()), dtype: e.dtype))
                }
            } else {
                out.append(e)
            }
        }
        return out
    }

    static func spec(
        _ name: String, dir: String, modelType: String, overrides: [String: Any],
        checkpoint: @escaping @Sendable ([Entry]) -> [Entry],
        stray: @escaping @Sendable ([Entry]) -> Set<String> = { _ in [] }
    ) throws -> ConformanceSynth.SynthSpec {
        .init(
            name: name, modelType: modelType,
            json: SynthFixture.string(try SynthFixture.tiny(dir, overrides: overrides)),
            checkpointEntries: checkpoint, strayKeys: stray)
    }

    static let nemotronH = try! spec(
        "nemotron_h (mamba2 + attention + MoE, hybrid pattern ME*M-)", dir: "nemotron_h", modelType: "nemotron_h",
        overrides: [
            "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 5, "hybrid_override_pattern": "ME*M-",
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 64,
            "mamba_num_heads": 4, "mamba_head_dim": 8, "n_groups": 2, "ssm_state_size": 8, "conv_kernel": 4,
            "moe_intermediate_size": 16, "moe_shared_expert_intermediate_size": 32, "n_routed_experts": 4,
            "n_shared_experts": 1, "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1,
            "max_position_embeddings": 256,
        ],
        checkpoint: { swapConv(unstack($0, rename: ["fc1": "up_proj", "fc2": "down_proj"])) })

    static let glm4Moe = try! spec(
        "glm4_moe (MoE + dense lead layer + MTP layer in checkpoint)", dir: "glm4_moe", modelType: "glm4_moe",
        overrides: [
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "moe_intermediate_size": 16,
            "num_hidden_layers": 3, "first_k_dense_replace": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "n_group": 1, "topk_group": 1, "max_position_embeddings": 256, "num_nextn_predict_layers": 1,
        ],
        checkpoint: { native in
            // Real GLM-4.5 checkpoints carry the next-token-predict block as layer `num_hidden_layers`.
            ConformanceSynth.unstackExperts(native) + [
                Entry(key: "model.layers.3.eh_proj.weight", shape: [32, 64], dtype: .float32),
                Entry(key: "model.layers.3.enorm.weight", shape: [32], dtype: .float32),
                Entry(key: "model.layers.3.hnorm.weight", shape: [32], dtype: .float32),
                Entry(key: "model.layers.3.shared_head.norm.weight", shape: [32], dtype: .float32),
            ]
        },
        stray: { _ in
            ["model.layers.3.eh_proj.weight", "model.layers.3.enorm.weight", "model.layers.3.hnorm.weight",
             "model.layers.3.shared_head.norm.weight"]
        })

    static let lfm2 = try! spec(
        "lfm2 (short-conv + attention hybrid)", dir: "lfm2", modelType: "lfm2",
        overrides: [
            "vocab_size": 64, "hidden_size": 32, "block_dim": 32, "block_ff_dim": 64, "intermediate_size": 64,
            "num_hidden_layers": 4, "layer_types": ["conv", "conv", "full_attention", "conv"],
            "num_attention_heads": 4, "num_heads": 4, "num_key_value_heads": 2, "conv_dim": 32,
            "max_position_embeddings": 256,
        ],
        checkpoint: { swapConv($0) })

    static let all: [ConformanceSynth.SynthSpec] = [nemotronH, glm4Moe, lfm2]
}

struct ConformanceSynthV4FamiliesTests {

    @Test("fixture-derived synthetic checkpoint: sanitize accepts exactly the module's keys, forward is finite",
        arguments: SynthFamilies.all)
    func familyRoundTrip(spec: ConformanceSynth.SynthSpec) async throws {
        let out = try await ConformanceSynth.run(spec)
        #expect(out.report.missing.isEmpty, "\(spec.name) missing: \(out.report.missing.prefix(5))")
        #expect(out.report.unexpected.isEmpty, "\(spec.name) unexpected: \(out.report.unexpected.prefix(5))")
        #expect(out.report.shapeMismatch.isEmpty, "\(spec.name) shapes: \(out.report.shapeMismatch.prefix(5))")
        #expect(out.logits.shape == [1, 4, 64], "\(spec.name) logits shape \(out.logits.shape)")
        #expect(out.logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test("the shrunk config keeps the real fixture's complete key set (only values change)")
    func keySetPreserved() throws {
        for (dir, spec) in [("nemotron_h", SynthFamilies.nemotronH), ("glm4_moe", SynthFamilies.glm4Moe),
                            ("lfm2", SynthFamilies.lfm2)]
        {
            let real = Set(try SynthFixture.object(dir).keys)
            let tiny = try #require(
                try JSONSerialization.jsonObject(with: Data(spec.json.utf8)) as? [String: Any])
            #expect(Set(tiny.keys) == real, "\(dir): key set changed")
        }
    }

    @Test("the MoE specs exercise the per-expert checkpoint path, the hybrid specs the conv axis swap")
    func pathsAreExercised() async throws {
        for spec in [SynthFamilies.nemotronH, SynthFamilies.glm4Moe] {
            let model = try await ConformanceBuild.llm(spec.json, modelType: spec.modelType)
            let native = ConformanceSynth.nativeEntries(try #require(model as? Module))
            let ckpt = spec.checkpointEntries(native)
            #expect(ckpt.contains { $0.key.contains(".experts.3.") }, "\(spec.name): no per-expert tensors")
            #expect(!ckpt.contains { $0.key.contains(".switch_mlp.") && $0.shape.count == 3 }, "\(spec.name)")
        }
        for spec in [SynthFamilies.nemotronH, SynthFamilies.lfm2] {
            let model = try await ConformanceBuild.llm(spec.json, modelType: spec.modelType)
            let native = ConformanceSynth.nativeEntries(try #require(model as? Module))
            let conv = try #require(native.first { $0.key.hasSuffix("conv1d.weight") || $0.key.hasSuffix("conv.conv.weight") })
            let ck = try #require(spec.checkpointEntries(native).first { $0.key == conv.key })
            #expect(conv.shape[2] == 1 && ck.shape == [conv.shape[0], 1, conv.shape[1]], "\(spec.name)")
        }
    }

    // The checker must be able to fail: otherwise "clean" proves nothing.
    // (A hole inside a stacked expert set is not used: GLM4MoE.sanitize force-unwraps removeValue and would trap.)
    @Test("a dropped dense tensor is detected as missing", arguments: SynthFamilies.all)
    func detectsMissingTensor(spec: ConformanceSynth.SynthSpec) async throws {
        var victim = ""
        let out = try await ConformanceSynth.run(spec) { ckpt in
            let i = ckpt.firstIndex { $0.key.hasSuffix(".norm.weight") || $0.key.hasSuffix("layernorm.weight")
                || $0.key.hasSuffix("_norm.weight") || $0.key.hasSuffix("layer_norm.weight") }
            if let i { victim = ckpt[i].key; ckpt.remove(at: i) }
        }
        #expect(!victim.isEmpty, "\(spec.name): no norm tensor found to drop")
        #expect(out.report.missing == [victim], "\(spec.name): \(out.report.missing)")
    }
}
