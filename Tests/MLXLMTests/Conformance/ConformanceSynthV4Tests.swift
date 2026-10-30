// Conformance harness, round 3 (SL-29 phase 2): synthetic deepseek_v4 checkpoint in the REAL key layout.
//
// Flow: Flash fixture shrunk to tiny dims (same key set) -> tensor list of the real checkpoint derived from
// vLLM's loader (ConformanceSynthV4Layout.swift) -> random weights -> safetensors on disk -> our
// `sanitize` + `loadWeights(modelDirectory:model:)` -> parameter-key parity against our module.
//
// Green tests record what matches today (shapes of every shared key, accounting, a clean load and forward).
// Every difference is a precise test wrapped in `ConformanceGap.knownIssue` (green while the gap exists AND once
// it is fixed; delete the wrapper in the PR that fixes it). The `.other` buckets are NOT wrapped: an
// unexplained difference is a real failure. `CONFORMANCE_GAP_RAW=1` runs the wrapped tests bare.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

struct ConformanceSynthV4Case {
    var cfg: [String: Any]
    var model: DeepseekV4Model
    var native: [ConformanceSynth.Entry]
    var raw: [ConformanceSynth.Entry]  // real layout, HF names
    var converted: [ConformanceSynth.Entry]  // real layout, mlx-converted names, per-expert tensors (loader input)
    var comparable: [ConformanceSynth.Entry]  // same, experts stacked like the module's switch_mlp (key diff input)
    var mainLayers: Int  // transformer layers OUR model builds
    var realMainLayers: Int  // transformer layers the REAL checkpoint has (num_hidden_layers)

    var nativeKeys: Set<String> { Set(native.map(\.key)) }
    var convertedKeys: Set<String> { Set(comparable.map(\.key)) }

    static func make() async throws -> ConformanceSynthV4Case {
        let cfg = try ConformanceSynthV4Layout.tinyConfig()
        let model = try await ConformanceSynthV4Layout.model(cfg)
        eval(model)
        let raw = ConformanceSynthV4Layout.realEntries(cfg)
        let converted = raw.map {
            ConformanceSynth.Entry(key: ConformanceSynthV4Layout.mlxName($0.key), shape: $0.shape, dtype: $0.dtype)
        }
        return ConformanceSynthV4Case(
            cfg: cfg, model: model, native: ConformanceSynth.nativeEntries(model), raw: raw,
            converted: converted, comparable: ConformanceSynthV4Layout.stackExperts(converted), mainLayers: model.model.layers.count,
            realMainLayers: cfg["num_hidden_layers"] as! Int)
    }

    func weights(_ entries: [ConformanceSynth.Entry], seed: UInt64 = 11) -> [String: MLXArray] {
        ConformanceSynth.randomWeights(entries, seed: seed)
    }

    /// Real keys (converted names) that OUR module has no parameter for, grouped.
    func realOnly(_ g: ConformanceSynthV4Layout.Group) -> [String] {
        convertedKeys.subtracting(nativeKeys).filter {
            ConformanceSynthV4Layout.group($0, mainLayers: mainLayers) == g
        }.sorted()
    }

    /// Parameters OUR module has that the real layout lacks, grouped.
    func oursOnly(_ g: ConformanceSynthV4Layout.Group) -> [String] {
        nativeKeys.subtracting(convertedKeys).filter {
            ConformanceSynthV4Layout.group($0, mainLayers: mainLayers) == g
        }.sorted()
    }
}

enum ConformanceSynthV4 {
    static func tempDir(_ name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("conformance-\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

struct ConformanceSynthV4Tests {

    // MARK: green facts

    @Test("real layout: per-layer tensor families follow the vLLM declarations")
    func layoutFollowsVllm() async throws {
        let c = try await ConformanceSynthV4Case.make()
        let keys = Set(c.raw.map(\.key))
        #expect(c.realMainLayers == 3 && (c.cfg["compress_ratios"] as? [Int]) == [0, 4, 128, 0])
        // hash layer 0: tid2eid, no gate bias; layers 1, 2: gate bias, no tid2eid.
        #expect(keys.contains("layers.0.ffn.gate.tid2eid") && !keys.contains("layers.0.ffn.gate.bias"))
        for l in [1, 2] {
            #expect(keys.contains("layers.\(l).ffn.gate.bias") && !keys.contains("layers.\(l).ffn.gate.tid2eid"))
        }
        // ratio 0 (-> 1): neither; ratio 4: compressor AND indexer (+ its own compressor); ratio 128: compressor only.
        #expect(!keys.contains { $0.hasPrefix("layers.0.attn.compressor.") || $0.hasPrefix("layers.0.attn.indexer.") })
        #expect(keys.contains("layers.1.attn.compressor.ape") && keys.contains("layers.1.attn.indexer.wq_b.weight"))
        #expect(keys.contains("layers.1.attn.indexer.compressor.wkv.weight"))
        #expect(keys.contains("layers.2.attn.compressor.ape") && !keys.contains { $0.hasPrefix("layers.2.attn.indexer.") })
        // ratio-4 compressors overlap (coff 2): ape [4, 2*head_dim]; ratio 128: [128, head_dim].
        let shapes = Dictionary(uniqueKeysWithValues: c.raw.map { ($0.key, $0.shape) })
        #expect(shapes["layers.1.attn.compressor.ape"] == [4, 32])
        #expect(shapes["layers.2.attn.compressor.ape"] == [128, 16])
        #expect(shapes["layers.1.attn.indexer.compressor.ape"] == [4, 16])
        // no mtp.* and no MTP layer index in the main-model layout.
        #expect(!keys.contains { $0.hasPrefix("mtp.") || $0.hasPrefix("layers.3.") })
    }

    @Test("synthetic V4 checkpoint survives a safetensors round trip")
    func safetensorsRoundTrip() async throws {
        let c = try await ConformanceSynthV4Case.make()
        let dir = try ConformanceSynthV4.tempDir("v4-st")
        defer { try? FileManager.default.removeItem(at: dir) }
        let w = c.weights(c.raw)
        let url = dir.appendingPathComponent("model.safetensors")
        try save(arrays: w, url: url)
        let back = try loadArrays(url: url)
        #expect(Set(back.keys) == Set(w.keys) && back.count == c.raw.count)
        for (k, v) in w {
            #expect(back[k]?.shape == v.shape && back[k]?.dtype == v.dtype, "\(k)")
        }
    }

    @Test("every key present in both layouts has the same shape")
    func commonKeysShapeParity() async throws {
        let c = try await ConformanceSynthV4Case.make()
        let native = Dictionary(uniqueKeysWithValues: c.native.map { ($0.key, $0.shape) })
        var bad: [String] = []
        var common = 0
        for e in c.comparable {
            guard let s = native[e.key] else { continue }
            common += 1
            if s != e.shape { bad.append("\(e.key): real \(e.shape) vs ours \(s)") }
        }
        #expect(bad.isEmpty, "shape differences: \(bad.prefix(5))")
        #expect(common > 40, "only \(common) keys in common: the rename layer probably broke")
    }

    @Test("accounting: real = shared + real-only, ours = shared + ours-only, each bucket classified")
    func accounting() async throws {
        let c = try await ConformanceSynthV4Case.make()
        let shared = c.convertedKeys.intersection(c.nativeKeys).count
        let realOnly = ConformanceSynthV4Layout.Group.allCases.reduce(0) { $0 + c.realOnly($1).count }
        let oursOnly = ConformanceSynthV4Layout.Group.allCases.reduce(0) { $0 + c.oursOnly($1).count }
        #expect(c.comparable.count == shared + realOnly)
        #expect(c.native.count == shared + oursOnly)
        #expect(c.convertedKeys.count == c.comparable.count, "renames collided")
        print("[synth-v4] real=\(c.comparable.count) (\(c.raw.count) tensors before expert stacking) ours=\(c.native.count) shared=\(shared) real-only=\(realOnly) ours-only=\(oursOnly)")
        for g in ConformanceSynthV4Layout.Group.allCases {
            print("[synth-v4]   real-only \(g.rawValue): \(c.realOnly(g).count)   ours-only \(g.rawValue): \(c.oursOnly(g).count)")
        }
    }

    @Test("our sanitize + the real loader accept the converted real checkpoint; forward is finite")
    func loadsThroughRealLoader() async throws {
        let c = try await ConformanceSynthV4Case.make()
        let dir = try ConformanceSynthV4.tempDir("v4-load")
        defer { try? FileManager.default.removeItem(at: dir) }
        try save(arrays: c.weights(c.converted), url: dir.appendingPathComponent("model.safetensors"))
        let fresh = try await ConformanceSynthV4Layout.model(c.cfg)
        try await loadWeights(modelDirectory: dir, model: fresh)
        eval(fresh)
        let out = fresh(MLXArray([1, 2, 3, 4]).reshaped(1, 4), cache: nil)
        eval(out)
        #expect(out.shape == [1, 4, 64])
        #expect(out.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    // MARK: wrapped gaps

    @Test("raw HF key names (embed.weight, layers.N.hc_attn_fn, ffn.experts.E.w1 ...) load through sanitize")
    func rawLayoutNeedsNoConverter() async throws {
        let c = try await ConformanceSynthV4Case.make()
        let sanitized = c.model.sanitize(weights: c.weights(c.raw))
        let unexpected = Set(sanitized.keys).subtracting(c.nativeKeys).sorted()
        // Non-vacuity (outside the wrapper): there is a raw checkpoint to sanitize, and sanitize returned tensors.
        #expect(!c.raw.isEmpty && !sanitized.isEmpty, "nothing to sanitize: the wrapped claim below would be vacuous")
        // sanitize has no rename layer: loading a raw checkpoint needs an external converter (mlx convert).
        ConformanceGap.knownIssue("DeepseekV4.sanitize has no HF->module rename layer; raw checkpoints need a converter") {
            ConformanceGap.expect(unexpected.isEmpty, "\(unexpected.count) unexpected, e.g. \(unexpected.prefix(4))")
        }
    }

    static let missingFromOurs: [ConformanceSynthV4Layout.Group] = [.compressor, .indexer, .tid2eid, .extraMainLayer]

    @Test("real tensors our module has no parameter for", arguments: ConformanceSynthV4Layout.Group.allCases)
    func realTensorsWithoutHome(_ g: ConformanceSynthV4Layout.Group) async throws {
        let c = try await ConformanceSynthV4Case.make()
        let keys = c.realOnly(g)
        let msg = "\(g.rawValue): \(keys.count) real tensors have no parameter, e.g. \(keys.prefix(3))"
        // Non-vacuity (outside the wrapper): both layouts are populated, and a gap group really occurs in the
        // real checkpoint (otherwise `keys.isEmpty` below would hold for the wrong reason).
        #expect(!c.convertedKeys.isEmpty && !c.nativeKeys.isEmpty && !c.convertedKeys.isDisjoint(with: c.nativeKeys))
        if Self.missingFromOurs.contains(g) {
            let present = c.convertedKeys.filter { ConformanceSynthV4Layout.group($0, mainLayers: c.mainLayers) == g }
            #expect(!present.isEmpty, "\(g.rawValue): the real layout has no such tensor; the gap pin is vacuous")
        }
        if Self.missingFromOurs.contains(g) {
            ConformanceGap.knownIssue("DeepseekV4 lacks a parameter for real `\(g.rawValue)` tensors") {
                ConformanceGap.expect(keys.isEmpty, "\(msg)")
            }
        } else {
            #expect(keys.isEmpty, "unexplained real-only tensors: \(msg)")
        }
    }

    @Test("parameters of our module that the real layout does not have", arguments: ConformanceSynthV4Layout.Group.allCases)
    func oursWithoutRealCounterpart(_ g: ConformanceSynthV4Layout.Group) async throws {
        let c = try await ConformanceSynthV4Case.make()
        let keys = c.oursOnly(g)
        let msg = "\(g.rawValue): \(keys.count) module parameters not in the real checkpoint, e.g. \(keys.prefix(3))"
        #expect(!c.convertedKeys.isEmpty && !c.nativeKeys.isEmpty)
        if g == .gateBias {
            // Non-vacuity: our module really has gate-bias parameters (the claim is that some are surplus).
            #expect(c.nativeKeys.contains { ConformanceSynthV4Layout.group($0, mainLayers: c.mainLayers) == .gateBias })
        }
        if g == .gateBias {
            // vLLM creates e_score_correction_bias only for non-hash layers (nvidia/model.py:306-310); ours
            // creates it on every layer and tolerates its absence via updateMissing.
            ConformanceGap.knownIssue("DeepseekV4Gate has e_score_correction_bias on hash layers; the real checkpoint has none") {
                ConformanceGap.expect(keys.isEmpty, "\(msg)")
            }
        } else {
            #expect(keys.isEmpty, "unexplained module-only parameters: \(msg)")
        }
    }

    @Test("sanitize keeps every real tensor of the layers it builds", arguments: [ConformanceSynthV4Layout.Group.compressor, .indexer, .tid2eid])
    func sanitizeDropsRealTensors(_ g: ConformanceSynthV4Layout.Group) async throws {
        let c = try await ConformanceSynthV4Case.make()
        let sanitized = c.model.sanitize(weights: c.weights(c.converted))
        let kept = Set(sanitized.keys)
        // only layers our model builds: the extra-layer drop is a separate gap (extraMainLayer).
        let dropped = c.converted.map(\.key).filter {
            ConformanceSynthV4Layout.group($0, mainLayers: c.mainLayers) == g && (ConformanceSynthV4Layout.layerIndex($0) ?? 99) < c.mainLayers
                && !kept.contains($0)
        }.sorted()
        // Non-vacuity (outside the wrapper): the group has real tensors in the layers we build, sanitize returned
        // tensors, and it kept ordinary ones (so `dropped` is a real subset test, not an empty-input artefact).
        let inGroup = c.converted.map(\.key).filter {
            ConformanceSynthV4Layout.group($0, mainLayers: c.mainLayers) == g
                && (ConformanceSynthV4Layout.layerIndex($0) ?? 99) < c.mainLayers
        }
        #expect(!inGroup.isEmpty, "\(g.rawValue): no real tensors in the built layers; nothing could be dropped")
        #expect(!kept.isEmpty && kept.contains { ConformanceSynthV4Layout.group($0, mainLayers: c.mainLayers) == .other })
        ConformanceGap.knownIssue("DeepseekV4.sanitize silently drops real `\(g.rawValue)` tensors") {
            ConformanceGap.expect(dropped.isEmpty, "\(dropped.count) dropped, e.g. \(dropped.prefix(3))")
        }
    }
}
