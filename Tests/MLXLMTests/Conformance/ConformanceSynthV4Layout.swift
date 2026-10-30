// Conformance harness, round 3 (SL-29 phase 2): shared helpers for the synthetic-checkpoint tests that
// start from a REAL offline config fixture, plus the deepseek_v4 checkpoint key layout.
//
// * `ConformanceSynthV4Fixture` shrinks a fixture in scripts/conformance/offline/fixtures to tiny dimensions while
//   keeping its full key set (only values change), so the Configuration under test sees every key the
//   real checkpoint's config carries.
// * `ConformanceSynthV4Layout` lists the tensors the REAL DeepSeek-V4 checkpoint contains, derived from vLLM's
//   loader (snapshot dc45a5f, vllm/models/deepseek_v4/...), NOT from our model:
//     - parameter declarations:  nvidia/model.py:128-330 (MLP, MoE gate, hash table),
//       :587-665 (decoder layer: attn_norm / ffn_norm / hc_attn_* / hc_ffn_*), :823-915 (hc_head_*),
//       attention.py:238-290 (wq_a, wkv, q_norm, wq_b, kv_norm, wo_a, wo_b, attn_sink),
//       attention.py:366-380 (compressor for compress_ratio > 1), :380-400 (indexer, ratio == 4),
//       attention.py:960-1010 + compressor.py:262-280 (indexer wq_b / weights_proj, compressor ape / wkv / wgate / norm)
//     - checkpoint -> parameter names: nvidia/model.py:1064-1160 (stacked w1/w3, wq_a+wkv, wkv+wgate) and
//       :1240-1290 (`_make_deepseek_v4_weights_mapper`: layers./embed./norm./hc_head/head.weight, .ffn.gate.bias).
//     - MTP tensors are `mtp.<i>.*` in the checkpoint (nvidia/mtp.py:330-358) and are dropped by the main
//       model loader (mapper `"mtp.": None`), so they are not part of the main-model layout below.
//   Quantization scales (`.scale`) are not generated: the synthetic checkpoint is unquantized.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm (compressor / indexer / tid2eid / layer count / key names).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

enum ConformanceSynthV4Fixture {
    /// All fixture files of a `fixtures/<dir>/` directory, sorted by name.
    static func files(_ dir: String) throws -> [URL] {
        let d = ConformanceBundle.root.appendingPathComponent("fixtures/\(dir)")
        return try FileManager.default.contentsOfDirectory(at: d, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// The fixture as a JSON object (first file of the directory unless `file` names one).
    static func object(_ dir: String, file: String? = nil) throws -> [String: Any] {
        let url = try file.map { ConformanceBundle.root.appendingPathComponent("fixtures/\(dir)/\($0)") }
            ?? #require(files(dir).first)
        return try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    /// Fixture with `overrides` applied (an `NSNull` value removes the key). Keys not mentioned keep their
    /// real value, so the key set is exactly the real config's.
    static func tiny(_ dir: String, file: String? = nil, overrides: [String: Any]) throws -> Data {
        var o = try object(dir, file: file)
        for (k, v) in overrides {
            if v is NSNull { o.removeValue(forKey: k) } else { o[k] = v }
        }
        return try JSONSerialization.data(withJSONObject: o)
    }

    static func string(_ data: Data) -> String { String(decoding: data, as: UTF8.self) }
}

enum ConformanceSynthV4Layout {
    typealias Entry = ConformanceSynth.Entry

    /// Tiny V4 config: Flash's key set, shrunk. 3 main layers + 1 MTP; layer 0 = hash MoE (ratio 0),
    /// layer 1 = compress ratio 4 (compressor + indexer), layer 2 = ratio 128 (compressor only).
    static func tinyConfig() throws -> [String: Any] {
        var o = try ConformanceSynthV4Fixture.object("deepseek_v4", file: "deepseek-ai__DeepSeek-V4-Flash.json")
        o["vocab_size"] = 64; o["hidden_size"] = 32; o["moe_intermediate_size"] = 16
        o["num_hidden_layers"] = 3; o["num_nextn_predict_layers"] = 1; o["num_hash_layers"] = 1
        o["num_attention_heads"] = 4; o["head_dim"] = 16; o["qk_rope_head_dim"] = 8
        o["q_lora_rank"] = 16; o["o_lora_rank"] = 8; o["o_groups"] = 2
        o["n_routed_experts"] = 4; o["n_shared_experts"] = 1; o["num_experts_per_tok"] = 2
        o["index_head_dim"] = 8; o["index_n_heads"] = 2; o["index_topk"] = 4
        o["sliding_window"] = 8; o["max_position_embeddings"] = 256; o["hc_mult"] = 2
        o["compress_ratios"] = [0, 4, 128, 0]
        o.removeValue(forKey: "rope_scaling")
        return o
    }

    static func data(_ o: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: o) }

    static func model(_ o: [String: Any]) async throws -> DeepseekV4Model {
        let m = try await LLMTypeRegistry.shared.createModel(
            configuration: data(o), modelType: "deepseek_v4")
        return try #require(m as? DeepseekV4Model)
    }

    private static func int(_ o: [String: Any], _ k: String) -> Int { (o[k] as? Int) ?? 0 }

    /// The tensors of the real (HF) checkpoint, main model only, as `(key, shape, dtype)`.
    static func realEntries(_ o: [String: Any]) -> [Entry] {
        let H = int(o, "hidden_size"), V = int(o, "vocab_size")
        let nh = int(o, "num_attention_heads"), hd = int(o, "head_dim")
        let qr = int(o, "q_lora_rank"), orank = int(o, "o_lora_rank"), og = int(o, "o_groups")
        let E = int(o, "n_routed_experts"), topk = int(o, "num_experts_per_tok")
        let I = int(o, "moe_intermediate_size"), nShared = int(o, "n_shared_experts")
        let hc = int(o, "hc_mult"), mix = (2 + hc) * hc
        let nHash = int(o, "num_hash_layers"), L = int(o, "num_hidden_layers")
        let ixHD = int(o, "index_head_dim"), ixN = int(o, "index_n_heads")
        let ratios = (o["compress_ratios"] as? [Int]) ?? []
        let noaux = (o["topk_method"] as? String) == "noaux_tc"

        var out: [Entry] = []
        func add(_ k: String, _ shape: [Int], _ dt: DType = .float32) {
            out.append(Entry(key: k, shape: shape, dtype: dt))
        }
        func compressor(_ p: String, headDim: Int, ratio: Int) {
            let coff = ratio == 4 ? 2 : 1  // compressor.py:~250 (`overlap = compress_ratio == 4`)
            add("\(p).ape", [ratio, coff * headDim])
            add("\(p).wkv.weight", [coff * headDim, H])
            add("\(p).wgate.weight", [coff * headDim, H])
            add("\(p).norm.weight", [headDim])
        }

        add("embed.weight", [V, H])
        add("head.weight", [V, H])
        add("norm.weight", [H])
        add("hc_head_fn", [hc, hc * H]); add("hc_head_base", [hc]); add("hc_head_scale", [1])

        for l in 0 ..< L {
            let p = "layers.\(l)"
            let ratio = max(1, l < ratios.count ? ratios[l] : 0)
            add("\(p).attn_norm.weight", [H]); add("\(p).ffn_norm.weight", [H])
            for n in ["attn", "ffn"] {
                add("\(p).hc_\(n)_fn", [mix, hc * H])
                add("\(p).hc_\(n)_base", [mix])
                add("\(p).hc_\(n)_scale", [3])
            }
            add("\(p).attn.attn_sink", [nh])
            add("\(p).attn.wq_a.weight", [qr, H])
            add("\(p).attn.q_norm.weight", [qr])
            add("\(p).attn.wq_b.weight", [nh * hd, qr])
            add("\(p).attn.wkv.weight", [hd, H])
            add("\(p).attn.kv_norm.weight", [hd])
            add("\(p).attn.wo_a.weight", [og * orank, nh * hd / og])
            add("\(p).attn.wo_b.weight", [H, og * orank])
            if ratio > 1 { compressor("\(p).attn.compressor", headDim: hd, ratio: ratio) }
            if ratio == 4 {
                add("\(p).attn.indexer.wq_b.weight", [ixHD * ixN, qr])
                add("\(p).attn.indexer.weights_proj.weight", [ixN, H])
                compressor("\(p).attn.indexer.compressor", headDim: ixHD, ratio: ratio)
            }
            add("\(p).ffn.gate.weight", [E, H])
            if l < nHash {
                add("\(p).ffn.gate.tid2eid", [V, topk], .int32)
            } else if noaux {
                add("\(p).ffn.gate.bias", [E])
            }
            for e in 0 ..< E {
                add("\(p).ffn.experts.\(e).w1.weight", [I, H])
                add("\(p).ffn.experts.\(e).w2.weight", [H, I])
                add("\(p).ffn.experts.\(e).w3.weight", [I, H])
            }
            if nShared > 0 {
                add("\(p).ffn.shared_experts.w1.weight", [I * nShared, H])
                add("\(p).ffn.shared_experts.w2.weight", [H, I * nShared])
                add("\(p).ffn.shared_experts.w3.weight", [I * nShared, H])
            }
        }
        return out.sorted { $0.key < $1.key }
    }

    /// Rename a real key into the naming an mlx-style converted checkpoint (and our module) uses.
    /// This is the vLLM mapper (model.py:1240-1290) plus the w1/w2/w3 -> gate/down/up and
    /// hc_*_{fn,base,scale} -> hc_*.{fn,base,scale} renames an mlx converter performs. It is the ONLY
    /// translation layer the tests assume; DeepseekV4.sanitize has none of it.
    static func mlxName(_ k: String) -> String {
        var s = k
        if s == "embed.weight" { return "model.embed_tokens.weight" }
        if s == "norm.weight" { return "model.norm.weight" }
        if s == "head.weight" { return "lm_head.weight" }
        if s.hasPrefix("hc_head_") { return "model.hc_head." + s.dropFirst("hc_head_".count) }
        if s.hasPrefix("layers.") { s = "model." + s }
        for n in ["attn", "ffn"] {
            for f in ["fn", "base", "scale"] {
                s = s.replacingOccurrences(of: ".hc_\(n)_\(f)", with: ".hc_\(n).\(f)")
            }
        }
        for (a, b) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
            s = s.replacingOccurrences(
                of: #"\.(experts\.\d+|shared_experts)\.\#(a)\."#, with: ".$1.\(b).",
                options: .regularExpression)
        }
        if s.hasSuffix(".ffn.gate.bias") {
            s = String(s.dropLast("bias".count)) + "e_score_correction_bias"
        }
        return s
    }

    /// Per-expert tensors `...ffn.experts.<e>.<proj>.weight [a, b]` -> the stacked `...ffn.switch_mlp.<proj>.weight
    /// [E, a, b]` that the module holds (what a loader's expert-stacking step produces). Used so the
    /// key comparison is not dominated by the (expected, handled) per-expert vs stacked difference.
    static func stackExperts(_ entries: [Entry]) -> [Entry] {
        var groups: [String: (count: Int, e: Entry)] = [:]
        var out: [Entry] = []
        for e in entries {
            if let r = e.key.range(of: #"\.ffn\.experts\.\d+\."#, options: .regularExpression) {
                let key = e.key.replacingCharacters(in: r, with: ".ffn.switch_mlp.")
                let g = groups[key] ?? (0, Entry(key: key, shape: [0] + e.shape, dtype: e.dtype))
                groups[key] = (g.count + 1, g.e)
            } else {
                out.append(e)
            }
        }
        for (_, g) in groups {
            out.append(Entry(key: g.e.key, shape: [g.count] + g.e.shape.dropFirst(), dtype: g.e.dtype))
        }
        return out.sorted { $0.key < $1.key }
    }

    // MARK: classification of a key difference

    enum Group: String, CaseIterable {
        case compressor, indexer, tid2eid, extraMainLayer, gateBias, other
    }

    /// `mainLayers` = how many transformer layers OUR model builds.
    static func group(_ k: String, mainLayers: Int) -> Group {
        if let layer = layerIndex(k), layer >= mainLayers { return .extraMainLayer }
        if k.contains(".attn.indexer.") { return .indexer }
        if k.contains(".attn.compressor.") { return .compressor }
        if k.contains(".ffn.gate.tid2eid") { return .tid2eid }
        if k.hasSuffix(".ffn.gate.e_score_correction_bias") { return .gateBias }
        return .other
    }

    static func layerIndex(_ k: String) -> Int? {
        let parts = k.split(separator: ".")
        guard parts.count > 2, parts[0] == "model", parts[1] == "layers" else { return nil }
        return Int(parts[2])
    }
}
