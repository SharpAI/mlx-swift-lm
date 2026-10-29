// Conformance gap pins, part 2: deepseek_v4 (registered, but #254 says the text path is incomplete).
//
// Tier 0/1 facts only: config JSON from the offline fixtures (scripts/conformance/offline/fixtures/
// deepseek_v4*), no weights, no network. Green tests record facts; every gap is a precise test
// wrapped in `withKnownIssue(isIntermittent: true)` so it is green before AND after the fix and an
// upstream sync that implements it is never blocked. Delete the wrapper in the PR that fixes it.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm (compressor / indexer / tid2eid / layer count).
//
// Evidence for each spec comes from the vLLM snapshot dc45a5f (vllm/models/deepseek_v4/*):
//   * MTP layers live at model.layers.{num_hidden_layers + i} (nvidia/mtp.py:217,352; quant_config.py:53),
//     and compress_ratios has num_hidden_layers + num_nextn_predict_layers entries (fixtures below).
//   * compressor exists for compress_ratio > 1; the indexer only for compress_ratio == 4 (attention.py:302,369).
//   * hash-MoE gate (gate.tid2eid [vocab, topk]) exists for layer index < num_hash_layers (nvidia/model.py:286-300).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

enum ConformanceDeepseekV4 {
    static func fixture(_ name: String) throws -> Data {
        try Data(
            contentsOf: ConformanceBundle.root.appendingPathComponent(
                "fixtures/deepseek_v4/deepseek-ai__\(name).json"))
    }

    static func flashObject() throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: fixture("DeepSeek-V4-Flash")) as? [String: Any])
    }

    /// Flash's real key set, with only the dimensions shrunk so the model can be built.
    /// 3 main layers + 1 MTP; layer 0 = hash MoE, layer 1 = compress ratio 4, layer 2 = ratio 128.
    static func tinyJSON() throws -> Data {
        var o = try flashObject()
        o["vocab_size"] = 64; o["hidden_size"] = 32; o["moe_intermediate_size"] = 16
        o["num_hidden_layers"] = 3; o["num_nextn_predict_layers"] = 1; o["num_hash_layers"] = 1
        o["num_attention_heads"] = 2; o["head_dim"] = 16; o["qk_rope_head_dim"] = 8
        o["q_lora_rank"] = 16; o["o_lora_rank"] = 8; o["o_groups"] = 2
        o["n_routed_experts"] = 4; o["n_shared_experts"] = 1; o["num_experts_per_tok"] = 2
        o["index_head_dim"] = 8; o["index_n_heads"] = 2; o["index_topk"] = 4
        o["sliding_window"] = 8; o["max_position_embeddings"] = 256; o["hc_mult"] = 2
        o["compress_ratios"] = [0, 4, 128, 0]
        o["rope_scaling"] = NSNull()
        o.removeValue(forKey: "rope_scaling")
        return try JSONSerialization.data(withJSONObject: o)
    }

    static func tinyModel() async throws -> DeepseekV4Model {
        let data = try tinyJSON()
        let m = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "deepseek_v4")
        return try #require(m as? DeepseekV4Model)
    }

    static func paramKeys(_ m: Module) -> [String] { m.parameters().flattened().map { $0.0 } }

    /// Keys present in the fixture that the Configuration struct does NOT round-trip, i.e. keys it
    /// silently ignores. Uses the Codable round trip of the real struct (CodingKeys are private).
    static func decodedKeys(_ data: Data) throws -> Set<String> {
        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        let enc = try JSONSerialization.jsonObject(with: JSONEncoder().encode(cfg)) as? [String: Any]
        return Set(try #require(enc).keys)
    }
}

struct ConformanceGapDeepseekV4Tests {

    // MARK: green facts

    @Test("V4 fixtures: compress_ratios has num_hidden_layers + num_nextn_predict_layers entries")
    func compressRatiosLengthFact() throws {
        // Flash 43+1, Pro 61+1, Vision-Exp 43+3, Pro-DSpark 61+1 (+2 dspark entries, see below).
        let expect: [(String, Int)] = [
            ("DeepSeek-V4-Flash", 44), ("DeepSeek-V4-Pro", 62), ("DeepSeek-V4-Flash-Vision-Exp", 46),
        ]
        for (name, n) in expect {
            let o = try #require(try JSONSerialization.jsonObject(with: ConformanceDeepseekV4.fixture(name)) as? [String: Any])
            let hidden = try #require(o["num_hidden_layers"] as? Int)
            let nextn = try #require(o["num_nextn_predict_layers"] as? Int)
            let ratios = try #require(o["compress_ratios"] as? [Int])
            #expect(ratios.count == n && ratios.count == hidden + nextn, "\(name): \(ratios.count) vs \(hidden)+\(nextn)")
        }
    }

    @Test("V4-Flash fixture decodes with the registry's configuration type")
    func flashDecodes() async throws {
        #expect(await LLMTypeRegistry.shared.contains("deepseek_v4"))
        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: ConformanceDeepseekV4.fixture("DeepSeek-V4-Flash"))
        #expect(cfg.numHiddenLayers == 43 && cfg.numNextnPredictLayers == 1 && cfg.nRoutedExperts == 256)
        #expect(cfg.compressRatios.count == 44 && cfg.numHashLayers == 3 && cfg.scoringFunc == "sqrtsoftplus")
    }

    @Test("the tiny reduced V4 config builds and a forward pass is finite")
    func tinyBuildsAndRuns() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        eval(m)
        let out = m(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(out)
        #expect(out.shape == [1, 3, 64])
        #expect(out.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    // MARK: known gaps -- keys the Configuration silently ignores

    /// Why each key matters (all from the V4 fixtures + vLLM source, see header):
    ///   index_head_dim / index_n_heads / index_topk -> sparse indexer shape (ratio-4 layers)
    ///   expert_dtype                                -> "fp4" selects the MXFP4 expert loader
    ///   quantization_config                         -> scale_fmt ue8m0 + weight_block_size; sanitize hardcodes 128 / non-ue8m0
    static let ignoredRequired = ["index_head_dim", "index_n_heads", "index_topk", "expert_dtype", "quantization_config"]

    @Test("V4 Configuration decodes the key", arguments: ignoredRequired)
    func configurationDecodesKey(_ key: String) throws {
        let data = try ConformanceDeepseekV4.fixture("DeepSeek-V4-Flash")
        let fixtureKeys = Set(try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]).keys)
        #expect(fixtureKeys.contains(key), "fixture must carry \(key)")
        let decoded = try ConformanceDeepseekV4.decodedKeys(data)
        ConformanceGap.knownIssue("DeepseekV4Configuration silently ignores `\(key)`") {
            #expect(decoded.contains(key))
        }
    }

    // MARK: known gaps -- structure

    @Test("V4 builds num_hidden_layers main layers (the MTP block is index num_hidden_layers, not inside it)")
    func mainLayerCount() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        // tiny config: num_hidden_layers = 3, num_nextn_predict_layers = 1.
        let built = m.model.layers.count
        ConformanceGap.knownIssue("DeepseekV4 builds num_hidden_layers - num_nextn_predict_layers main layers") {
            #expect(built == 3, "built \(built) layers, spec says 3 (compress_ratios has 4 = 3 + 1 MTP)")
        }
    }

    @Test("V4 attention layers with compress_ratio > 1 own a compressor; ratio-4 layers also an indexer")
    func compressorAndIndexerModules() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        let keys = ConformanceDeepseekV4.paramKeys(m)
        // layer 1 has ratio 4 (compressor + indexer), layer 2 has ratio 128 (compressor only).
        // Layers 1 and 2 are only built once the layer-count gap above is fixed, so key off layer 1 only
        // (it exists today) for the indexer and either layer for the compressor.
        ConformanceGap.knownIssue("DeepseekV4 has no compressor module (sanitize drops .attn.compressor.*)") {
            #expect(keys.contains { $0.hasPrefix("model.layers.1.attn.compressor.") })
        }
        ConformanceGap.knownIssue("DeepseekV4 has no indexer module (sanitize drops .attn.indexer.*)") {
            #expect(keys.contains { $0.hasPrefix("model.layers.1.attn.indexer.") })
        }
    }

    @Test("V4 hash-MoE layers (index < num_hash_layers) own a gate.tid2eid [vocab, top-k] table")
    func hashRoutingTable() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        let params = Dictionary(uniqueKeysWithValues: m.parameters().flattened())
        let t = params["model.layers.0.ffn.gate.tid2eid"]
        ConformanceGap.knownIssue("DeepseekV4 has no tid2eid parameter (sanitize drops it)") {
            #expect(t != nil)
            #expect(t?.shape == [64, 2])
        }
    }

    @Test("V4 sanitize keeps the compressor / indexer / tid2eid tensors of a main layer")
    func sanitizeKeepsTensors() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        let probe = ["model.layers.0.attn.compressor.wkv.weight", "model.layers.0.attn.indexer.wq_b.weight",
                     "model.layers.0.ffn.gate.tid2eid"]
        let out = m.sanitize(weights: Dictionary(uniqueKeysWithValues: probe.map { ($0, MLXArray.zeros([2, 2])) }))
        for k in probe {
            ConformanceGap.knownIssue("DeepseekV4.sanitize drops \(k)") {
                #expect(out[k] != nil)
            }
        }
    }
}
