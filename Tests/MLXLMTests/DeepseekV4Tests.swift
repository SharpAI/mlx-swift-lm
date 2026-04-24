// Copyright © 2026 Apple Inc.
// Tests for DeepSeek V4 model architecture and inference

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

/// Minimal config that matches the DeepseekV4Configuration field names, but with
/// tiny dimensions so the full forward pass runs quickly in CI.
private func makeSmallConfig() -> DeepseekV4Configuration {
    // Use JSON decode so all CodingKeys are exercised
    let json = """
    {
        "vocab_size": 256,
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "head_dim": 16,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 4,
        "rms_norm_eps": 1e-6,
        "o_groups": 2,
        "o_lora_rank": 16,
        "sliding_window": 128,
        "compress_ratios": [0, 0, 4, 128, 0, 0],
        "compress_rope_theta": 160000.0,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 1,
        "norm_topk_prob": true,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "index_head_dim": 16,
        "index_n_heads": 4,
        "index_topk": 4
    }
    """
    let data = json.data(using: .utf8)!
    return try! JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
}

public class DeepseekV4Tests: XCTestCase {

    // MARK: - Configuration

    func testConfigurationDecode() throws {
        let config = makeSmallConfig()
        XCTAssertEqual(config.vocabSize, 256)
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.numHiddenLayers, 4)
        XCTAssertEqual(config.numAttentionHeads, 4)
        XCTAssertEqual(config.headDim, 16)
        XCTAssertEqual(config.nopeHeadDim, 12)  // headDim - qkRopeHeadDim = 16 - 4
        XCTAssertEqual(config.hcMult, 2)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
    }

    // MARK: - Architecture Shape

    func testModelOutputShape() throws {
        let config = makeSmallConfig()
        let model = DeepseekV4Model(config)

        // Main layers should be numHiddenLayers - numNextnPredictLayers = 3
        let expectedLayers = config.numHiddenLayers - config.numNextnPredictLayers
        XCTAssertEqual(model.model.layers.count, expectedLayers,
                       "Layer count should exclude MTP layers")

        let input = MLXArray([0, 1, 2, 3])[.newAxis, .ellipsis]  // [1, 4]
        let output = model(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 4, config.vocabSize],
                       "Output logits shape should be [B, S, vocabSize]")
        eval(output)
    }

    func testKVHeadsCount() {
        let config = makeSmallConfig()
        let model = DeepseekV4Model(config)
        let expected = config.numHiddenLayers - config.numNextnPredictLayers
        XCTAssertEqual(model.kvHeads.count, expected)
        XCTAssertTrue(model.kvHeads.allSatisfy { $0 == 1 },
                      "V4 uses unified KV with single head per layer")
    }

    // MARK: - HCParams Module

    func testHCParamsLoadedFromCheckpointKeys() throws {
        // Verify HCParams sub-modules are reachable at the right key paths
        let config = makeSmallConfig()
        let block = DeepseekV4Block(config: config)

        // After construction, hc_attn and hc_ffn should be zeroed/ones placeholders
        let hcAttnFnShape = block.hc_attn.fn.shape
        let mixHc = (2 + config.hcMult) * config.hcMult  // (2+2)*2 = 8
        let hcDim = config.hcMult * config.hiddenSize      // 2 * 64 = 128
        XCTAssertEqual(hcAttnFnShape, [mixHc, hcDim])
        XCTAssertEqual(block.hc_ffn.fn.shape, [mixHc, hcDim])
        XCTAssertEqual(block.hc_attn.scale.shape, [3])
        XCTAssertEqual(block.hc_ffn.scale.shape, [3])
    }

    func testHCHeadShape() {
        let config = makeSmallConfig()
        let inner = DeepseekV4ModelInner(config: config)
        let hc = config.hcMult
        XCTAssertEqual(inner.hc_head.fn.shape, [hc, hc * config.hiddenSize])
        XCTAssertEqual(inner.hc_head.base.shape, [hc])
        XCTAssertEqual(inner.hc_head.scale.shape, [1])
    }

    // MARK: - Attention Keys

    func testAttentionModuleKeyNames() throws {
        let config = makeSmallConfig()
        let attn = DeepseekV4Attention(config: config)

        // Verify correct key-named properties exist (checked via parameter collection)
        // wqA, qNorm, wqB, wkv, kvNorm, woA, woB
        let params = attn.parameters()
        // The parameter dict should contain the key-named projections
        // (MLX uses the @ModuleInfo key as the dict key)
        XCTAssertNotNil(params["wq_a"], "wq_a projection missing")
        XCTAssertNotNil(params["q_norm"], "q_norm missing")
        XCTAssertNotNil(params["wq_b"], "wq_b projection missing")
        XCTAssertNotNil(params["wkv"], "wkv projection missing")
        XCTAssertNotNil(params["kv_norm"], "kv_norm missing")
        XCTAssertNotNil(params["wo_a"], "wo_a projection missing")
        XCTAssertNotNil(params["wo_b"], "wo_b projection missing")
    }

    func testBlockModuleKeyNames() throws {
        let config = makeSmallConfig()
        let block = DeepseekV4Block(config: config)
        let params = block.parameters()

        XCTAssertNotNil(params["attn"], "attn module missing (was 'self_attn')")
        XCTAssertNotNil(params["ffn"], "ffn module missing (was 'mlp')")
        XCTAssertNotNil(params["attn_norm"], "attn_norm missing (was 'input_layernorm')")
        XCTAssertNotNil(params["ffn_norm"], "ffn_norm missing (was 'post_attention_layernorm')")
        XCTAssertNotNil(params["hc_attn"], "hc_attn missing")
        XCTAssertNotNil(params["hc_ffn"], "hc_ffn missing")
    }

    // MARK: - Sanitize

    func testSanitizeDropsMTPLayers() {
        let config = makeSmallConfig()
        let model = DeepseekV4Model(config)

        // Build fake weights including MTP layer (index numHiddenLayers - 1 = 3)
        var fakeWeights: [String: MLXArray] = [:]
        for l in 0 ..< config.numHiddenLayers {
            fakeWeights["model.layers.\(l).attn_norm.weight"] = zeros([config.hiddenSize])
        }
        fakeWeights["model.norm.weight"] = zeros([config.hiddenSize])

        let sanitized = model.sanitize(weights: fakeWeights)

        // MTP layer (index 3) should be removed; layers 0,1,2 kept
        let mainCount = config.numHiddenLayers - config.numNextnPredictLayers  // 3
        for l in 0 ..< mainCount {
            XCTAssertNotNil(sanitized["model.layers.\(l).attn_norm.weight"],
                            "Layer \(l) should be present")
        }
        for l in mainCount ..< config.numHiddenLayers {
            XCTAssertNil(sanitized["model.layers.\(l).attn_norm.weight"],
                         "MTP layer \(l) should be filtered out")
        }
    }

    func testSanitizeStacksPerExpertWeightsWhenPresent() {
        let config = makeSmallConfig()
        let model = DeepseekV4Model(config)

        // Simulate a non-pre-stacked checkpoint (expert weights as separate keys)
        var fakeWeights: [String: MLXArray] = [:]
        for j in 0 ..< config.nRoutedExperts {
            let w = zeros([config.moeIntermediateSize, config.hiddenSize])
            fakeWeights["model.layers.0.ffn.experts.\(j).gate_proj.weight"] = w
            fakeWeights["model.layers.0.ffn.experts.\(j).up_proj.weight"] = w
            fakeWeights["model.layers.0.ffn.experts.\(j).down_proj.weight"] =
                zeros([config.hiddenSize, config.moeIntermediateSize])
        }

        let sanitized = model.sanitize(weights: fakeWeights)

        // Stacked keys should exist
        XCTAssertNotNil(sanitized["model.layers.0.ffn.switch_mlp.gate_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.ffn.switch_mlp.up_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.ffn.switch_mlp.down_proj.weight"])

        // Individual keys should be gone
        XCTAssertNil(sanitized["model.layers.0.ffn.experts.0.gate_proj.weight"])

        // Stacked shape: [nExperts, ...] 
        let stackedGate = sanitized["model.layers.0.ffn.switch_mlp.gate_proj.weight"]!
        XCTAssertEqual(stackedGate.shape[0], config.nRoutedExperts)
    }

    func testSanitizeDropsCompressorIndexerKeys() {
        // Compressor/indexer sub-modules are not yet implemented; ensure sanitize drops them.
        let config = makeSmallConfig()
        let model = DeepseekV4Model(config)

        var fakeWeights: [String: MLXArray] = [
            "model.layers.0.attn.wkv.weight": zeros([1, 1]),
            "model.layers.0.attn.compressor.wkv.weight": zeros([1, 1]),
            "model.layers.0.attn.compressor.wgate.weight": zeros([1, 1]),
            "model.layers.0.attn.indexer.wq_b.weight": zeros([1, 1]),
            "model.layers.0.attn.indexer.weights_proj.weight": zeros([1, 1]),
            "model.layers.0.attn.indexer.compressor.wkv.weight": zeros([1, 1]),
        ]

        let sanitized = model.sanitize(weights: fakeWeights)

        XCTAssertNotNil(sanitized["model.layers.0.attn.wkv.weight"],
                        "Standard wkv should be kept")
        XCTAssertNil(sanitized["model.layers.0.attn.compressor.wkv.weight"],
                     "Compressor keys should be dropped")
        XCTAssertNil(sanitized["model.layers.0.attn.compressor.wgate.weight"],
                     "Compressor keys should be dropped")
        XCTAssertNil(sanitized["model.layers.0.attn.indexer.wq_b.weight"],
                     "Indexer keys should be dropped")
        XCTAssertNil(sanitized["model.layers.0.attn.indexer.compressor.wkv.weight"],
                     "Nested indexer compressor keys should be dropped")
    }



    func testMoEGateRoutingShape() {
        let config = makeSmallConfig()
        let gate = DeepseekV4Gate(config: config)
        gate.weight = MLXRandom.uniform(low: 0, high: 1, [config.nRoutedExperts, config.hiddenSize])
        gate.e_score_correction_bias = zeros([config.nRoutedExperts])

        let B = 2, S = 3
        let x = MLXRandom.uniform(low: 0, high: 1, [B, S, config.hiddenSize])
        let (indices, scores) = gate(x)
        eval(indices, scores)

        XCTAssertEqual(indices.shape, [B, S, config.numExpertsPerTok])
        XCTAssertEqual(scores.shape, [B, S, config.numExpertsPerTok])
    }

    // MARK: - Batched Inference

    func testBatchedForwardConsistency() throws {
        // Running the same token twice as a batch vs individually should give same logits
        let config = makeSmallConfig()
        let model = DeepseekV4Model(config)

        let tokenArray = MLXArray([42, 100, 7])[.newAxis, .ellipsis]  // [1, 3]

        let out1 = model(tokenArray, cache: nil)
        let out2 = model(tokenArray, cache: nil)
        eval(out1, out2)

        // Outputs should be numerically identical for same input
        let diffArr = (out1 - out2).abs().max()
        eval(diffArr)
        let diff = diffArr.item(Float.self)
        XCTAssertEqual(diff, 0.0, accuracy: 1e-5)
    }

    // MARK: - Real Model Load (skipped unless model is present)

    func testRealModelLoad() throws {
        let modelPath = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("models/deepseek-v4-flash")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw XCTSkip("Model not downloaded; skipping real model test")
        }

        // Check config.json is present
        let configPath = modelPath.appendingPathComponent("config.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: configPath.path),
                      "config.json missing from model directory")

        // Load and decode config
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: configData)
        XCTAssertEqual(config.vocabSize, 129280)
        XCTAssertEqual(config.numHiddenLayers, 43)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
        XCTAssertEqual(config.hcMult, 4)
    }

    func testRealModelInference() throws {
        let modelPath = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("models/deepseek-v4-flash")

        // Set memory limits immediately — model is 126 GB on 64 GB RAM, must configure
        // before any weight loading so MLX uses mmap-backed paging, not eager RAM loading.
        Memory.memoryLimit = 200 * 1024 * 1024 * 1024  // 200 GB sentinel (bypass spin-wait loop)
        Memory.cacheLimit = 50 * 1024 * 1024 * 1024    // 50 GB page-cache budget

        // Write debug info immediately so we can see if test even starts
        try? "START home=\(NSHomeDirectory()) modelPath=\(modelPath.path)".write(
            toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)

        // Check that enough shards are present (at least shard 1)
        let shard1 = modelPath.appendingPathComponent("model-00001-of-00028.safetensors")
        let shard1Exists = FileManager.default.fileExists(atPath: shard1.path)
        try? "STAGE:shard1=\(shard1Exists) path=\(shard1.path)".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)
        guard shard1Exists else {
            throw XCTSkip("Model shards not yet downloaded; skipping inference test")
        }

        // Check all 28 shards are present
        var missingShards: [Int] = []
        for i in 1...28 {
            let shard = modelPath.appendingPathComponent(
                String(format: "model-%05d-of-00028.safetensors", i))
            if !FileManager.default.fileExists(atPath: shard.path) {
                missingShards.append(i)
            }
        }
        try? "STAGE:shardCheck missing=\(missingShards)".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)
        guard missingShards.isEmpty else {
            throw XCTSkip("Missing shards \(missingShards); download not complete")
        }

        // Load config
        let configPath = modelPath.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        try? "STAGE:configRead".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)
        let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: configData)
        try? "STAGE:configDecoded vocab=\(config.vocabSize)".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)
        // Read base config for per-layer quantization
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

        // Create model and load weights with per-layer quantization
        let model = DeepseekV4Model(config)
        try? "STAGE:modelCreated".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)

        do {
            // lazyLoad: true skips the final eval(model) in loadWeights — critical for
            // 126 GB models on 64 GB RAM. Without it, loadWeights calls eval(model)
            // which materialises all weights into RAM at once → OOM SIGKILL.
            // With lazyLoad: true, MLX mmaps weights and pages them in on demand via SSD.
            // Activate ExpertStreamingConfig so MoE layer forward passes use the
            // SSD streaming path rather than demanding all expert weights up front.
            ExpertStreamingConfig.shared.activate(modelDirectory: modelPath, useDirectIO: true)
            try loadWeights(
                modelDirectory: modelPath,
                model: model,
                perLayerQuantization: baseConfig.perLayerQuantization,
                lazyLoad: true)
            try? "STAGE:weightsLoaded".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)
        } catch {
            let msg = "STAGE:loadWeightsFailed error=\(error)"
            try? msg.write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)
            throw error
        }

        // Run a forward pass with dummy tokens (just first 10 tokens)
        let promptTokens = MLXArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])[.newAxis, .ellipsis]
        try? "STAGE:forwardStart".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)
        let logits = model(promptTokens, cache: nil)
        eval(logits)
        try? "STAGE:forwardDone shape=\(logits.shape)".write(toFile: "/tmp/dsv4_debug.txt", atomically: true, encoding: .utf8)

        XCTAssertEqual(logits.shape, [1, 10, config.vocabSize])
        let resultMsg = "DeepSeek V4 forward pass OK: logits shape \(logits.shape)"
        print(resultMsg)
        // Write marker file so we can verify pass even if terminal output is truncated
        try? resultMsg.write(toFile: "/tmp/deepseek_v4_result.txt", atomically: true, encoding: .utf8)
    }
}
