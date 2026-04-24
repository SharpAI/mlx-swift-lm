#!/usr/bin/env swift-frontend -parse-as-library
// Standalone smoke test for DeepSeek V4 — run directly or via `swift scripts/test_deepseek_v4.swift`
// Usage: cd /Users/simba/SwiftLM/mlx-swift-lm && swift run test_deepseek_v4 (if added as target)
// Or build the package and run the test binary directly.
//
// This script imports the model and runs a forward pass.
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

print("=== DeepSeek V4 Smoke Test ===")

let modelPath = URL(fileURLWithPath: NSHomeDirectory())
    .appendingPathComponent("models/deepseek-v4-flash")

// 1. Check files
let shard1 = modelPath.appendingPathComponent("model-00001-of-00028.safetensors")
guard FileManager.default.fileExists(atPath: shard1.path) else {
    print("SKIP: shard1 not found at \(shard1.path)")
    exit(0)
}
print("✓ Model shards found at \(modelPath.path)")

// 2. Load config
let configPath = modelPath.appendingPathComponent("config.json")
let configData = try Data(contentsOf: configPath)
let config = try JSONDecoder().decode(DeepseekV4Configuration.self, from: configData)
let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
print("✓ Config decoded: vocab=\(config.vocabSize), layers=\(config.numHiddenLayers), mtp=\(config.numNextnPredictLayers)")

// 3. Build model
print("Building model graph...")
let model = DeepseekV4Model(config)
print("✓ Model graph built")

// 4. Load weights
print("Loading weights (this takes a while on 64GB with 126GB model)...")
let start = Date()
try loadWeights(
    modelDirectory: modelPath,
    model: model,
    perLayerQuantization: baseConfig.perLayerQuantization)
let loadTime = Date().timeIntervalSince(start)
print(String(format: "✓ Weights loaded in %.1fs", loadTime))

// 5. Forward pass
print("Running forward pass with 10 tokens...")
let tokens = MLXArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])[.newAxis, .ellipsis]
let fwdStart = Date()
let logits = model(tokens, cache: nil)
eval(logits)
let fwdTime = Date().timeIntervalSince(fwdStart)
print(String(format: "✓ Forward pass complete in %.1fs", fwdTime))
print("✓ Logits shape: \(logits.shape)  (expected [1, 10, \(config.vocabSize)])")

if logits.shape == [1, 10, config.vocabSize] {
    print("\n=== PASS ===")
} else {
    print("\n=== FAIL: shape mismatch ===")
    exit(1)
}
