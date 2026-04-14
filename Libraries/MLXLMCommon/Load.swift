// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    lazyLoad: Bool = false
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // ExpertStreamingConfig: Initialize the ExpertStreamerManager when streaming is active.
    // On macOS: pread() from NVMe at ~5 GB/s.
    // On iOS:   mmap page-cache from APFS at ~2-3 GB/s — same struct, different bandwidth.
    if ExpertStreamingConfig.shared.isEnabled {
        ExpertStreamerManager.shared = ExpertStreamerManager(modelDirectory: modelDirectory)
    }

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    let dict = perLayerQuantization.perLayerQuantization
                    if let opt = dict[path] ?? 
                                 dict["language_model.\(path)"] ??
                                 dict[path.replacingOccurrences(of: ".experts.router.", with: ".router.")] ??
                                 dict["language_model." + path.replacingOccurrences(of: ".experts.router.", with: ".router.")] {
                        switch opt {
                        case .skip: return nil
                        case .quantize(let q): return q.asTuple
                        }
                    }
                    return perLayerQuantization.quantization?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    if ExpertStreamingConfig.shared.isEnabled {
        // Assign tensorName to each QuantizedSwitchLinear.
        //
        // CRITICAL: tensorName must be the ORIGINAL key in the safetensors shard
        // (before sanitize() strips VLM wrapper prefixes like "language_model."),
        // because BOTH ExpertStreamerManager.getFile() and the C++ streamedGatherMM
        // pread() use this key to locate the tensor bytes within the shard file.
        //
        // Example for Mistral4:
        //   post-sanitize path → "model.layers.0.mlp.switch_mlp.gate_proj"
        //   original shard key → "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
        //
        // We probe the ExpertStreamerManager weight map with common VLM prefixes
        // and fall back to the bare path if none match.
        let knownPrefixes = ["language_model.", "model.language_model.", ""]
        for (path, module) in model.leafModules().flattened() {
            if let qsl = module as? QuantizedSwitchLinear {
                let bareName = "\(path).weight"
                // Find the original key that exists in the shard index
                let originalKey = knownPrefixes.lazy
                    .map { $0 + bareName }
                    .first { ExpertStreamerManager.shared?.getFile(for: $0) != nil }
                    ?? bareName  // fallback: use bare name (works when model has no VLM wrapper)
                qsl.tensorName = originalKey
            }
        }
    }

    if !lazyLoad {
        eval(model)
    }
}
