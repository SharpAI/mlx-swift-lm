// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    lazyLoad: Bool = false
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()

    // When the repo declares its weights in an index, that index defines the model.
    // The sweep below is recursive, so without this filter any stray .safetensors in a
    // subdirectory is merged in as if it were part of the model. Repos do ship such
    // files — e.g. `optiq/mtp.safetensors`, an MTP head that is absent from
    // model.safetensors.index.json. Loading it silently changed how the model's own
    // weights were interpreted (SwiftLM issue #118: every RMSNorm weight shifted by 1,
    // producing noise with no error) and broke the VLM path outright with
    // `Unhandled keys ["mtp"]`.
    //
    // MTP add-ons stay loadable when the user asks for them via SWIFTLM_MTP_ENABLE.
    var indexedFiles: Set<String>? = nil
    if let data = try? Data(
        contentsOf: modelDirectory.appendingPathComponent("model.safetensors.index.json")),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let weightMap = json["weight_map"] as? [String: String]
    {
        indexedFiles = Set(weightMap.values)
    }

    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            if let indexedFiles {
                let relativePath = url.path.hasPrefix(modelDirectory.path + "/")
                    ? String(url.path.dropFirst(modelDirectory.path.count + 1))
                    : url.lastPathComponent
                let isIndexed =
                    indexedFiles.contains(relativePath) || indexedFiles.contains(url.lastPathComponent)
                let isRequestedMTPAddOn =
                    MTPConfig.retainMTPWeights && relativePath.lowercased().contains("mtp")
                if !isIndexed && !isRequestedMTPAddOn {
                    print("[loadWeights] skipping \(relativePath): not listed in model.safetensors.index.json")
                    continue
                }
            }
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

    // Extract weight_scale_inv for switch_mlp layers BEFORE update to avoid Unhandled Keys
    var stackedScales = [String: MLXArray]()
    for key in weights.keys {
        if key.contains(".switch_mlp.") && key.hasSuffix(".weight_scale_inv") {
            if let val = weights[key] {
                stackedScales[key] = val
                weights.removeValue(forKey: key)
            }
        }
    }

    // apply the loaded weights
    // When SSD streaming is active, expert weights are intentionally absent from `weights`
    // (they are paged from NVMe on demand). Using .all would reject the load.
    // .noUnusedKeys still catches genuinely stray/misspelled keys without requiring
    // every @ModuleInfo slot to be populated up-front.
    let parameters = ModuleParameters.unflattened(weights)
    if ExpertStreamingConfig.shared.isEnabled {
        // Expert weights are intentionally absent — paged from SSD on demand.
        // .noUnusedKeys still rejects stray/misspelled keys without requiring
        // every @ModuleInfo slot to be pre-populated.
        try model.update(parameters: parameters, verify: .noUnusedKeys)
    } else {
        try model.update(parameters: parameters, verify: .all)
    }

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
            if let sl = module as? SwitchLinear {
                let bareName = "\(path).weight"
                
                // First, check for unstacked format (e.g. Qwen FP8: "experts.N.gate_proj")
                if bareName.contains(".switch_mlp.") {
                    let unstackedBaseName = bareName.replacingOccurrences(of: ".switch_mlp.", with: ".experts.")
                    let expert0Name = unstackedBaseName.replacingOccurrences(of: ".experts.", with: ".experts.0.")
                    var stripped0Name = expert0Name.replacingOccurrences(of: "language_model.model.", with: "")
                    stripped0Name = stripped0Name.replacingOccurrences(of: "language_model.", with: "")
                    stripped0Name = stripped0Name.replacingOccurrences(of: "model.", with: "")
                    let strippedMtpName = stripped0Name.replacingOccurrences(of: ".mtp.0.", with: ".mtp.")
                    
                    let allPrefixes = ["", "model.", "language_model.", "model.language_model."]
                    let candidates = [expert0Name, stripped0Name, strippedMtpName] + allPrefixes.map { $0 + stripped0Name } + allPrefixes.map { $0 + strippedMtpName }
                    var foundUnstacked = false
                    var matchedCandidate = ""
                    
                    for candidate in candidates {
                        if ExpertStreamerManager.shared?.getFile(for: candidate) != nil {
                            foundUnstacked = true
                            matchedCandidate = candidate
                            var map = [Int: (path: String, tensorName: String)]()
                            for i in 0 ..< sl.numExperts {
                                let c = candidate.replacingOccurrences(of: ".experts.0.", with: ".experts.\(i).")
                                if let file = ExpertStreamerManager.shared?.getFile(for: c),
                                   let dir = ExpertStreamingConfig.shared.modelDirectory {
                                    map[i] = (dir.appendingPathComponent(file).path, c)
                                }
                            }
                            sl.unstackedSSDMap = map
                            
                            break
                        }
                    }
                    
                    // ALWAYS check if we have a stacked scale tensor for switch_mlp
                    let scaleKey = path + ".weight_scale_inv"

                    if let scaleTensor = stackedScales[scaleKey] {

                        if !foundUnstacked {
                            print("[Load] WARNING: foundUnstacked is FALSE for \(scaleKey)!!! Forcing weightScaleInv.")
                        }
                        sl.weightScaleInv = scaleTensor
                    }
                    
                    if foundUnstacked { continue }
                }

                // Normal stacked format
                var strippedBareName = bareName.replacingOccurrences(of: "language_model.model.", with: "")
                strippedBareName = strippedBareName.replacingOccurrences(of: "language_model.", with: "")
                strippedBareName = strippedBareName.replacingOccurrences(of: "model.", with: "")
                let strippedMtpBareName = strippedBareName.replacingOccurrences(of: ".mtp.0.", with: ".mtp.")
                
                let allPrefixes = ["", "model.", "language_model.", "model.language_model."]
                let normalCandidates = [bareName, strippedBareName, strippedMtpBareName] + allPrefixes.map { $0 + strippedBareName } + allPrefixes.map { $0 + strippedMtpBareName }
                
                let originalKey = normalCandidates
                    .first { ExpertStreamerManager.shared?.getFile(for: $0) != nil }
                    ?? bareName  // fallback: use bare name
                sl.tensorName = originalKey
            }
        }
    }

    if !lazyLoad {
        eval(model)
    }
}
