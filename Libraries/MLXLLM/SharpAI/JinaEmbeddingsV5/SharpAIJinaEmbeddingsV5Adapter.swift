// SharpAI fork overlay: jina_embeddings_v5 task adapter merge. Not part of upstream.
// Follows vLLM jina.py (_load_adapter, _build_lora_pairs, _load_jina_v5_weights).
// The merge is done on raw base weights, so PyTorch [out, in] layout is kept and
// no transpose is applied (unlike LoRAContainer.fromPEFT, which targets MLX layers).

import Foundation
import MLX

public enum SharpAIJinaEmbeddingsV5Error: Error, Equatable {
    case unsupportedTask(String)
    /// The adapter was found but none of its LoRA pairs matched a base weight, so merging
    /// would silently leave the base model unchanged.
    case noMatchingAdapterPairs(task: String)
    /// The nano variant (`is_decoder: false`) is a bidirectional encoder whose pooling path is
    /// not wired, so it is refused at load time instead of failing at first use.
    case nanoVariantNotWired
}

public enum SharpAIJinaEmbeddingsV5Adapter {

    public static let supportedTasks = ["retrieval", "text-matching", "clustering", "classification"]
    public static let defaultTask = "retrieval"

    public struct AdapterConfig: Decodable, Sendable {
        public let r: Int
        public let loraAlpha: Float

        enum CodingKeys: String, CodingKey {
            case r
            case loraAlpha = "lora_alpha"
        }
    }

    /// Whether `root` ships task adapters at all (`adapters/` exists). A base checkpoint
    /// without them is used as-is.
    public static func hasAdapters(root: URL) -> Bool {
        var isDirectory: ObjCBool = false
        let path = root.appending(component: "adapters").path
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    /// Loads `adapters/<task>` from `root`, then returns `weights` with the adapter merged in.
    /// Throws when the adapter matches no base weight.
    public static func applyTaskAdapter(
        root: URL, task: String, to weights: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        guard supportedTasks.contains(task) else {
            throw SharpAIJinaEmbeddingsV5Error.unsupportedTask(task)
        }
        let directory = root.appending(component: "adapters").appending(component: task)

        let configData = try Data(
            contentsOf: directory.appending(component: "adapter_config.json"))
        let config = try JSONDecoder().decode(AdapterConfig.self, from: configData)
        let tensors = try MLX.loadArrays(
            url: directory.appending(component: "adapter_model.safetensors"))

        let (merged, matched) = mergeCounting(weights, config: config, adapterTensors: tensors)
        guard matched > 0 else {
            throw SharpAIJinaEmbeddingsV5Error.noMatchingAdapterPairs(task: task)
        }
        return merged
    }

    /// Applies `W += (B @ A) * (lora_alpha / r)` to each base weight that has a LoRA pair.
    /// A base key `model.X.weight` matches the pair keyed `X.weight`, as in vLLM.
    public static func merge(
        _ weights: [String: MLXArray],
        config: AdapterConfig,
        adapterTensors: [String: MLXArray]
    ) -> [String: MLXArray] {
        mergeCounting(weights, config: config, adapterTensors: adapterTensors).weights
    }

    private static func mergeCounting(
        _ weights: [String: MLXArray],
        config: AdapterConfig,
        adapterTensors: [String: MLXArray]
    ) -> (weights: [String: MLXArray], matched: Int) {
        let pairs = loraPairs(adapterTensors)
        let scaling = config.loraAlpha / Float(config.r)

        var merged = weights
        var matched = 0
        for (key, value) in weights {
            let inner = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
            guard let pair = pairs[inner], let a = pair.a, let b = pair.b else { continue }
            let delta = (MLX.matmul(b, a) * MLXArray(scaling)).asType(value.dtype)
            merged[key] = value + delta
            matched += 1
        }
        return (merged, matched)
    }

    private struct LoRAPair {
        var a: MLXArray?
        var b: MLXArray?
    }

    private static func loraPairs(_ tensors: [String: MLXArray]) -> [String: LoRAPair] {
        var pairs: [String: LoRAPair] = [:]
        for (key, tensor) in tensors {
            var clean = key
            if clean.hasPrefix("base_model.model.") {
                clean = String(clean.dropFirst("base_model.model.".count))
            }
            if let range = clean.range(of: ".lora_A.") {
                let base = String(clean[..<range.lowerBound]) + ".weight"
                pairs[base, default: LoRAPair()].a = tensor
            } else if let range = clean.range(of: ".lora_B.") {
                let base = String(clean[..<range.lowerBound]) + ".weight"
                pairs[base, default: LoRAPair()].b = tensor
            }
        }
        return pairs
    }
}
