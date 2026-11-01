// SharpAI fork overlay: moondream3 checkpoint key sanitization. Not part of upstream.
//
// Most keys already match the module tree. Property names and @ModuleInfo keys were chosen to
// match the HF names (`attn.qkv`, `attn.tau.wq`, `mlp.router`, ...). Only two rules are left:
// drop the `model.` prefix, add `.weight` to the bare `text.wte` embedding table, and drop
// `region.*` keys the Swift tree does not build.
// UNVERIFIED: the `model.` prefix comes from vLLM's load_weights, not from a checkpoint index.

import Foundation
import MLX

public enum SharpAIMoondream3Sanitizer {

    static func remappedName(_ key: String) -> String {
        var name = key
        if name.hasPrefix("model.") {
            name = String(name.dropFirst("model.".count))
        }
        if name == "text.wte" {
            return "text.wte.weight"
        }
        return name
    }

    /// Module prefixes the Swift tree does not build. Dropped, as vLLM's load_weights skips them.
    /// UNVERIFIED: the `region.` prefix is taken from the vLLM reference, not a checkpoint index.
    static let unbuiltPrefixes = ["region."]

    static func isUnbuilt(_ name: String) -> Bool {
        unbuiltPrefixes.contains { name.hasPrefix($0) }
    }

    /// Renames checkpoint keys and drops unbuilt ones. Throws if two keys map to the same name.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        for (key, value) in weights {
            let name = remappedName(key)
            if isUnbuilt(name) {
                continue
            }
            guard result[name] == nil else {
                throw SharpAIMoondream3Error.duplicateWeightKey(name)
            }
            result[name] = value
        }
        return result
    }
}
