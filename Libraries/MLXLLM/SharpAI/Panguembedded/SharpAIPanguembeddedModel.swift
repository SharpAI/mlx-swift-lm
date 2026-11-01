// SharpAI fork overlay: PanguEmbedded model. Not part of upstream mlx-swift-lm.
//
// The module tree is LlamaModel's, so checkpoint keys such as
// model.layers.N.self_attn.q_proj.bias load unchanged. This subclass only adds a
// sanitize step for an absent lm_head.

import Foundation
import MLX
import MLXLMCommon

public final class SharpAIPanguembeddedModel: LlamaModel {

    public init(_ configuration: SharpAIPanguembeddedConfiguration) {
        super.init(configuration.llamaConfiguration)
    }

    /// Adds `lm_head.*` from `model.embed_tokens.*` when an untied checkpoint omits it.
    /// This follows llama.cpp, which reuses the token embedding for a missing output tensor.
    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = super.sanitize(weights: weights)

        let embedPrefix = "model.embed_tokens."
        guard !configuration.tieWordEmbeddings,
            !sanitized.keys.contains(where: { $0.hasPrefix("lm_head.") })
        else {
            return sanitized
        }

        var head: [String: MLXArray] = [:]
        for (key, value) in sanitized where key.hasPrefix(embedPrefix) {
            let suffix = String(key.dropFirst(embedPrefix.count))
            head["lm_head." + suffix] = value
        }
        sanitized.merge(head) { existing, _ in existing }
        return sanitized
    }
}
