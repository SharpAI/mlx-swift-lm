// SharpAI fork overlay: telechat3 model. Not part of upstream mlx-swift-lm.
//
// Subclasses LlamaModel for the forward pass, so the attention, MLP and norm layout is
// LlamaModel's. Only sanitize differs, and it uses SharpAITelechat3Sanitizer.
// UNVERIFIED: the telechat3 attention and MLP layout is assumed to match Llama.

import Foundation
import MLX
import MLXLMCommon

public final class SharpAITelechat3Model: LlamaModel {

    public override init(_ args: LlamaConfiguration) {
        super.init(args)
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAITelechat3Sanitizer.sanitize(
            weights, tieWordEmbeddings: configuration.tieWordEmbeddings)
    }
}
