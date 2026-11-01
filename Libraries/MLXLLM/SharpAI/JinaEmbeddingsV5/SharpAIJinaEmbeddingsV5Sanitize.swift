// SharpAI fork overlay: jina_embeddings_v5 weight sanitize. Not part of upstream.
// Checkpoint names match Qwen3 (`model.layers.N...`), so no renaming is needed.

import Foundation
import MLX
import MLXLMCommon

enum SharpAIJinaEmbeddingsV5Sanitize {

    static func sanitize(
        _ weights: [String: MLXArray], tiedWordEmbeddings: Bool
    ) -> [String: MLXArray] {
        // Adapter tensors are merged from `adapters/<task>` by SharpAIJinaEmbeddingsV5Adapter.
        // Leaving them in the base weights would fail the `noUnusedKeys` check on load.
        let base = weights.filter { !isAdapterKey($0.key) }
        return filterLMHeadWeights(from: base, tiedWordEmbeddings: tiedWordEmbeddings)
    }

    static func isAdapterKey(_ key: String) -> Bool {
        key.contains(".lora_A.") || key.contains(".lora_B.")
    }
}
