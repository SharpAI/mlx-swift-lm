// SharpAI fork overlay: llama 4 position-based query scaling for the ParamsJson model.
// Not part of upstream mlx-swift-lm. Same formula as Mistral3Text's getLlama4AttentionScale.
//
// Verified against: vLLM vllm/model_executor/models/deepseek_v2.py. _get_llama_4_scaling computes
// 1 + beta * log(1 + floor(pos / original_max)) over absolute positions, and the MLA forward runs
// `q *= llama_4_scaling` on the full query (q_nope and the rotary part), after RoPE. The drafter
// is MistralLarge3 (deepseek_v3 model_type) and reads llama_4_scaling from its params.json.
// The per-chunk offset here is the same absolute position. Not checked against a dump.

import Foundation
import MLX
import MLXLMCommon

public enum SharpAIParamsJsonLlama4Scaling {

    /// Scale for positions `start..<stop`: 1 + beta * log(1 + floor(pos / originalMax)).
    /// Returns shape [stop - start, 1] so it broadcasts over [B, H, L, D] queries.
    public static func scale(
        start: Int, stop: Int, beta: Float, originalMaxPositionEmbeddings: Int
    ) -> MLXArray {
        let positions = MLXArray(Int32(start) ..< Int32(stop))
        let scaling =
            1 + beta
            * MLX.log(
                1 + MLX.floor(positions.asType(.float32) / Float(originalMaxPositionEmbeddings))
            )
        return scaling[0..., .newAxis]
    }

    /// Query scale for a chunk that starts at `offset`. Ones when the config has no llama_4_scaling.
    public static func queryScale(
        config: SharpAIParamsJsonConfiguration, offset: Int, length: Int, dtype: DType
    ) -> MLXArray {
        guard let scaling = config.llama4Scaling else {
            return MLXArray.ones([length, 1]).asType(dtype)
        }
        return scale(
            start: offset,
            stop: offset + length,
            beta: scaling.beta,
            originalMaxPositionEmbeddings: scaling.originalMaxPositionEmbeddings
        ).asType(dtype)
    }
}
