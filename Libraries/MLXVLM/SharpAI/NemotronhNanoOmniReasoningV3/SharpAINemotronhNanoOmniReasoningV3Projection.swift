// SharpAI fork overlay: NemotronH_Nano_Omni_Reasoning_V3 sound projection. Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/audio_model.py
// (SoundProjection: RMSNorm `norm`, then `linear1`, then SquaredReLU, then `linear2`; RMSNorm eps 1e-5;
// both linears take the projection_bias flag, which is false in the released config.)
//
// Only the projection is ported. The Parakeet encoder that feeds it (`sound_encoder.*`) is not.

import MLX
import MLXNN

public final class SharpAINemotronhNanoOmniReasoningV3SoundProjection: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    public init(inputSize: Int, hiddenSize: Int, outputSize: Int, bias: Bool) {
        self._norm.wrappedValue = RMSNorm(dimensions: inputSize, eps: 1e-5)
        self._linear1.wrappedValue = Linear(inputSize, hiddenSize, bias: bias)
        self._linear2.wrappedValue = Linear(hiddenSize, outputSize, bias: bias)
        super.init()
    }

    /// (..., sound_hidden) encoder features to (..., llm_hidden) embeddings.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = MLX.maximum(linear1(norm(x)), MLXArray(Float(0)))
        return linear2(activated * activated)
    }
}
