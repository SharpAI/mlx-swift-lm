//
//  SharpAIMinimaxM3MtpModel.swift
//  MLXLLM
//
//  SharpAI fork overlay: MiniMax M3 multi-token-prediction (MTP) draft head. Not part of upstream
//  mlx-swift-lm.
//
//  One MTP depth, per vLLM `MiniMaxM3MultiTokenPredictorLayer` (vllm/models/minimax_m3/amd/mtp.py):
//    e  = enorm(inputs_embeds), with rows at position 0 zeroed first
//    h' = hnorm(previous_hidden_states)
//    x  = eh_proj(concat([e, h'], dim=-1))          (2*hidden -> hidden, no bias)
//    x  = transformer_layer(x)                       (MoE + sparse attention, always)
//    out = x                                         (residual already added)
//  Logits (vLLM `compute_logits`): lm_head(final_layernorm(out)). The lm_head and the token
//  embedding belong to the target model and are passed in by the caller, so this head owns neither.
//
//  The transformer block is `SharpAIMinimaxM3TextDecoderLayer` built at the last decoder index,
//  which NVIDIA NeMo AutoModel also documents for its MTP block. vLLM forces the MoE and sparse
//  paths on that block. Here the last index must already be MoE and sparse, and validation rejects
//  any config where it is not, so no forced variant is needed.
//
//  Verified against: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/minimax_m3/amd/mtp.py
//  Verified against: https://docs.nvidia.com/nemo/automodel/v0.4/nemo-automodel/nemo_automodel/components/models/minimax_m3_vl/mtp
//    (the single-depth block shape, `eh_proj` 2*hidden -> hidden, and the shared lm_head).
//
//  Known limits (remaining, see the registration creator):
//  - The forward runs on a full sequence from position 0. It takes no RoPE offset and no KV cache,
//    so it cannot yet drive cached speculative decoding.
//  - The public M3 checkpoint has no `mtp.*` tensors, so the layer key layout is only as vLLM names it.
//  - The shared embedding and lm_head are not loaded here.

import Foundation
import MLX
import MLXNN

/// One MTP depth: `enorm`, `hnorm`, `eh_proj`, `transformer_layer` and `final_layernorm`.
/// Parameter keys match vLLM's module names under `model.layers.<i>.`.
final class SharpAIMinimaxM3MtpLayer: Module {
    @ModuleInfo(key: "enorm") var enorm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "transformer_layer") var transformerLayer: SharpAIMinimaxM3TextDecoderLayer
    @ModuleInfo(key: "final_layernorm") var finalLayernorm: SharpAIMinimaxM3TextRMSNorm

    init(_ config: SharpAIMinimaxM3TextConfiguration) {
        _enorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _hnorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ehProj.wrappedValue = Linear(2 * config.hiddenSize, config.hiddenSize, bias: false)
        _transformerLayer.wrappedValue = SharpAIMinimaxM3TextDecoderLayer(
            config, layer: config.hiddenLayers - 1)
        _finalLayernorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

/// The `model` container: `layers` holds `num_mtp_modules` depths, indexed as `model.layers.<i>`.
final class SharpAIMinimaxM3MtpInner: Module {
    let layers: [SharpAIMinimaxM3MtpLayer]

    init(_ config: SharpAIMinimaxM3TextConfiguration, count: Int) {
        self.layers = (0 ..< count).map { _ in SharpAIMinimaxM3MtpLayer(config) }
        super.init()
    }
}

/// The MiniMax M3 MTP draft head. Takes the target's embeddings and hidden states and returns
/// logits through the target's lm_head.
public final class SharpAIMinimaxM3MtpHead: Module {
    public let numMtpLayers: Int

    @ModuleInfo(key: "model") var model: SharpAIMinimaxM3MtpInner

    /// Builds the head from the text-model fields of the MTP config.
    /// - Throws: `SharpAIMinimaxM3MtpError.invalidConfiguration` when the config cannot build the
    ///   last-index MoE + sparse block or a field is out of range.
    public init(_ text: SharpAIMinimaxM3TextConfiguration, numMtpModules: Int) throws {
        try Self.validate(text, numMtpModules: numMtpModules)
        self.numMtpLayers = numMtpModules
        _model.wrappedValue = SharpAIMinimaxM3MtpInner(text, count: numMtpModules)
        super.init()
    }

    /// Checks what the head needs to build: the last decoder index must be MoE and sparse, and
    /// the MoE, sparse-indexer and rotary sizes must be valid.
    public static func validate(
        _ text: SharpAIMinimaxM3TextConfiguration, numMtpModules: Int
    ) throws {
        guard numMtpModules > 0 else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "num_mtp_modules must be positive")
        }
        guard text.hiddenSize > 0, text.hiddenLayers > 0, text.attentionHeads > 0,
            text.kvHeads > 0, text.attentionHeads % text.kvHeads == 0
        else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "hidden_size, num_hidden_layers and attention heads must be positive, with heads a multiple of kv heads")
        }
        let rotary = text.rotaryDimensions
        guard rotary > 0, rotary <= text.resolvedHeadDim, rotary % 2 == 0 else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "rotary_dim must be an even value in 1...head_dim")
        }
        let last = text.hiddenLayers - 1
        guard text.isMoELayer(last) else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "the MTP block sits at the last decoder index \(last), which must be a MoE layer")
        }
        guard text.isSparseAttentionLayer(last), let sparse = text.sparseAttention else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "the MTP block sits at the last decoder index \(last), which must use sparse attention")
        }
        guard let experts = text.numLocalExperts, let perToken = text.numExpertsPerTok,
            experts > 0, perToken > 0, perToken <= experts, text.intermediateSize != nil
        else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "MoE needs num_local_experts, num_experts_per_tok (<= experts) and intermediate_size")
        }
        guard sparse.scoreType == "max" else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "sparse_score_type must be max, got \(sparse.scoreType)")
        }
        guard sparse.indexDim > 0, sparse.indexDim % 2 == 0, sparse.indexHeads > 0,
            text.attentionHeads % sparse.indexHeads == 0, sparse.topkBlocks > 0,
            sparse.blockSize > 0, sparse.localBlocks > 0
        else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "sparse_attention_config needs an even index_dim, index heads dividing attention heads, and positive topk, block and local sizes")
        }
    }

    /// Runs one MTP depth over a full sequence that starts at position 0.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: `[B, L, hidden]` target token embeddings for the drafted positions.
    ///   - hiddenStates: `[B, L, hidden]` target hidden states at the same positions, before the
    ///     target's final norm.
    ///   - step: MTP depth index, taken modulo `numMtpLayers` as vLLM does.
    /// - Returns: `[B, L, hidden]` depth output before `final_layernorm`. Pass it to `computeLogits`.
    public func callAsFunction(
        inputsEmbeds: MLXArray, hiddenStates: MLXArray, step: Int = 0
    ) -> MLXArray {
        let layer = model.layers[step % numMtpLayers]
        let length = inputsEmbeds.dim(1)
        // vLLM zeroes the embedding at position 0 before enorm. Position 0 is index 0 here.
        let positions = MLXArray((0 ..< length).map { Int32($0) }).reshaped(1, length, 1)
        let atStart = positions .== MLXArray(Int32(0))
        let embeds = which(atStart, MLXArray(Float(0)).asType(inputsEmbeds.dtype), inputsEmbeds)
        let combined = concatenated(
            [layer.enorm(embeds), layer.hnorm(hiddenStates)], axis: -1)
        let x = layer.ehProj(combined)
        return layer.transformerLayer(x, mask: .none, cache: nil)
    }

    /// vLLM `compute_logits`: `lm_head(final_layernorm(hidden))`, with the target's lm_head.
    public func computeLogits(
        _ hiddenStates: MLXArray, step: Int = 0, lmHead: (MLXArray) -> MLXArray
    ) -> MLXArray {
        lmHead(model.layers[step % numMtpLayers].finalLayernorm(hiddenStates))
    }
}
