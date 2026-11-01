//
//  SharpAIMimoV2MTP.swift
//  MLXLLM
//
//  Multi-token prediction layer for mimo_v2, from model.mtp.layers.0.* in
//  vllm/model_executor/models/mimo_v2_mtp.py. It uses the sliding-window
//  attention geometry and a dense MLP. The token embedding and lm_head are
//  shared with the base model. Only layer 0 is built, as vLLM does.
//  Verified against: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/mimo_v2_mtp.py
//  (MiMoV2MTPLayer.forward order: eh_proj of [enorm(embed), hnorm(hidden)], attention and MLP
//  residuals, final norm; load_weights keeps model.embed_tokens and lm_head keys, which are the
//  base checkpoint tensors). The MTP module holds its own lm_head object in vLLM, but it loads
//  the same tensors, so sharing matches the checkpoint values.
//  UNVERIFIED: no checkpoint run. Not wired into the decode loop.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIMimoV2MTPLayer: Module {
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMimoV2Attention
    @ModuleInfo(key: "pre_mlp_layernorm") var preMLPLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIMimoV2MLP
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm

    init(_ config: SharpAIMimoV2Configuration) {
        _enorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _hnorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _ehProj.wrappedValue = Linear(config.hiddenSize * 2, config.hiddenSize, bias: false)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _selfAttn.wrappedValue = SharpAIMimoV2Attention(config, isSlidingWindow: true)
        _preMLPLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
        _mlp.wrappedValue = SharpAIMimoV2MLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _finalLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layernormEpsilon)
    }

    /// Residual layout follows vLLM: the attention and MLP residuals add onto
    /// the projected (embedding, hidden) input.
    func callAsFunction(
        _ embeddings: MLXArray,
        hidden previous: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let fused = ehProj(
            concatenated([enorm(embeddings), hnorm(previous)], axis: -1))
        let afterAttention = fused + selfAttn(inputLayerNorm(fused), mask: mask, cache: cache)
        let afterMLP = afterAttention + mlp(preMLPLayerNorm(afterAttention))
        return finalLayerNorm(afterMLP)
    }
}

/// Wrapper so parameter paths read `model.mtp.layers.0.*`.
class SharpAIMimoV2MTPLayers: Module {
    let layers: [SharpAIMimoV2MTPLayer]

    init(_ config: SharpAIMimoV2Configuration) {
        self.layers = [SharpAIMimoV2MTPLayer(config)]
        super.init()
    }
}
