// SharpAI fork overlay: dense MLP, routed MoE and decoder layer for the ParamsJson drafter.
// Not part of upstream mlx-swift-lm. Structure copied from Mistral4MLP, Mistral4MoE and
// Mistral4DecoderLayer.
//
// Verified against: vLLM vllm/transformers_utils/configs/mistral.py (_remap_moe_args sets
// scoring_func="softmax", topk_method=None, norm_topk_prob=True), and params.json at
// https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle. That file sets
// num_expert_groups=1 and num_expert_groups_per_tok=1, a single group, so group selection keeps
// every expert and the router is a plain softmax top-k with renormalized weights, scaled by
// routed_scale after normalization, as coded here. The group keys are not read.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIParamsJsonMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenDim: Int, intermediateDim: Int) {
        _gateProj.wrappedValue = Linear(hiddenDim, intermediateDim, bias: false)
        _upProj.wrappedValue = Linear(hiddenDim, intermediateDim, bias: false)
        _downProj.wrappedValue = Linear(intermediateDim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class SharpAIParamsJsonMoE: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIParamsJsonMLP?

    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    init(_ config: SharpAIParamsJsonConfiguration) {
        topK = config.moe.numExpertsPerTok
        normTopkProb = config.moe.normTopkProb
        routedScalingFactor = config.moe.routedScale

        _gate.wrappedValue = Linear(config.dim, config.moe.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.dim,
            hiddenDims: config.moe.expertHiddenDim,
            numExperts: config.moe.numExperts,
            activation: silu
        )
        if config.moe.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = SharpAIParamsJsonMLP(
                hiddenDim: config.dim,
                intermediateDim: config.moe.expertHiddenDim * config.moe.numSharedExperts
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scores = softmax(gate(x).asType(.float32), axis: -1)

        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var selected = takeAlong(scores, inds, axis: -1)

        if normTopkProb {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        selected = (selected * routedScalingFactor).asType(x.dtype)

        var y = switchMLP(x, inds)
        y = (y * selected[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class SharpAIParamsJsonDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIParamsJsonAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIParamsJsonMoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIParamsJsonConfiguration) {
        _selfAttn.wrappedValue = SharpAIParamsJsonAttention(config)
        _mlp.wrappedValue = SharpAIParamsJsonMoE(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.dim, eps: config.normEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attnScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}
