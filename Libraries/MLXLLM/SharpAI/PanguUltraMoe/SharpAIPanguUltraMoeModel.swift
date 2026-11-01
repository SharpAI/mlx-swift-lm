//
//  SharpAIPanguUltraMoeModel.swift
//  MLXLLM
//
//  Text model for `pangu_ultra_moe` (PanguUltraMoEForCausalLM), a port of modeling_openpangu_moe.py
//  from the openPangu-Ultra-MoE-718B repos. Attention is multi-head latent attention with a
//  q LoRA and a kv LoRA; the RoPE is interleaved over the rope channels. Layers below
//  num_dense_layers use a dense MLP, the rest a sigmoid-gated MoE with shared experts. With
//  sandwich_norm, a second norm wraps the attention output and the MLP output.
//
//  Weight keys are the HF names. Routed experts stay separate modules (`mlp.experts.N.*`), so
//  sanitize does not stack them. The multi-token-prediction layer (index num_hidden_layers) is
//  not part of this model and is dropped in sanitize.
//  UNVERIFIED: no checkpoint was loaded and no output was compared with the reference Python.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The reference hard-codes 1e-5 for the latent norms, not rms_norm_eps.
private let latentNormEps: Float = 1e-5

/// Reads a dimension that `validateForForward()` has already checked.
private func forwardDim<T>(_ value: T?, _ key: String) -> T {
    guard let value else {
        fatalError("pangu_ultra_moe: \(key) missing. Call validateForForward() first.")
    }
    return value
}

class SharpAIPanguUltraMoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Sigmoid router with top-k selection and no grouping or correction bias, as in MoEGate.
/// The gate weight is `mlp.gate.weight`.
class SharpAIPanguUltraMoeMoE: Module, UnaryLayer {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: Linear
    let experts: [SharpAIPanguUltraMoeMLP]
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIPanguUltraMoeMLP?

    init(_ config: SharpAIPanguUltraMoeConfiguration) {
        let numRouted = forwardDim(config.routedExperts, "num_routed_experts")
        let moeIntermediate = forwardDim(config.moeIntermediateSize, "moe_intermediate_size")

        self.topK = forwardDim(config.expertsPerToken, "num_experts_per_tok")
        self.normTopkProb = config.normTopkProb
        self.routedScalingFactor = forwardDim(
            config.routedScalingFactor, "routed_scaling_factor")

        _gate.wrappedValue = Linear(config.hiddenSize, numRouted, bias: false)
        self.experts = (0 ..< numRouted).map { _ in
            SharpAIPanguUltraMoeMLP(
                hiddenSize: config.hiddenSize, intermediateSize: moeIntermediate)
        }
        if let shared = config.sharedExperts, shared > 0 {
            _sharedExperts.wrappedValue = SharpAIPanguUltraMoeMLP(
                hiddenSize: config.hiddenSize, intermediateSize: moeIntermediate * shared)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MoEGate.forward: router logits and scores are float32.
        let scores = sigmoid(gate(x.asType(.float32)))  // [B, L, E]
        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]  // [B, L, K]
        var weights = takeAlong(scores, inds, axis: -1)  // [B, L, K]
        if topK > 1, normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor

        // Sum of the selected gate weights per expert. Unselected experts get weight zero,
        // which gives the same result as dispatching each token to its K experts.
        var routed = zeros(x.shape)
        for (e, expert) in experts.enumerated() {
            let selected = (inds .== MLXArray(Int32(e))).asType(weights.dtype)
            let expertWeight = (weights * selected).sum(axis: -1, keepDims: true)  // [B, L, 1]
            routed = routed + expertWeight * expert(x).asType(weights.dtype)
        }

        var out = routed.asType(x.dtype)
        if let sharedExperts {
            out = out + sharedExperts(x)
        }
        return out
    }
}

class SharpAIPanguUltraMoeAttention: Module {
    let numHeads: Int
    let qkDim: Int
    let ropeDim: Int
    let vDim: Int
    let qHeadDim: Int
    let kvLoraDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIPanguUltraMoeConfiguration) {
        numHeads = config.attentionHeads
        qkDim = forwardDim(config.attentionQKDim, "attention_qk_dim")
        ropeDim = forwardDim(config.attentionQKRopeDim, "attention_qk_rope_dim")
        vDim = forwardDim(config.attentionVDim, "attention_v_dim")
        kvLoraDim = forwardDim(config.attentionKVLoraDim, "attention_kv_lora_dim")
        qHeadDim = qkDim + ropeDim
        scale = pow(Float(qHeadDim), -0.5)

        if let qLora = config.attentionQLoraDim {
            _qAProj.wrappedValue = Linear(config.hiddenSize, qLora, bias: false)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLora, eps: latentNormEps)
            _qBProj.wrappedValue = Linear(qLora, numHeads * qHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * qHeadDim, bias: false)
        }

        _kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraDim + ropeDim, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraDim, eps: latentNormEps)
        _kvBProj.wrappedValue = Linear(
            kvLoraDim, numHeads * (qkDim + vDim), bias: false)
        _oProj.wrappedValue = Linear(numHeads * vDim, config.hiddenSize, bias: false)

        // The reference rotates interleaved pairs (2i, 2i+1), with base rope_theta over the rope
        // channels. MLX's traditional RoPE does the same.
        rope = initializeRope(
            dims: ropeDim,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qFlat: MLXArray
        if let qAProj, let qALayerNorm, let qBProj {
            qFlat = qBProj(qALayerNorm(qAProj(x)))
        } else {
            qFlat = qProj!(x)
        }
        let q = qFlat.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkDim], axis: -1)
        let qNope = splitQ[0]
        var qPe = splitQ[1]

        let latentKv = kvAProjWithMqa(x)
        let splitLatent = split(latentKv, indices: [kvLoraDim], axis: -1)
        let kvLatent = splitLatent[0]
        var kPe = splitLatent[1].reshaped(B, L, 1, ropeDim).transposed(0, 2, 1, 3)

        let kv = kvBProj(kvALayerNorm(kvLatent))
            .reshaped(B, L, numHeads, qkDim + vDim)
            .transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkDim], axis: -1)
        let kNope = splitKv[0]
        let values = splitKv[1]

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let queries = concatenated([qNope, qPe], axis: -1)
        let keys = concatenated([kNope, kPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

class SharpAIPanguUltraMoeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIPanguUltraMoeAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_mlp_layernorm") var preMlpLayerNorm: RMSNorm?
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayerNorm: RMSNorm?

    init(_ config: SharpAIPanguUltraMoeConfiguration, layer: Int) {
        _selfAttn.wrappedValue = SharpAIPanguUltraMoeAttention(config)

        if layer >= forwardDim(config.firstDenseLayers, "num_dense_layers") {
            _mlp.wrappedValue = SharpAIPanguUltraMoeMoE(config)
        } else {
            _mlp.wrappedValue = SharpAIPanguUltraMoeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: forwardDim(config.intermediateSize, "intermediate_size"))
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.sandwichNorm {
            _preMlpLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postMlpLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let attn = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)

        if let pre = preMlpLayerNorm, let post = postMlpLayerNorm {
            // Sandwich norm: the attention output is normalised before the residual add, and
            // the MLP output after its own.
            let h = x + postAttentionLayerNorm(attn)
            return h + post(mlp(pre(h)))
        }

        let h = x + attn
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAIPanguUltraMoeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIPanguUltraMoeDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIPanguUltraMoeConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIPanguUltraMoeDecoderLayer(config, layer: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

class SharpAIPanguUltraMoeModel: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAIPanguUltraMoeModelInner
    let configuration: SharpAIPanguUltraMoeConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIPanguUltraMoeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.attentionHeads }
        self.model = SharpAIPanguUltraMoeModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Drops rotary buffers, the tied LM head and the MTP layers. Routed experts keep their HF
    /// names, so nothing is stacked.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIPanguUltraMoeSanitizer.sanitize(weights, configuration: configuration)
    }
}

extension SharpAIPanguUltraMoeModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
