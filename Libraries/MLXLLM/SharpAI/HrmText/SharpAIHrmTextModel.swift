// SharpAI fork overlay: hrm_text model. Not part of upstream mlx-swift-lm.
//
// Port of transformers modeling_hrm_text.py (v5.22.1) and the HRM-Text docs. Source facts:
// - Pre-norm decoder layers. input_layernorm -> self_attn -> residual, post_attention_layernorm -> mlp -> residual.
// - Parameterless RMSNorm: x * rsqrt(mean(x^2) + eps), computed in float32. It has no tensors.
// - Attention: q_proj, k_proj, v_proj, gate_proj, o_proj. RoPE on q and k only (rotate-half).
//   The sigmoid(gate_proj(x)) output gate multiplies the attention output before o_proj.
//   Scale is head_dim^-0.5. There are no q_norm or k_norm modules.
// - Each stack ends with final_norm, which is parameterless as above.
// - Recurrence (HrmTextModel.forward): z_H = embed(ids) * embedding_scale, z_L = z_L_init broadcast.
//   for h in H_cycles: for l in L_cycles: z_L = L(z_L + z_H); then z_H = H(z_H + z_L).
//   The last z_H is the output. Input embeddings are not re-added each step.
// - Cache slot for each stack call is (h * (L_cycles + 1) + l) * num_layers_per_stack, and the H call
//   uses l = L_cycles. A layer at index i in that call uses slot offset + i.
// - L_bp_cycles only changes which steps keep gradients. It has no effect at inference.
// - prefix_lm: tokens with token_type_ids == 1 form one bidirectional block during prefill.
//
// Weight names mirror HF: model.embed_tokens, model.z_L_init, model.L_module.layers.N.*,
// model.H_module.layers.N.*, lm_head. Each stack's final_norm and each layer's norms have no tensors.
// UNVERIFIED: the released checkpoint's safetensors tensor names are not read here. A legacy fused
// gqkv_proj is not split. Load will fail on such a checkpoint rather than guess a layout.
// The K/V head count is config.kvHeads. HF projects K/V with num_attention_heads and the released
// checkpoint has kv == heads, so the grouped-query path is not used by it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Parameterless RMSNorm as in HrmTextRMSNorm: computed in float32 and cast back to the input dtype.
func sharpAIHrmTextRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let xf = x.asType(.float32)
    let normed = xf * (mean(xf.square(), axis: -1, keepDims: true) + eps).rsqrt()
    return normed.asType(x.dtype)
}

final class SharpAIHrmTextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIHrmTextConfiguration) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        // The gate stays [B, L, heads, headDim], as in HF (no transpose).
        let gate = gateProj(x).reshaped(B, L, heads, headDim)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        // [B, heads, L, headDim] -> [B, L, heads, headDim] to match the gate layout.
        .transposed(0, 2, 1, 3)

        let gated = (sigmoid(gate) * attended).reshaped(B, L, -1)
        return oProj(gated)
    }
}

final class SharpAIHrmTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: SharpAIHrmTextConfiguration) {
        _gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class SharpAIHrmTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIHrmTextAttention
    let mlp: SharpAIHrmTextMLP
    let eps: Float

    init(_ config: SharpAIHrmTextConfiguration) {
        _selfAttn.wrappedValue = SharpAIHrmTextAttention(config)
        self.mlp = SharpAIHrmTextMLP(config)
        self.eps = config.rmsNormEps
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(sharpAIHrmTextRMSNorm(x, eps: eps), mask: mask, cache: cache)
        return h + mlp(sharpAIHrmTextRMSNorm(h, eps: eps))
    }
}

/// One H or L stack: `hiddenLayers` decoder layers followed by a parameterless final norm.
final class SharpAIHrmTextStack: Module {
    let layers: [SharpAIHrmTextDecoderLayer]
    let eps: Float

    init(_ config: SharpAIHrmTextConfiguration) {
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIHrmTextDecoderLayer(config)
        }
        self.eps = config.rmsNormEps
    }

    /// `cycleOffset` is the first cache slot of this stack call (HF `cycle_offset`).
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: [KVCache]?,
        cycleOffset: Int
    ) -> MLXArray {
        var h = x
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[cycleOffset + i])
        }
        return sharpAIHrmTextRMSNorm(h, eps: eps)
    }
}

final class SharpAIHrmTextModelInner: Module {
    let hCycles: Int
    let lCycles: Int
    let stackLayers: Int
    let embeddingScale: Float
    let prefixLM: Bool

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "L_module") var lModule: SharpAIHrmTextStack
    @ModuleInfo(key: "H_module") var hModule: SharpAIHrmTextStack
    @ParameterInfo(key: "z_L_init") var zLInit: MLXArray

    init(_ config: SharpAIHrmTextConfiguration) {
        hCycles = config.hCycles
        lCycles = config.lCycles
        stackLayers = config.hiddenLayers
        embeddingScale = config.embeddingScale
        prefixLM = config.prefixLM

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _lModule.wrappedValue = SharpAIHrmTextStack(config)
        _hModule.wrappedValue = SharpAIHrmTextStack(config)
        _zLInit.wrappedValue = zeros([config.hiddenSize])
    }

    /// Hidden states after the H/L recurrence. `tokenTypeIds` ([B, L], optional) enables the
    /// prefix-LM block during prefill when `prefix_lm` is set.
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, tokenTypeIds: MLXArray?
    ) -> MLXArray {
        if let cache {
            precondition(
                cache.count >= hCycles * (lCycles + 1) * stackLayers,
                "hrm_text needs \(hCycles * (lCycles + 1) * stackLayers) cache slots")
        }

        var zH = embedTokens(inputs) * embeddingScale

        let cacheOffset = cache?.first?.offset ?? 0
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if prefixLM, cacheOffset == 0, let tokenTypeIds {
            mask = .array(prefixBlockMask(tokenTypeIds, length: inputs.dim(1)))
        } else {
            mask = createAttentionMask(h: zH, cache: cache?.first)
        }

        var zL = broadcast(zLInit.asType(zH.dtype), to: zH.shape)
        for h in 0 ..< hCycles {
            for l in 0 ..< lCycles {
                let slot = (h * (lCycles + 1) + l) * stackLayers
                zL = lModule(zL + zH, mask: mask, cache: cache, cycleOffset: slot)
            }
            let slot = (h * (lCycles + 1) + lCycles) * stackLayers
            zH = hModule(zH + zL, mask: mask, cache: cache, cycleOffset: slot)
        }
        return zH
    }

    /// [B, 1, L, L] bool mask: causal, plus the bidirectional block where token_type_ids == 1.
    private func prefixBlockMask(_ tokenTypeIds: MLXArray, length: Int) -> MLXArray {
        let batch = tokenTypeIds.dim(0)
        let flat = tokenTypeIds.reshaped(-1).asType(.int32).asArray(Int32.self)
        var masks: [MLXArray] = []
        masks.reserveCapacity(batch)
        for b in 0 ..< batch {
            let blockIds = (0 ..< length).map { flat[b * length + $0] == 1 ? 0 : -1 }
            masks.append(
                SharpAIHrmTextPrefixMask.attentionMask(
                    queryLength: length, cacheOffset: 0, blockIds: blockIds))
        }
        return MLX.stacked(masks, axis: 0).expandedDimensions(axis: 1)
    }
}

/// hrm_text causal LM. Registered under `model_type` "hrm_text" through the overlay creators.
final class SharpAIHrmTextForCausalLM: Module, LLMModel, KVCacheDimensionProvider {
    let configuration: SharpAIHrmTextConfiguration
    /// One entry per cache slot (H_cycles * (L_cycles + 1) * num_layers_per_stack). Used by newCache.
    let kvHeads: [Int]

    let model: SharpAIHrmTextModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ configuration: SharpAIHrmTextConfiguration) {
        self.configuration = configuration
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.cacheSlotCount)
        self.model = SharpAIHrmTextModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(model(inputs, cache: cache, tokenTypeIds: nil))
    }

    /// Prefill with the prefix-LM block. `tokenTypeIds` is [B, L], where 1 marks the
    /// bidirectional block. Pass nil for ordinary causal attention.
    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, tokenTypeIds: MLXArray?
    ) -> MLXArray {
        projectLogits(model(inputs, cache: cache, tokenTypeIds: tokenTypeIds))
    }

    private func projectLogits(_ hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIHrmTextSanitizer.sanitize(weights, configuration: configuration)
    }
}

// MARK: - LoRA

extension SharpAIHrmTextForCausalLM: LoRAModel {
    var loraLayers: [Module] {
        let layers: [Module] = model.lModule.layers + model.hModule.layers
        return layers
    }
}
