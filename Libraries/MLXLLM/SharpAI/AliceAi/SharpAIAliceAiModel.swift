// SharpAI fork overlay: alice_ai model. Not part of upstream mlx-swift-lm.
//
// Hybrid stack: KDA linear attention on most layers, gated full attention where layer_types says so,
// a sigmoid-routed MoE on every layer, and block attention residuals (AttnRes) across layers.
// Topology follows vLLM's alice_ai.py. Forward pass only; there is no compiled decode path.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Full attention

final class SharpAIAliceAiAttention: Module {
    let numHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: SharpAIAliceAiConfiguration) {
        numHeads = args.attentionHeads
        kvHeads = args.kvHeads
        scale = pow(Float(args.headDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, numHeads * args.headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, kvHeads * args.headDim, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, kvHeads * args.headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * args.headDim, args.hiddenSize, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(args.headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // UNVERIFIED: per-head interleaved q/gate split, copied from Qwen3-Next.
        // No alice_ai source in the tree confirms it.
        let qSplit = qProj(x).reshaped(B, L, numHeads, -1).split(parts: 2, axis: -1)
        let queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let gate = qSplit[1].reshaped(B, L, -1)
        let keys = kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        let q = applyRotaryPosition(rope, to: queries, offset: offset)
        let k = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: q, keys: k, values: values, cache: cache, scale: scale, mask: mask)

        let merged = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(merged * sigmoid(gate))
    }
}

// MARK: - MoE

final class SharpAIAliceAiMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Router weight plus the DeepSeek-style score correction bias (`mlp.gate.*` in the checkpoint).
final class SharpAIAliceAiRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(hiddenSize: Int, numExperts: Int) {
        _weight.wrappedValue = MLXArray.zeros([numExperts, hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([numExperts], dtype: .float32)
        super.init()
    }

    /// Experts are picked on sigmoid scores plus the bias. Weights use the unbiased scores.
    /// UNVERIFIED: the top-k selection rule follows vLLM's e_score_correction_bias path (not read here).
    func route(_ x: MLXArray, topK: Int, normalize: Bool) -> (indices: MLXArray, weights: MLXArray) {
        let probs = sigmoid(MLX.matmul(x, weight.T).asType(.float32))
        let selection = probs + correctionBias
        let indices = MLX.argPartition(
            selection * MLXArray(Float(-1)), kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]
        var weights = MLX.takeAlong(probs, indices, axis: -1)
        if normalize {
            weights = weights / MLX.sum(weights, axis: -1, keepDims: true)
        }
        return (indices, weights)
    }
}

final class SharpAIAliceAiSparseMoeBlock: Module {
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: SharpAIAliceAiRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: SharpAIAliceAiMLP
    /// UNVERIFIED: sigmoid-gated shared expert, as in Qwen2-MoE. vLLM passes it as expert_gate.
    /// If the checkpoint lacks this key, `verify: .all` fails the load. No safe default is known.
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: SharpAIAliceAiConfiguration) {
        topK = args.numExpertsPerTok
        normTopkProb = args.normTopkProb

        _gate.wrappedValue = SharpAIAliceAiRouter(
            hiddenSize: args.hiddenSize, numExperts: args.numExperts)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedExpert.wrappedValue = SharpAIAliceAiMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate.route(x, topK: topK, normalize: normTopkProb)
        let expertOut = switchMLP(x, indices)  // [B, T, k, D]
        let routed =
            (expertOut
            * MLX.expandedDimensions(weights, axis: -1).asType(expertOut.dtype))
            .sum(axis: -2)
        let shared = sigmoid(sharedExpertGate(x)) * sharedExpert(x)
        return routed + shared
    }
}

// MARK: - AttnRes parameters

/// A bias-free `[1, D]` projection used as the AttnRes query (`attn_res_proj`, `mlp_res_proj`, `res_proj`).
final class SharpAIAliceAiResProj: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(hiddenSize: Int) {
        _weight.wrappedValue = MLXArray.zeros([1, hiddenSize])
        super.init()
    }
}

final class SharpAIAliceAiFinalAttnRes: Module {
    @ModuleInfo(key: "res_proj") var resProj: SharpAIAliceAiResProj
    @ParameterInfo(key: "res_norm_weight") var resNormWeight: MLXArray

    init(hiddenSize: Int) {
        _resProj.wrappedValue = SharpAIAliceAiResProj(hiddenSize: hiddenSize)
        _resNormWeight.wrappedValue = MLXArray.ones([hiddenSize])
        super.init()
    }
}

// MARK: - Decoder layer

final class SharpAIAliceAiDecoderLayer: Module {
    let isLinear: Bool
    let rmsNormEps: Float

    @ModuleInfo(key: "linear_attn") var linearAttn: SharpAIAliceAiGatedDeltaKDA?
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIAliceAiAttention?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: SharpAIAliceAiSparseMoeBlock

    // Layer 0 may omit these keys. sanitize fills missing ones with init defaults.
    @ModuleInfo(key: "attn_res_proj") var attnResProj: SharpAIAliceAiResProj
    @ParameterInfo(key: "attn_res_norm_weight") var attnResNormWeight: MLXArray
    @ModuleInfo(key: "mlp_res_proj") var mlpResProj: SharpAIAliceAiResProj
    @ParameterInfo(key: "mlp_res_norm_weight") var mlpResNormWeight: MLXArray

    init(_ args: SharpAIAliceAiConfiguration, layerIdx: Int) {
        self.isLinear = args.layerTypes[layerIdx] == "linear_attention"
        self.rmsNormEps = args.rmsNormEps

        if isLinear {
            _linearAttn.wrappedValue = SharpAIAliceAiGatedDeltaKDA(args)
        } else {
            _selfAttn.wrappedValue = SharpAIAliceAiAttention(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        _mlp.wrappedValue = SharpAIAliceAiSparseMoeBlock(args)

        _attnResProj.wrappedValue = SharpAIAliceAiResProj(hiddenSize: args.hiddenSize)
        _attnResNormWeight.wrappedValue = MLXArray.ones([args.hiddenSize])
        _mlpResProj.wrappedValue = SharpAIAliceAiResProj(hiddenSize: args.hiddenSize)
        _mlpResNormWeight.wrappedValue = MLXArray.ones([args.hiddenSize])

        super.init()
    }

    /// One layer of the AttnRes stack (vLLM `forward_attn_res`).
    /// `blocks` are completed block sums, `partial` is the open block. Returns the new open block.
    func step(
        blocks: [MLXArray],
        partial: MLXArray?,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        var sources = blocks
        if let partial {
            sources.append(partial)
        }
        let attnInput = sharpAIAliceAiAttnRes(
            sources: sources,
            normWeight: attnResNormWeight,
            qkWeight: attnResProj.weight.reshaped(-1),
            eps: rmsNormEps)

        let normed = inputLayerNorm(attnInput)
        let attnOut: MLXArray
        if let linearAttn {
            attnOut = linearAttn(normed, mask: ssmMask, cache: cache as? MambaCache)
        } else {
            attnOut = selfAttn!(normed, mask: attentionMask, cache: cache)
        }

        let open = partial.map { $0 + attnOut } ?? attnOut
        let mlpInput = sharpAIAliceAiAttnRes(
            sources: blocks + [open],
            normWeight: mlpResNormWeight,
            qkWeight: mlpResProj.weight.reshaped(-1),
            eps: rmsNormEps)
        return open + mlp(postAttentionLayerNorm(mlpInput))
    }
}

// MARK: - Inner model

final class SharpAIAliceAiModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIAliceAiDecoderLayer]
    let norm: RMSNorm
    @ModuleInfo(key: "attnres_final") var attnResFinal: SharpAIAliceAiFinalAttnRes

    let blockSize: Int
    let faIdx: Int
    let ssmIdx: Int
    let rmsNormEps: Float

    init(_ args: SharpAIAliceAiConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            SharpAIAliceAiDecoderLayer(args, layerIdx: layerIdx)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _attnResFinal.wrappedValue = SharpAIAliceAiFinalAttnRes(hiddenSize: args.hiddenSize)

        self.rmsNormEps = args.rmsNormEps
        self.blockSize = args.attnResBlockSize
        self.faIdx = args.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.ssmIdx = args.layerTypes.firstIndex(of: "linear_attention") ?? 0

        super.init()
    }

    func forward(_ inputs: MLXArray, cache: [KVCache?]?) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil || cacheArray?.count != layers.count {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        // Block 0 is the embedding, as in vLLM's residual bank.
        var blocks: [MLXArray] = [hiddenStates]
        var partial: MLXArray? = nil

        for (i, layer) in layers.enumerated() {
            if i > 0, i % blockSize == 0, let open = partial {
                blocks.append(open)
                partial = nil
            }
            let layerMask = layer.isLinear ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            let layerSsmMask = layer.isLinear ? ssmMask : nil
            partial = layer.step(
                blocks: blocks,
                partial: partial,
                attentionMask: layerMask,
                ssmMask: layerSsmMask,
                cache: cacheArray?[i])
        }

        guard let open = partial else {
            preconditionFailure("AliceAI produced an empty final AttnRes block")
        }
        hiddenStates = sharpAIAliceAiAttnRes(
            sources: blocks + [open],
            normWeight: attnResFinal.resNormWeight,
            qkWeight: attnResFinal.resProj.weight.reshaped(-1),
            eps: rmsNormEps)
        return norm(hiddenStates)
    }
}

// MARK: - Model

public class SharpAIAliceAiModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let model: SharpAIAliceAiModelInner
    let configuration: SharpAIAliceAiConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SharpAIAliceAiConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = SharpAIAliceAiModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model.forward(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    public func makeCache() -> [KVCache] {
        model.layers.map { layer in
            layer.isLinear ? MambaCache() : KVCacheSimple()
        }
    }

    /// Maps HF checkpoint keys onto this module tree. Drops MTP and rotary buffers,
    /// merges split KDA conv weights, stacks experts, fills missing layer-0 AttnRes keys,
    /// and adds 1 to norms (UNVERIFIED, see below).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        for key in Array(out.keys) where key.hasPrefix("mtp.") || key.contains("rotary_emb.inv_freq") {
            out[key] = nil
        }

        // Layer 0 may omit its AttnRes keys, and `verify: .all` would reject that.
        // Fill only the missing keys with init defaults. UNVERIFIED: vLLM's layer-0 mlp mix may differ.
        let layer0 = "model.layers.0"
        let attnResDefaults: [String: MLXArray] = [
            "\(layer0).attn_res_proj.weight": MLXArray.zeros([1, configuration.hiddenSize]),
            "\(layer0).attn_res_norm_weight": MLXArray.ones([configuration.hiddenSize]),
            "\(layer0).mlp_res_proj.weight": MLXArray.zeros([1, configuration.hiddenSize]),
            "\(layer0).mlp_res_norm_weight": MLXArray.ones([configuration.hiddenSize]),
        ]
        for (key, value) in attnResDefaults where out[key] == nil {
            out[key] = value
        }

        // Raw HF layout still has per-expert or split q/k/v conv keys. Converted input has neither
        // and its norms are already shifted, so the shift below must not run again.
        let isRawCheckpoint = out.keys.contains { key in
            key.contains(".mlp.experts.0.")
                || (key.contains(".linear_attn.") && key.hasSuffix("_conv1d.weight"))
        }

        // vLLM maps `linear_attn.a_log_bias` to `A_log`. Both may be stored as 1-D or 4-D.
        for key in Array(out.keys) where key.hasSuffix(".linear_attn.a_log_bias") {
            let renamed = String(key.dropLast("a_log_bias".count)) + "A_log"
            out[renamed] = out.removeValue(forKey: key)?.reshaped(-1)
        }
        for key in Array(out.keys) where key.hasSuffix(".linear_attn.dt_bias") {
            out[key] = out[key]?.reshaped(-1)
        }

        // Split q/k/v conv weights become one depthwise conv over the concatenated channels.
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).linear_attn"
            let splitKeys = ["q", "k", "v"].map { "\(prefix).\($0)_conv1d.weight" }
            guard splitKeys.allSatisfy({ out[$0] != nil }) else { continue }
            let channelFirst = splitKeys.map { swiftConvLayout(out.removeValue(forKey: $0)!) }
            out["\(prefix).conv1d.weight"] = concatenated(channelFirst, axis: 0)
        }

        // Experts: per-expert HF weights become one switch_mlp tensor.
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                guard out["\(prefix).experts.0.\(proj).weight"] != nil else { continue }
                let toJoin = (0 ..< configuration.numExperts).map { expert in
                    out.removeValue(forKey: "\(prefix).experts.\(expert).\(proj).weight")!
                }
                out["\(prefix).switch_mlp.\(proj).weight"] = MLX.stacked(toJoin)
            }
        }

        // UNVERIFIED: assumes zero-centred norms (x * (1 + w)), as in Qwen3Next. No alice_ai source
        // in the tree confirms it. If the checkpoint stores plain RMSNorm weights, every norm is +1 off.
        let zeroCenteredNorms = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]
        for key in Array(out.keys) where isRawCheckpoint {
            guard let value = out[key], value.ndim == 1,
                zeroCenteredNorms.contains(where: { key.hasSuffix($0) })
            else { continue }
            out[key] = value + MLXArray(1, dtype: value.dtype)
        }

        return out
    }
}

/// Torch depthwise conv weights are `[C, 1, K]` (or `[C, K]`). MLX wants `[C, K, 1]`.
private func swiftConvLayout(_ weight: MLXArray) -> MLXArray {
    if weight.ndim == 2 {
        return MLX.expandedDimensions(weight, axis: 2)
    }
    if weight.ndim == 3, weight.dim(1) == 1 {
        return weight.movedAxis(source: 2, destination: 1)
    }
    return weight
}

extension SharpAIAliceAiModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
