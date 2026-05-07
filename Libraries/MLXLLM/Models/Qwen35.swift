//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

@preconcurrency import AVFoundation
@preconcurrency import CoreImage.CIFilterBuiltins
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    // MTP fields
    public var numNextnPredictLayers: Int = 0
    public var mtpNumHiddenLayers: Int? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        
        let mtpLayers = try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
        self.numNextnPredictLayers = try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? mtpLayers

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (-(convKernelSize - 1))...]
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        var state = cache?[1]
        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        var out: MLXArray

        (out, state) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask
        )

        if let cache {
            cache[1] = state
        }

        out = norm(out, gate: z)
        return outProj(out.reshaped(B, S, -1))
    }
}

final class MathRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let isCPU = Device.defaultDevice().deviceType == .cpu
        if isCPU {
            let variance = mean(square(hiddenStates), axis: -1, keepDims: true)
            return (hiddenStates * rsqrt(variance + eps)) * weight
        }
        return MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: MathRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MathRMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = MathRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = MathRMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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

        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = {
            if let override = ProcessInfo.processInfo.environment["SWIFTLM_TOP_K"],
               let k = Int(override), k > 0 {
                let effective = min(k, args.numExpertsPerTok)
                print("[SwiftLM] Top-K override: \(args.numExpertsPerTok) -> \(effective)")
                return effective
            }
            return args.numExpertsPerTok
        }()

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MathRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MathRMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = MathRMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = MathRMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    private func prefaultEvaluatable(_ evalObj: Evaluatable?) {
        guard let evalObj = evalObj else { return }
        for array in evalObj.innerState() {
            MLXFast.prefault(array)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        
        // OS VM SWAP WATCHDOG BYPASS: 
        // For massive models (e.g. 122B Qwen on 64GB Mac), Apple Unified Memory will swap
        // weights and KV Cache to disk. If the GPU page-faults these during execution, it blocks the thread
        // generating kIOGPUCommandBufferCallbackErrorTimeout after 5 seconds.
        // By evaluating innerState() sequentially on the CPU thread here, we force the SSD load
        // safely outside of the Apple Metal limits. KERNEL EXECUTION DROPS FROM 5s+ TO 100ms.
        if ProcessInfo.processInfo.environment["EXPERIMENTAL_SSD_STREAM"] != nil {
            prefaultEvaluatable(self.inputLayerNorm)
            prefaultEvaluatable(self.postAttentionLayerNorm)
            prefaultEvaluatable(self.selfAttn)
            prefaultEvaluatable(self.linearAttn)
            prefaultEvaluatable(cache)
            // WE MUST NOT PREFAULT MOE EXPERTS TO AVOID MEMORY CRASH
        }

        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        
        // ─────────────────────────────────────────────────────────────────────
        // FLUSH-LOAD-EXECUTE ARCHITECTURE: Phase 1 (Flush & Split)
        // ─────────────────────────────────────────────────────────────────────
        // If we are processing a Mixture of Experts layer AND SSD expert streaming
        // is active, we explicitly evaluate the attention subgraph (`h`) and
        // synchronize the Metal GPU queue here.
        //
        // THIS IS VITAL FOR SSD STREAMING: When SSD Expert Streaming evaluates
        // the `mlp` custom op, it performs a highly latency-sensitive `load_sync`
        // (blocking the CPU). Ensuring the previous GPU work is committed and
        // completed means the expert GEMM executes on an isolated, empty Metal
        // Command Buffer.
        //
        // GATING: The flush is ONLY needed when streaming experts from SSD. With
        // experts resident in RAM (default 35B-A3B path), these two per-layer
        // syncs drain the Metal command queue 2x per layer x 32 layers = 64 hard
        // CPU<->GPU syncs per token, capping GPU utilization well below 100% and
        // serializing kernel launches that MLX would otherwise pipeline.
        // ─────────────────────────────────────────────────────────────────────
        let needsMoeFlush = (self.mlp is Qwen35SparseMoeBlock)
            && ExpertStreamingConfig.shared.isEnabled
        if needsMoeFlush {
            if let cacheState = cache {
                eval([h] + cacheState.innerState())
            } else {
                eval(h)
            }
            Stream.gpu.synchronize()
        }

        let mlpOutput = (self.mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        let finalH = h + mlpOutput
        if needsMoeFlush {
            eval(finalH)
            Stream.gpu.synchronize()
        }
        return finalH
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module, LayerPartitionable, StreamableMoE {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: MathRMSNorm

    let ssmIdx: Int
    let faIdx: Int

    // LayerPartitionable
    public var gpuLayerCount: Int?
    public var totalLayerCount: Int { layers.count }
    
    // StreamableMoE
    public var streamExperts: Bool = false

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = MathRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil || cacheArray?.count != layers.count {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts, cacheToEval: cacheArray?[i]) {
                layer(
                    hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
            }
        }

        return norm(hiddenStates)
    }

    public func callCapturing(_ inputs: MLXArray, cache: [KVCache?]? = nil, captureLayerIDs: Set<Int>) -> (MLXArray, [Int: MLXArray]) {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil || cacheArray?.count != layers.count {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        var captured = [Int: MLXArray]()

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts, cacheToEval: cacheArray?[i]) {
                layer(
                    hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i])
            }
            
            if captureLayerIDs.contains(i) {
                captured[i] = hiddenStates
            }
        }

        return (norm(hiddenStates), captured)
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    // MTP heads — loaded only when SWIFTLM_MTP_ENABLE=1 and the checkpoint retains them.
    // Key path: "mtp.{i}.{subkey}" maps into mtp[i].
    @ModuleInfo(key: "mtp") public var mtp: [Qwen35MTPLayer]

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        // Allocate MTP head modules (populated by weight loader if SWIFTLM_MTP_ENABLE=1)
        let numMTP = MTPConfig.retainMTPWeights ? args.numNextnPredictLayers : 0
        _mtp.wrappedValue = (0 ..< numMTP).map { i in
            Qwen35MTPLayer(args, layerIdx: args.hiddenLayers + i)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let hasMTPWeights = weights.keys.contains { $0.contains("mtp.") }
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasMTPWeights || hasUnsanitizedConv1d

        var weights = weights
        if !MTPConfig.retainMTPWeights {
            weights = weights.filter { !$0.key.contains("mtp.") }
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            
            // Map community MTP checkpoint keys (e.g. language_model.mtp.fc) to array indices (language_model.mtp.0.fc)
            // Some checkpoints use .mtp.fc instead of the array index .mtp.0.fc
            let updatedKey = k.contains(".mtp.") && !k.contains(".mtp.0.") ? k.replacingOccurrences(of: ".mtp.", with: ".mtp.0.") : k
            let updatedVal = v
            
            if updatedKey != k {
                weights.removeValue(forKey: k)
                weights[updatedKey] = v
            }
            
            if updatedKey.contains("conv1d.weight") && updatedVal.dim(-1) != 1 {
                weights[updatedKey] = updatedVal.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { updatedKey.hasSuffix($0) })
                && updatedVal.ndim == 1
            {
                weights[updatedKey] = updatedVal + MLXArray(1, dtype: updatedVal.dtype)
            }
        }

        return weights
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") public var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        // FP8 block-wise dequantization for Qwen3.6-27B-FP8 (dense checkpoint).
        // Official FP8 checkpoints ship each weight tensor alongside a
        // "weight_scale_inv" tensor with shape [outFeatures/128, inFeatures/128].
        // We dequantize eagerly here (dense model fits in 64 GB without lazy streaming).
        var processed = [String: MLXArray]()
        for (key, value) in sanitized {
            if key.hasSuffix(".weight_scale_inv") {
                let wKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let w = sanitized[wKey], processed[wKey] == nil {
                    // Block-wise: scale_inv is [outBlocks, inBlocks], w is [outDim, inDim]
                    // Swift MLX maps F8_E4M3 → uint8; fromFp8 gives the same signed
                    // [-448,448] range that Python mx.load() produces automatically.
                    let wFp: MLXArray = MLXFast.fromFp8(w, dtype: .bfloat16)
                    let bs = 128
                    let (m, n) = (wFp.dim(0), wFp.dim(1))
                    let padBottom = (bs - m % bs) % bs
                    let padSide   = (bs - n % bs) % bs
                    var padded = MLX.padded(wFp, widths: [[0, padBottom], [0, padSide]])
                    padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
                    let scaled = padded * value[0..., .newAxis, 0..., .newAxis]
                    let dequant = scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                    processed[wKey] = dequant.asType(.bfloat16)
                }
            } else if processed[key] == nil {
                processed[key] = value
            }
        }
        if !processed.isEmpty { sanitized = processed }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - MTP Module

/// A single MTP (Multi-Token Prediction) head for Qwen3.6.
/// Architecture mirrors the official schema:
///   pre_fc_norm_embedding: RMSNorm on the embedded token
///   pre_fc_norm_hidden: RMSNorm on the hidden state
///   fc: Linear that combines enorm(embed) + hnorm(h) -> hidden_size
///   layers: Array of Qwen35DecoderLayer for extra context
///   norm: Final RMSNorm on the MTP output
public class Qwen35MTPLayer: Module {
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: MathRMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: MathRMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: MathRMSNorm

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        _preFCNormEmbedding.wrappedValue = MathRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormHidden.wrappedValue = MathRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        // MTP layers in Qwen3.6 use full attention. Force this by passing a full attention layerIdx.
        _layers.wrappedValue = [Qwen35DecoderLayer(args, layerIdx: args.fullAttentionInterval - 1)]
        _norm.wrappedValue = MathRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenState: MLXArray,
        embedding: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        var h = fc(concatenated([preFCNormEmbedding(embedding), preFCNormHidden(hiddenState)], axis: -1))
        for layer in layers {
            h = layer(h, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
        }
        return norm(h)
    }
}

// MARK: - MTPLanguageModel Conformance for Qwen35TextModel

extension Qwen35TextModel: MTPLanguageModel {
    /// Forward pass through the main model **and** all MTP heads.
    /// Returns: [main_logits, mtp_head_0_logits, mtp_head_1_logits, ...]
    public func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?) -> [MLXArray] {
        guard !mtp.isEmpty else {
            // Fallback: no MTP heads loaded; return only main logits
            return [callAsFunction(inputs, cache: cache)]
        }

        // Embed tokens — needed as the MTP layer input alongside main hidden state
        let embedding = model.embedTokens(inputs)   // [B, S, D]
        let mainHidden = model(inputs, cache: cache) // [B, S, D] (normed)

        // Main logits
        let mainLogits: MLXArray
        if let head = lmHead {
            mainLogits = head(mainHidden)
        } else {
            mainLogits = model.embedTokens.asLinear(mainHidden)
        }

        // MTP heads — each refines the previous hidden state
        var result = [mainLogits]
        var prevHidden = mainHidden
        for (i, mtpLayer) in mtp.enumerated() {
            let mtpCache: [KVCache]? = mtpCaches?[i]
            let faMask = createAttentionMask(h: prevHidden, cache: mtpCache?.first)
            let mtpHidden = mtpLayer(
                prevHidden, embedding: embedding,
                attentionMask: faMask, ssmMask: nil, cache: mtpCache?.first
            )
            
            // Project the MTP hidden state to vocabulary logits using the shared lm_head
            if let head = lmHead {
                result.append(head(mtpHidden))
            } else {
                result.append(model.embedTokens.asLinear(mtpHidden))
            }
            
            // The hidden state is passed to the next MTP layer
            prevHidden = mtpHidden
        }
        return result
    }

    /// Allocate persistent KVCache arrays for each MTP head
    public func makeMTPCaches(parameters: GenerateParameters?) -> [[KVCache]] {
        return mtp.map { mtpLayer in
            // Each MTP layer contains a single DecoderLayer which needs one KVCache
            [KVCacheSimple()]
        }
    }
}

