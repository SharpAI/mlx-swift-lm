// SharpAI fork overlay: qwen4_exp text model. Not part of upstream mlx-swift-lm.
//
// Port of transformers/models/qwen4_exp/modeling_qwen4_exp.py, text path only:
// Qwen4ExpTextRMSNorm (1 + w), Qwen4ExpTextGatedDeltaNet, Qwen4ExpTextQSAIndexer,
// Qwen4ExpTextAttention, Qwen4ExpTextSparseMoeBlock, Qwen4ExpTextGatedResidual (hyper-connections),
// Qwen4ExpTextPLELayer, Qwen4ExpTextDecoderLayer, Qwen4ExpTextModel and Qwen4ExpForCausalLM.
//
// Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
// (fetched 2026-10-30; the same file's configuration defaults seed = 1234 and norm_topk_prob = True
// are used because config.json does not set them).
//
// Not ported in this file: the vision tower (model.visual.*) and the MTP head (mtp.*). The
// sanitizer drops both key families. Inputs must not be left-padded: the QSA indexer treats the
// visible keys of each query as the causal prefix of the cache, which is what the generation path
// produces for a single unpadded sequence.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Resolved settings

/// Non-optional view of the text config. `seed` and `norm_topk_prob` are absent from config.json and
/// take the reference defaults.
struct SharpAIQwen4ExpSettings {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let attentionBias: Bool
    let rotaryDim: Int
    let ropeTheta: Float
    let layerTypes: [String]
    let linearConvKernelDim: Int
    let linearNumKeyHeads: Int
    let linearNumValueHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let outputGateSiLU: Bool
    let numExperts: Int
    let numExpertsPerTok: Int
    let moeIntermediateSize: Int
    let sharedExpertIntermediateSize: Int
    let normTopKProb: Bool
    let hcCount: Int
    let hcLowrank: Int
    let indexerBudget: Int
    let indexerCompressRatio: Int
    let indexerHeadDim: Int
    let indexerKVHeads: Int
    let indexerNHeads: Int
    let ngramSize: Int
    let ngramVocabSizeBase: Int
    let headsPerNgram: Int
    let ngramVocabSizeDivisor: Int
    let pleEmbedDim: Int
    let pleConvKernelSize: Int
    let pleLayerIds: [Int]
    let eosTokenId: Int
    let seed: Int

    init(_ text: SharpAIQwen4ExpTextConfiguration) throws {
        func required<T>(_ value: T?, _ name: String) throws -> T {
            guard let value else {
                throw ModelFactoryError.invalidConfiguration("qwen4_exp: \(name) is required")
            }
            return value
        }
        let hiddenAct = text.hiddenAct ?? "silu"
        guard hiddenAct == "silu" else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: hidden_act must be silu, got \(hiddenAct)")
        }
        let gateType = text.outputGateType ?? "sigmoid"
        guard gateType == "sigmoid" || gateType == "silu" else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: output_gate_type must be sigmoid or silu, got \(gateType)")
        }
        let indexerKV = try required(text.indexerKvHeads, "indexer_kv_heads")
        guard indexerKV == 1 else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: only indexer_kv_heads == 1 is ported, got \(indexerKV)")
        }
        guard text.linearNumValueHeads % text.linearNumKeyHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: linear_num_value_heads must be a multiple of linear_num_key_heads")
        }

        hiddenSize = text.hiddenSize
        numHiddenLayers = text.numHiddenLayers
        numAttentionHeads = text.numAttentionHeads
        numKeyValueHeads = text.numKeyValueHeads
        headDim = text.headDim
        vocabSize = text.vocabSize
        rmsNormEps = Float(text.rmsNormEps)
        attentionBias = text.attentionBias ?? false
        rotaryDim = Int(Double(text.headDim) * text.ropeParameters.partialRotaryFactor)
        ropeTheta = Float(text.ropeParameters.ropeTheta)
        layerTypes = text.layerTypes
        linearConvKernelDim = text.linearConvKernelDim
        linearNumKeyHeads = text.linearNumKeyHeads
        linearNumValueHeads = text.linearNumValueHeads
        linearKeyHeadDim = text.linearKeyHeadDim
        linearValueHeadDim = text.linearValueHeadDim
        outputGateSiLU = gateType == "silu"
        numExperts = text.numExperts
        numExpertsPerTok = text.numExpertsPerTok
        moeIntermediateSize = text.moeIntermediateSize
        sharedExpertIntermediateSize = text.sharedExpertIntermediateSize
        normTopKProb = true
        hcCount = try required(text.hcCount, "hc_count")
        hcLowrank = try required(text.hcLowrank, "hc_lowrank")
        indexerBudget = try required(text.indexerBudget, "indexer_budget")
        indexerCompressRatio = try required(text.indexerCompressRatio, "indexer_compress_ratio")
        indexerHeadDim = try required(text.indexerHeadDim, "indexer_head_dim")
        indexerKVHeads = indexerKV
        indexerNHeads = try required(text.indexerNHeads, "indexer_n_heads")
        ngramSize = try required(text.ngramSize, "ngram_size")
        ngramVocabSizeBase = try required(text.ngramVocabSizeBase, "ngram_vocab_size_base")
        headsPerNgram = try required(text.headsPerNgram, "heads_per_ngram")
        ngramVocabSizeDivisor = try required(
            text.makeNgramVocabSizeDivisibleBy, "make_ngram_vocab_size_divisible_by")
        pleEmbedDim = try required(text.pleEmbedDim, "ple_embed_dim")
        pleConvKernelSize = try required(text.pleConvKernelSize, "ple_conv_kernel_size")
        pleLayerIds = text.pleLayerIds ?? []
        eosTokenId = try required(text.eosTokenId, "eos_token_id")
        seed = 1234

        guard indexerCompressRatio > 0, indexerBudget >= indexerCompressRatio else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: indexer_budget must be at least indexer_compress_ratio")
        }
        guard hcCount > 0, ngramSize >= 2 else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: hc_count must be positive and ngram_size at least 2")
        }
        guard pleEmbedDim % ((ngramSize - 1) * headsPerNgram) == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: ple_embed_dim must be divisible by the number of n-gram heads")
        }
    }
}

// MARK: - Norms

/// Qwen4ExpTextRMSNorm: x / rms(x) * (1 + weight). With `groupSize`, each group of channels is
/// normalized separately (used by the hyper-connection and PLE streams).
final class SharpAIQwen4ExpRMSNorm: Module {
    let eps: Float
    let groupSize: Int?

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, groupSize: Int? = nil, eps: Float) {
        self.eps = eps
        self.groupSize = groupSize
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        var normed: MLXArray
        if let groupSize {
            let shape = xf.shape
            let grouped = xf.reshaped(Array(shape.dropLast()) + [-1, groupSize])
            normed = grouped * rsqrt(grouped.square().mean(axis: -1, keepDims: true) + eps)
            normed = normed.reshaped(shape)
        } else {
            normed = xf * rsqrt(xf.square().mean(axis: -1, keepDims: true) + eps)
        }
        return (normed * (weight.asType(.float32) + Float(1))).asType(x.dtype)
    }
}

/// Qwen4ExpTextRMSNormGated: weight * rms(x) * act(gate), with weight initialized to ones.
/// The activation is sigmoid or silu, from output_gate_type.
final class SharpAIQwen4ExpGatedRMSNorm: Module {
    let eps: Float
    let useSiLU: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float, useSiLU: Bool) {
        self.eps = eps
        self.useSiLU = useSiLU
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let normed = (xf * rsqrt(xf.square().mean(axis: -1, keepDims: true) + eps)).asType(x.dtype)
        let gf = gate.asType(.float32)
        let activation = useSiLU ? silu(gf) : sigmoid(gf)
        let scaled = (weight * normed).asType(.float32) * activation
        return scaled.asType(x.dtype)
    }
}

// MARK: - Rotary embedding

/// Partial rotate-half RoPE over the first `rotaryDim` channels with theta from the config.
/// Text-only positions: the reference's three mrope axes all carry the same position, so the
/// frequencies reduce to a plain partial RoPE. Stored as plain values so the Module tree stays clean.
struct SharpAIQwen4ExpRope {
    let rotaryDim: Int
    let base: Float

    /// cos and sin for absolute positions, shape (positions.count, rotaryDim).
    func cosSin(positions: [Int], dtype: DType) -> (MLXArray, MLXArray) {
        let exponents = (0 ..< rotaryDim / 2).map { Float(2 * $0) / Float(rotaryDim) }
        let invFreq = MLXArray(Float(1)) / MLX.pow(MLXArray(base), MLXArray(exponents))
        let pos = MLXArray(positions.map { Float($0) })
        let freqs = expandedDimensions(pos, axis: 1) * expandedDimensions(invFreq, axis: 0)
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }
}

/// Rotates the first `cos.dim(-1)` channels of `x` (rotate-half) and passes the rest through.
func sharpAIQwen4ExpApplyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let d = cos.dim(-1)
    let half = d / 2
    let rot = x[.ellipsis, ..<d]
    let rest = x[.ellipsis, d...]
    let negSecond = MLXArray(Float(0)) - rot[.ellipsis, half...]
    let rotateHalf = concatenated([negSecond, rot[.ellipsis, ..<half]], axis: -1)
    let rotated = rot * cos + rotateHalf * sin
    return concatenated([rotated, rest], axis: -1)
}

// MARK: - Gated DeltaNet (linear attention)

/// Qwen4ExpTextGatedDeltaNet. Separate in_proj_qkv / in_proj_z / in_proj_b / in_proj_a projections,
/// a causal depthwise conv with silu, the gated delta rule, and a gated RMS norm whose gate is
/// sigmoid (output_gate_type) rather than silu.
final class SharpAIQwen4ExpGatedDeltaNet: Module {
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: SharpAIQwen4ExpGatedRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ s: SharpAIQwen4ExpSettings) {
        numVHeads = s.linearNumValueHeads
        numKHeads = s.linearNumKeyHeads
        headKDim = s.linearKeyHeadDim
        headVDim = s.linearValueHeadDim
        keyDim = headKDim * numKHeads
        valueDim = headVDim * numVHeads
        convKernelSize = s.linearConvKernelDim
        convDim = keyDim * 2 + valueDim

        _inProjQKV.wrappedValue = Linear(s.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(s.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(s.hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(s.hiddenSize, numVHeads, bias: false)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernelSize,
            stride: 1, padding: 0, dilation: 1, groups: convDim, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLX.log(MLXRandom.uniform(low: Float(0.01), high: Float(16), [numVHeads]))
        _norm.wrappedValue = SharpAIQwen4ExpGatedRMSNorm(
            dimensions: headVDim, eps: s.rmsNormEps, useSiLU: s.outputGateSiLU)
        _outProj.wrappedValue = Linear(valueDim, s.hiddenSize, bias: false)

        super.init()
    }

    /// Returns (output, new conv state, new recurrent state). `convState` is (B, k-1, convDim).
    func callAsFunction(
        _ inputs: MLXArray, convState: MLXArray?, recState: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        let mixedQKV = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        // Causal depthwise conv: the reference pads with zeros, or with the cached last k-1 inputs.
        let prev = convState ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: mixedQKV.dtype)
        let convInput = concatenated([prev, mixedQKV], axis: 1)
        let newConvState = contiguous(
            convInput[0..., (convInput.dim(1) - (convKernelSize - 1))..., 0...])
        let conved = silu(conv1d(convInput))

        let split = MLX.split(conved, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = split[0].reshaped(B, S, numKHeads, headKDim)
        var k = split[1].reshaped(B, S, numKHeads, headKDim)
        let v = split[2].reshaped(B, S, numVHeads, headVDim)

        // l2norm(q) / sqrt(dk) and l2norm(k), matching the reference's use_qk_l2norm_in_kernel path.
        let dtype = inputs.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qkEps = 1e-6 / Float(headKDim)
        q = MLXArray(invScale * invScale).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: qkEps)
        k = MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: qkEps)

        let (out, newRecState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: recState, mask: nil, useKernel: !training)

        let normed = norm(out, gate: z)
        return (outProj(normed.reshaped(B, S, -1)), newConvState, newRecState)
    }
}

// MARK: - QSA indexer and full attention

/// Qwen4ExpTextQSAIndexer. Selects, for each query, the top `budget / ratio` complete key blocks by
/// relu(q . pooled_block) summed over indexer heads, and always keeps the tail tokens that do not
/// fill a block. The result is a boolean mask over the cache, combined with causality.
final class SharpAIQwen4ExpIndexer: Module {
    let nHeads: Int
    let headDim: Int
    let blockTopK: Int
    let ratio: Int

    @ModuleInfo(key: "index_qk_proj") var qkProj: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: SharpAIQwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: SharpAIQwen4ExpRMSNorm

    init(_ s: SharpAIQwen4ExpSettings) {
        nHeads = s.indexerNHeads
        headDim = s.indexerHeadDim
        ratio = s.indexerCompressRatio
        blockTopK = s.indexerBudget / s.indexerCompressRatio
        _qkProj.wrappedValue = Linear(
            s.hiddenSize, (s.indexerNHeads + s.indexerKVHeads) * s.indexerHeadDim, bias: false)
        _qNorm.wrappedValue = SharpAIQwen4ExpRMSNorm(dimensions: s.indexerHeadDim, eps: s.rmsNormEps)
        _kNorm.wrappedValue = SharpAIQwen4ExpRMSNorm(dimensions: s.indexerHeadDim, eps: s.rmsNormEps)
        super.init()
    }

    /// Boolean mask (B, 1, L, S) over the keys after this step's update. `keyCache` stores the raw,
    /// pre-norm, pre-rope indexer keys, as the reference's update_indexer does.
    func selectedMask(
        _ hidden: MLXArray, rope: SharpAIQwen4ExpRope, offset: Int, keyCache: KVCache?
    ) -> MLXArray {
        let B = hidden.dim(0)
        let L = hidden.dim(1)
        let qSize = nHeads * headDim

        let qk = qkProj(hidden)
        let qNormed = qNorm(qk[.ellipsis, ..<qSize].reshaped(B, L, nHeads, headDim))
        let rawKeys = qk[.ellipsis, qSize...].reshaped(B, L, headDim)

        let qPositions = (0 ..< L).map { offset + $0 }
        let (cosQ, sinQ) = rope.cosSin(positions: qPositions, dtype: hidden.dtype)
        let q = sharpAIQwen4ExpApplyRope(
            qNormed, cos: cosQ.reshaped(1, L, 1, -1), sin: sinQ.reshaped(1, L, 1, -1))

        var keys: MLXArray
        if let keyCache {
            let raw4 = rawKeys.reshaped(B, 1, L, headDim)
            let (all, _) = keyCache.update(keys: raw4, values: raw4)
            keys = all.reshaped(B, all.dim(2), headDim)
        } else {
            keys = rawKeys
        }
        let S = keys.dim(1)
        let numBlocks = S / ratio

        // Complete blocks visible to each query (causal, no padding): (qpos + 1) / ratio.
        let ncList = qPositions.map { ($0 + 1) / ratio }
        let ncArr = MLXArray(ncList.map { Int32($0) })

        var selectedBlock = MLXArray.zeros([B, L, numBlocks], dtype: .bool)
        if numBlocks > 0 {
            let blocks = keys[0..., ..<(numBlocks * ratio), 0...].reshaped(B, numBlocks, ratio, headDim)
            let meanKeys = blocks.asType(.float32).mean(axis: 2).asType(hidden.dtype)
            let pooledNormed = kNorm(meanKeys)
            let starts = (0 ..< numBlocks).map { $0 * ratio }
            let (cosK, sinK) = rope.cosSin(positions: starts, dtype: hidden.dtype)
            let pooled = sharpAIQwen4ExpApplyRope(
                pooledNormed, cos: cosK.reshaped(1, numBlocks, -1), sin: sinK.reshaped(1, numBlocks, -1))

            let qFlat = q.asType(.float32).reshaped(B, L * nHeads, headDim)
            let dots = matmul(qFlat, pooled.asType(.float32).transposed(0, 2, 1))
            let relu = MLX.maximum(dots, MLXArray(Float(0)))
            let scores = relu.reshaped(B, L, nHeads, numBlocks).sum(axis: 2)
                / Float(headDim).squareRoot()

            let blockIdx = MLXArray((0 ..< numBlocks).map { Int32($0) })
            let validBlock = less(blockIdx.reshaped(1, 1, numBlocks), ncArr.reshaped(1, L, 1))
            // Invalid blocks get -inf, so they rank after every valid block (valid scores are >= 0).
            let masked = MLX.where(validBlock, scores, MLXArray(-Float.infinity))
            let order = argSort(MLXArray(Float(0)) - masked, axis: -1)
            let rank = argSort(order, axis: -1)
            let k = min(blockTopK, numBlocks)
            selectedBlock = logicalAnd(less(rank, MLXArray(UInt32(k))), validBlock)
        }

        // Token mask: tokens inside a complete block follow that block's selection, tail tokens and
        // everything else visible are kept. Columns past the last block read a false padding column.
        let padded = concatenated(
            [selectedBlock, MLXArray.zeros([B, L, 1], dtype: .bool)], axis: -1)
        let jIdx = MLXArray((0 ..< S).map { Int32($0) })
        let blockOfToken = MLXArray((0 ..< S).map { Int32($0 / ratio) }).reshaped(1, 1, S)
        let ncTokens = MLXArray(ncList.map { Int32($0 * ratio) }).reshaped(1, L, 1)
        let inBlock = less(jIdx.reshaped(1, 1, S), ncTokens)
        let qPosArr = MLXArray(qPositions.map { Int32($0) }).reshaped(1, L, 1)
        let causal = lessEqual(jIdx.reshaped(1, 1, S), qPosArr)
        let gatherIdx = MLX.where(inBlock, blockOfToken, MLXArray(Int32(numBlocks)))
        let gathered = takeAlong(padded, gatherIdx, axis: -1)
        let allowed = MLX.where(inBlock, gathered, causal)
        return allowed.reshaped(B, 1, L, S)
    }
}

/// Qwen4ExpTextAttention: GQA with per-head q/k RMS norm, partial RoPE, a sigmoid output gate
/// taken from the query projection, and the indexer mask.
final class SharpAIQwen4ExpFullAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIQwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIQwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") var indexer: SharpAIQwen4ExpIndexer

    init(_ s: SharpAIQwen4ExpSettings) {
        numHeads = s.numAttentionHeads
        numKVHeads = s.numKeyValueHeads
        headDim = s.headDim
        scale = pow(Float(s.headDim), -0.5)
        _qProj.wrappedValue = Linear(
            s.hiddenSize, s.numAttentionHeads * s.headDim * 2, bias: s.attentionBias)
        _kProj.wrappedValue = Linear(
            s.hiddenSize, s.numKeyValueHeads * s.headDim, bias: s.attentionBias)
        _vProj.wrappedValue = Linear(
            s.hiddenSize, s.numKeyValueHeads * s.headDim, bias: s.attentionBias)
        _oProj.wrappedValue = Linear(
            s.numAttentionHeads * s.headDim, s.hiddenSize, bias: s.attentionBias)
        _qNorm.wrappedValue = SharpAIQwen4ExpRMSNorm(dimensions: s.headDim, eps: s.rmsNormEps)
        _kNorm.wrappedValue = SharpAIQwen4ExpRMSNorm(dimensions: s.headDim, eps: s.rmsNormEps)
        _indexer.wrappedValue = SharpAIQwen4ExpIndexer(s)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, rope: SharpAIQwen4ExpRope, kvCache: KVCache?, indexerKeyCache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let offset = kvCache?.offset ?? 0

        let allowed = indexer.selectedMask(
            x, rope: rope, offset: offset, keyCache: indexerKeyCache)

        let qGate = qProj(x).reshaped(B, L, numHeads, 2 * headDim)
        let parts = MLX.split(qGate, indices: [headDim], axis: -1)
        let gate = parts[1].reshaped(B, L, numHeads * headDim)
        var q = qNorm(parts[0]).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let (cos, sin) = rope.cosSin(positions: (0 ..< L).map { offset + $0 }, dtype: x.dtype)
        let cos4 = cos.reshaped(1, 1, L, -1)
        let sin4 = sin.reshaped(1, 1, L, -1)
        q = sharpAIQwen4ExpApplyRope(q, cos: cos4, sin: sin4)
        k = sharpAIQwen4ExpApplyRope(k, cos: cos4, sin: sin4)

        if let kvCache {
            (k, v) = kvCache.update(keys: k, values: v)
        }
        let repeatFactor = numHeads / numKVHeads
        if repeatFactor > 1 {
            k = MLX.repeated(k, count: repeatFactor, axis: 1)
            v = MLX.repeated(v, count: repeatFactor, axis: 1)
        }

        let additive = MLX.where(
            allowed, MLXArray(Float(0)), MLXArray(-Float.infinity)
        ).asType(q.dtype)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(additive))

        let merged = attended.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim)
        return oProj(merged * sigmoid(gate))
    }
}

// MARK: - MoE and hyper-connections

final class SharpAIQwen4ExpMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Qwen4ExpTextSparseMoeBlock: softmax router with top-k renormalization, routed SwitchGLU experts,
/// and a sigmoid-gated shared expert.
final class SharpAIQwen4ExpSparseMoe: Module {
    let numExperts: Int
    let topK: Int
    let normTopKProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: SharpAIQwen4ExpMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ s: SharpAIQwen4ExpSettings) {
        numExperts = s.numExperts
        topK = s.numExpertsPerTok
        normTopKProb = s.normTopKProb
        _gate.wrappedValue = Linear(s.hiddenSize, s.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: s.hiddenSize, hiddenDims: s.moeIntermediateSize, numExperts: s.numExperts)
        _sharedExpert.wrappedValue = SharpAIQwen4ExpMLP(
            hiddenSize: s.hiddenSize, intermediateSize: s.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(s.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        let H = x.dim(2)
        let flat = x.reshaped(-1, H)

        let probs = MLX.softmax(gate(flat).asType(.float32), axis: -1, precise: true)
        let kth = numExperts - topK
        let inds = argPartition(probs, kth: kth, axis: -1)[.ellipsis, kth...]
        var scores = takeAlong(probs, inds, axis: -1)
        if normTopKProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        let expertOut = switchMLP(flat, inds)
        let routed = (expertOut * expandedDimensions(scores.asType(expertOut.dtype), axis: -1))
            .sum(axis: -2)

        let shared = sigmoid(sharedExpertGate(flat)) * sharedExpert(flat)
        return (routed + shared).reshaped(B, S, H)
    }
}

/// Qwen4ExpTextGatedResidual. Mixes `hc` residual streams into one input (`mixed`), and, when
/// `useCombine` is set, also returns the per-stream injection weights used to write a block's output
/// back into the streams.
final class SharpAIQwen4ExpGatedResidual: Module {
    let hcCount: Int
    let hiddenSize: Int

    @ModuleInfo(key: "hc_norm") var hcNorm: SharpAIQwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var mixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var mixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInject: Linear?

    init(_ s: SharpAIQwen4ExpSettings, useCombine: Bool) {
        hcCount = s.hcCount
        hiddenSize = s.hiddenSize
        let streamSize = s.hcCount * s.hiddenSize
        _hcNorm.wrappedValue = SharpAIQwen4ExpRMSNorm(
            dimensions: streamSize, groupSize: s.hiddenSize, eps: s.rmsNormEps)
        _mixDown.wrappedValue = Linear(streamSize, s.hcLowrank, bias: false)
        _mixUp.wrappedValue = Linear(s.hcLowrank, streamSize, bias: false)
        _blockInject.wrappedValue = useCombine ? Linear(streamSize, s.hcCount, bias: false) : nil
        super.init()
    }

    /// (B, S, hc*H) -> (mixed (B, S, H), injection weights (B, S, hc)) when combine is enabled.
    func callAsFunction(_ hyperInput: MLXArray) -> (MLXArray, MLXArray?) {
        let B = hyperInput.dim(0)
        let S = hyperInput.dim(1)
        let normed = hcNorm(hyperInput)
        var mix = silu(mixDown(normed) / Float(hcCount))
        mix = sigmoid(mixUp(mix))
        let mixed = (mix.reshaped(B, S, hcCount, hiddenSize)
            * normed.reshaped(B, S, hcCount, hiddenSize)).mean(axis: 2)
        guard let blockInject else { return (mixed, nil) }
        let injection = Float(2) * sigmoid(blockInject(normed) / Float(hcCount))
        return (mixed, injection)
    }
}

// MARK: - Decoder layer and text model

/// Qwen4ExpTextDecoderLayer. Layer caches:
///   linear_attention: ArraysCache slots [conv, recurrent] (+ [ple ctx, ple conv] on the PLE layer)
///   full_attention:   CacheList(KV, indexer keys) (+ ArraysCache [ple ctx, ple conv] on the PLE layer)
final class SharpAIQwen4ExpDecoderLayer: Module {
    let isLinear: Bool
    let hasPLE: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: SharpAIQwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIQwen4ExpFullAttention?
    @ModuleInfo(key: "ple") var ple: SharpAIQwen4ExpPLE?
    @ModuleInfo(key: "attn_hyper_connection") var attnHC: SharpAIQwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHC: SharpAIQwen4ExpGatedResidual
    @ModuleInfo(key: "mlp") var mlp: SharpAIQwen4ExpSparseMoe

    init(_ s: SharpAIQwen4ExpSettings, layerIndex: Int) throws {
        isLinear = s.layerTypes[layerIndex] == "linear_attention"
        if isLinear {
            _linearAttn.wrappedValue = SharpAIQwen4ExpGatedDeltaNet(s)
            _selfAttn.wrappedValue = nil
        } else {
            _linearAttn.wrappedValue = nil
            _selfAttn.wrappedValue = SharpAIQwen4ExpFullAttention(s)
        }
        let pleIndex = s.pleLayerIds.firstIndex(of: layerIndex + 1)
        hasPLE = pleIndex != nil
        if let pleIndex {
            _ple.wrappedValue = SharpAIQwen4ExpPLE(s, pleLayerIndex: pleIndex)
        } else {
            _ple.wrappedValue = nil
        }
        _attnHC.wrappedValue = SharpAIQwen4ExpGatedResidual(s, useCombine: true)
        _mlpHC.wrappedValue = SharpAIQwen4ExpGatedResidual(s, useCombine: true)
        _mlp.wrappedValue = SharpAIQwen4ExpSparseMoe(s)
        super.init()
    }

    /// The cache for this layer, in the layout described above.
    func makeCache() -> KVCache {
        if isLinear {
            return ArraysCache(size: hasPLE ? 4 : 2)
        }
        if hasPLE {
            return CacheList(KVCacheSimple(), KVCacheSimple(), ArraysCache(size: 2))
        }
        return CacheList(KVCacheSimple(), KVCacheSimple())
    }

    func callAsFunction(
        _ hidden: MLXArray, inputIds: MLXArray, rope: SharpAIQwen4ExpRope, cache: KVCache?
    ) -> MLXArray {
        let B = hidden.dim(0)
        let S = hidden.dim(1)
        let streamSize = hidden.dim(2)

        // Resolve this layer's cache slots.
        var recurrentSlots: ArraysCache?
        var pleSlots: ArraysCache?
        var pleBase = 0
        var kvCache: KVCache?
        var indexerKeyCache: KVCache?
        if isLinear {
            recurrentSlots = cache as? ArraysCache
            pleSlots = recurrentSlots
            pleBase = 2
        } else if let list = cache as? CacheList {
            kvCache = list[0]
            indexerKeyCache = list[1]
            if hasPLE {
                pleSlots = list[2] as? ArraysCache
            }
        }

        var h = hidden
        if let ple {
            let (pleOut, newContext, newConv) = ple(
                h, inputIds: inputIds,
                context: pleSlots?[pleBase], convState: pleSlots?[pleBase + 1])
            if let pleSlots {
                pleSlots[pleBase] = newContext
                pleSlots[pleBase + 1] = newConv
            }
            h = h + pleOut
        }

        let attnHyperInput = h
        let (attnInput, attnInjection) = attnHC(h)
        guard let attnInjection else {
            preconditionFailure("qwen4_exp: attention hyper-connection requires block_inject_weight")
        }
        let attnOut: MLXArray
        if isLinear, let linearAttn {
            let (out, newConv, newRec) = linearAttn(
                attnInput,
                convState: recurrentSlots?[0],
                recState: recurrentSlots?[1])
            recurrentSlots?[0] = newConv
            recurrentSlots?[1] = newRec
            attnOut = out
        } else if let selfAttn {
            attnOut = selfAttn(
                attnInput, rope: rope, kvCache: kvCache, indexerKeyCache: indexerKeyCache)
        } else {
            preconditionFailure("qwen4_exp: layer has no attention block")
        }
        let attnWritten = (attnOut.expandedDimensions(axis: -2)
            * attnInjection.expandedDimensions(axis: -1)).reshaped(B, S, streamSize)
        h = attnHyperInput + attnWritten

        let mlpHyperInput = h
        let (mlpInput, mlpInjection) = mlpHC(h)
        guard let mlpInjection else {
            preconditionFailure("qwen4_exp: MLP hyper-connection requires block_inject_weight")
        }
        let mlpOut = mlp(mlpInput)
        let mlpWritten = (mlpOut.expandedDimensions(axis: -2)
            * mlpInjection.expandedDimensions(axis: -1)).reshaped(B, S, streamSize)
        h = mlpHyperInput + mlpWritten
        return h
    }
}

/// Qwen4ExpTextModel: embeddings, the hc-stream expansion, the decoder stack, and the final mixer.
final class SharpAIQwen4ExpTextModel: Module {
    let hcCount: Int
    let rope: SharpAIQwen4ExpRope

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIQwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperMixer: SharpAIQwen4ExpGatedResidual

    init(_ s: SharpAIQwen4ExpSettings) throws {
        hcCount = s.hcCount
        rope = SharpAIQwen4ExpRope(rotaryDim: s.rotaryDim, base: s.ropeTheta)
        _embedTokens.wrappedValue = Embedding(embeddingCount: s.vocabSize, dimensions: s.hiddenSize)
        _layers.wrappedValue = try (0 ..< s.numHiddenLayers).map {
            try SharpAIQwen4ExpDecoderLayer(s, layerIndex: $0)
        }
        _hyperMixer.wrappedValue = SharpAIQwen4ExpGatedResidual(s, useCombine: false)
        super.init()
    }

    func makeCaches() -> [KVCache] {
        layers.map { $0.makeCache() }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        h = concatenated(Array(repeating: h, count: hcCount), axis: -1)
        for (index, layer) in layers.enumerated() {
            h = layer(h, inputIds: inputs, rope: rope, cache: cache?[index])
        }
        return hyperMixer(h).0
    }
}

/// Wraps the text model under the checkpoint prefix `model.language_model.*`.
final class SharpAIQwen4ExpBackbone: Module {
    @ModuleInfo(key: "language_model") var languageModel: SharpAIQwen4ExpTextModel

    init(_ s: SharpAIQwen4ExpSettings) throws {
        _languageModel.wrappedValue = try SharpAIQwen4ExpTextModel(s)
        super.init()
    }
}

/// Qwen4ExpForCausalLM (text path). Parameter names follow the checkpoint, e.g.
/// `model.language_model.layers.N.linear_attn.in_proj_qkv.weight` and `lm_head.weight`.
final class SharpAIQwen4ExpForCausalLM: Module, LanguageModel {
    @ModuleInfo(key: "model") var model: SharpAIQwen4ExpBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ configuration: SharpAIQwen4ExpConfiguration) throws {
        let settings = try SharpAIQwen4ExpSettings(configuration.textConfig)
        _model.wrappedValue = try SharpAIQwen4ExpBackbone(settings)
        _lmHead.wrappedValue = Linear(settings.hiddenSize, settings.vocabSize, bias: false)
        super.init()
    }

    /// Text-only path: the whole prompt is handed to the TokenIterator, which evaluates it through
    /// `callAsFunction`. The hyper-connection and n-gram state is carried by the cache and the
    /// token ids, so no model-side chunking is needed.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model.languageModel(inputs, cache: cache))
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.languageModel.makeCaches()
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIQwen4ExpSanitizer.sanitize(weights)
    }
}
