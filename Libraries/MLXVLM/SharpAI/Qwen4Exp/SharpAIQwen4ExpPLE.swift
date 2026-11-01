// SharpAI fork overlay: qwen4_exp per-layer embedding (PLE) and hashed n-gram embedding. Not part
// of upstream mlx-swift-lm.
//
// Port of Qwen4ExpTextNGramEmbedding, Qwen4ExpTextPLELayer and the module-level helpers
// _splitmix64, _build_layer_multipliers, _is_prime, _find_nth_prime_after and
// _shift_right_ignore_eos from modeling_qwen4_exp.py.
// Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
//
// The n-gram hash runs on the CPU in exact Int64 arithmetic (wrapping multiply, XOR, floor
// remainder), because MLX Swift does not expose a bitwise XOR op in this checkout. The multipliers
// and head sizes are functions of the config. They match the checkpoint's buffers
// (model.language_model.layers.1.ple.ple_embedding.layer_multipliers, ngram_heads_vocab_sizes,
// ngram_heads_offsets) for the released Qwen3.8-Flash-Next config, so the sanitizer drops those buffers.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Pure integer helpers from the reference. Public so that tests can check them against the
/// checkpoint's stored buffers.
public enum SharpAIQwen4ExpNGramHashing {

    private static let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    private static let mix1: UInt64 = 0xBF58_476D_1CE4_E5B9
    private static let mix2: UInt64 = 0x94D0_49BB_1331_11EB
    private static let prime1 = 10007

    /// _splitmix64 with 64-bit wrapping arithmetic.
    public static func splitMix64(_ input: UInt64) -> UInt64 {
        var value = input &+ gamma
        value = (value ^ (value >> 30)) &* mix1
        value = (value ^ (value >> 27)) &* mix2
        return value ^ (value >> 31)
    }

    /// _build_layer_multipliers: odd multipliers in [1, 2 * half_bound), one per n-gram position.
    public static func layerMultipliers(
        unigramVocabSize: Int, ngramSize: Int, pleLayerIndex: Int, seed: Int
    ) -> [Int64] {
        let multiplierMax = Int.max / max(unigramVocabSize, 1)
        let halfBound = UInt64(max(1, multiplierMax / 2))
        let baseSeed = UInt64(seed + prime1 * pleLayerIndex)
        return (0 ..< ngramSize).map { index in
            let value = baseSeed &+ gamma &* UInt64(index + 1)
            let draw = splitMix64(value) % halfBound
            return Int64(2 * draw + 1)
        }
    }

    public static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    /// _find_nth_prime_after: the `count`-th prime strictly greater than `start`.
    public static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0 ..< count {
            prime += 1
            while !isPrime(prime) {
                prime += 1
            }
        }
        return prime
    }

    /// Vocabulary size of each n-gram head on `pleLayerIndex`. Head h of that layer is global head
    /// `pleLayerIndex * ngramHeads + h`, and its size is the (global + 1)-th prime after `base - 1`.
    public static func headVocabSizes(base: Int, ngramHeads: Int, pleLayerIndex: Int) -> [Int] {
        (0 ..< ngramHeads).map { head in
            nthPrimeAfter(base - 1, count: pleLayerIndex * ngramHeads + head + 1)
        }
    }

    /// Exclusive prefix sums of the head sizes, used as the per-head row offsets.
    public static func headVocabOffsets(_ sizes: [Int]) -> [Int] {
        var running = 0
        return sizes.map { size in
            defer { running += size }
            return running
        }
    }

    /// _shift_right_ignore_eos for one sequence. Position p receives tokens[p - shift] when that
    /// token lies in the same EOS-delimited segment (at least `shift` tokens after its last EOS),
    /// and EOS otherwise.
    public static func shiftRightIgnoreEOS(_ tokens: [Int64], shift: Int, eos: Int64) -> [Int64] {
        if shift == 0 { return tokens }
        var out = [Int64](repeating: eos, count: tokens.count)
        var lastEOSBefore = -1
        for p in 0 ..< tokens.count {
            let positionInSegment = p - (lastEOSBefore + 1)
            let source = p - shift
            if positionInSegment >= shift && source >= 0 {
                out[p] = tokens[source]
            }
            if tokens[p] == eos {
                lastEOSBefore = p
            }
        }
        return out
    }
}

/// Qwen4ExpTextNGramEmbedding: hashed n-gram ids over the token history (with a carried context
/// of `ngram_size - 1` tokens), gathered from one table that stores every head's rows back to back.
final class SharpAIQwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let contextLength: Int
    let headsPerNgram: Int
    let ngramHeads: Int
    let eosTokenId: Int64
    let multipliers: [Int64]
    let headVocabSizes: [Int64]
    let headOffsets: [Int64]

    @ModuleInfo(key: "ngram_embedding") var table: Embedding

    init(_ s: SharpAIQwen4ExpSettings, pleLayerIndex: Int) {
        ngramSize = s.ngramSize
        contextLength = s.ngramSize - 1
        headsPerNgram = s.headsPerNgram
        ngramHeads = (s.ngramSize - 1) * s.headsPerNgram
        eosTokenId = Int64(s.eosTokenId)
        multipliers = SharpAIQwen4ExpNGramHashing.layerMultipliers(
            unigramVocabSize: s.vocabSize, ngramSize: s.ngramSize,
            pleLayerIndex: pleLayerIndex, seed: s.seed)
        let sizes = SharpAIQwen4ExpNGramHashing.headVocabSizes(
            base: s.ngramVocabSizeBase, ngramHeads: ngramHeads, pleLayerIndex: pleLayerIndex)
        headVocabSizes = sizes.map { Int64($0) }
        headOffsets = SharpAIQwen4ExpNGramHashing.headVocabOffsets(sizes).map { Int64($0) }

        let total = sizes.reduce(0, +)
        let divisor = s.ngramVocabSizeDivisor
        let padded = (total + divisor - 1) / divisor * divisor
        _table.wrappedValue = Embedding(
            embeddingCount: padded, dimensions: s.pleEmbedDim / ngramHeads)
        super.init()
    }

    /// `inputIds` is (B, S). `context` is the (B, contextLength) history carried from earlier steps,
    /// or nil for the start of a sequence, which is padded with EOS. Returns the n-gram embeddings
    /// (B, S, pleEmbedDim) and the new context.
    func callAsFunction(
        _ inputIds: MLXArray, context: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let B = inputIds.dim(0)
        let S = inputIds.dim(1)
        let ids = inputIds.asType(.int64).asArray(Int64.self)
        let ctxFlat: [Int64] =
            context.map { $0.asType(.int64).asArray(Int64.self) }
            ?? [Int64](repeating: eosTokenId, count: B * contextLength)

        var columns = [Int32](repeating: 0, count: B * S * ngramHeads)
        var newContext = [Int64]()
        newContext.reserveCapacity(B * contextLength)

        for b in 0 ..< B {
            let history = Array(ctxFlat[(b * contextLength) ..< ((b + 1) * contextLength)])
                + Array(ids[(b * S) ..< ((b + 1) * S)])
            let total = history.count
            let shifted = (0 ..< ngramSize).map {
                SharpAIQwen4ExpNGramHashing.shiftRightIgnoreEOS(
                    history, shift: $0, eos: eosTokenId)
            }
            for t in (total - S) ..< total {
                for order in 2 ... ngramSize {
                    var mixed = shifted[0][t] &* multipliers[0]
                    for position in 1 ..< order {
                        mixed ^= shifted[position][t] &* multipliers[position]
                    }
                    let start = (order - 2) * headsPerNgram
                    for head in 0 ..< headsPerNgram {
                        let size = headVocabSizes[start + head]
                        var remainder = mixed % size
                        if remainder < 0 { remainder += size }
                        let row = remainder + headOffsets[start + head]
                        columns[((b * S) + (t - (total - S))) * ngramHeads + start + head] =
                            Int32(row)
                    }
                }
            }
            newContext.append(contentsOf: history.suffix(contextLength))
        }

        let rowIds = MLXArray(columns, [B, S, ngramHeads])
        let embedded = table(rowIds).reshaped(B, S, -1)
        return (embedded, MLXArray(newContext, [B, contextLength]))
    }
}

/// Qwen4ExpTextPLELayer. Gates the hashed n-gram value by how well each hc stream matches it, adds a
/// dilated causal short conv over those gated values, and returns the result (B, S, hc*H).
/// The short conv keeps the last `(kernel - 1) * ngram_size` gated inputs as its state.
final class SharpAIQwen4ExpPLE: Module {
    let hiddenSize: Int
    let hcCount: Int
    let shortConvStateLength: Int

    @ModuleInfo(key: "ple_embedding") var pleEmbedding: SharpAIQwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: SharpAIQwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: SharpAIQwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: SharpAIQwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(_ s: SharpAIQwen4ExpSettings, pleLayerIndex: Int) {
        hiddenSize = s.hiddenSize
        hcCount = s.hcCount
        shortConvStateLength = (s.pleConvKernelSize - 1) * s.ngramSize
        let streamSize = s.hiddenSize * s.hcCount
        _pleEmbedding.wrappedValue = SharpAIQwen4ExpNGramEmbedding(s, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(s.pleEmbedDim, streamSize, bias: false)
        _valueProj.wrappedValue = Linear(s.pleEmbedDim, s.hiddenSize, bias: false)
        _normKey.wrappedValue = SharpAIQwen4ExpRMSNorm(
            dimensions: streamSize, groupSize: s.hiddenSize, eps: s.rmsNormEps)
        _normQuery.wrappedValue = SharpAIQwen4ExpRMSNorm(
            dimensions: streamSize, groupSize: s.hiddenSize, eps: s.rmsNormEps)
        _normConv.wrappedValue = SharpAIQwen4ExpRMSNorm(
            dimensions: streamSize, groupSize: s.hiddenSize, eps: s.rmsNormEps)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: streamSize, outputChannels: streamSize,
            kernelSize: s.pleConvKernelSize, stride: 1, padding: 0,
            dilation: s.ngramSize, groups: streamSize, bias: false)
        super.init()
    }

    /// Returns (output (B, S, hc*H), new n-gram context, new conv state (B, shortLen, hc*H)).
    func callAsFunction(
        _ hidden: MLXArray, inputIds: MLXArray, context: MLXArray?, convState: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let B = hidden.dim(0)
        let S = hidden.dim(1)
        let streamSize = hiddenSize * hcCount

        let (embeddings, newContext) = pleEmbedding(inputIds, context: context)

        let keyNormed = normKey(keyProj(embeddings)).reshaped(B, S, hcCount, hiddenSize)
        let value = valueProj(embeddings)
        let queryNormed = normQuery(hidden).reshaped(B, S, hcCount, hiddenSize)
        var gate = (keyNormed * queryNormed).sum(axis: -1, keepDims: true)
            / Float(hiddenSize).squareRoot()
        gate = MLX.sqrt(MLX.maximum(MLX.abs(gate), MLXArray(Float(1e-6)))) * MLX.sign(gate)

        let gatedValue = sigmoid(gate) * expandedDimensions(value, axis: -2)
        let gatedFlat = gatedValue.reshaped(B, S, streamSize)
        let gatedNormed = normConv(gatedFlat)

        // Dilated causal conv: the input is the carried state (zeros at the start) then the new tokens.
        let prev = convState
            ?? MLXArray.zeros([B, shortConvStateLength, streamSize], dtype: gatedNormed.dtype)
        let convInput = concatenated([prev, gatedNormed], axis: 1)
        let newConvState = contiguous(
            convInput[0..., (convInput.dim(1) - shortConvStateLength)..., 0...])
        let conved = silu(conv1d(convInput))

        return (gatedFlat + conved, newContext, newConvState)
    }
}
