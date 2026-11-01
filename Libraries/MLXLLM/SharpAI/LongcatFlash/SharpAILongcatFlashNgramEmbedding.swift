// SharpAI fork overlay: hashed n-gram embedding for longcat_flash.
// Mirrors NgramEmbedding in mlx-lm longcat_flash_ngram.py. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAILongcatFlashNgramEmbedding: Module {
    let vocabSize: Int
    let ngramVocabSize: Int
    let splitNum: Int
    let neighborNum: Int
    /// Indexed by `(order - 2) * splitNum + split`. Entry `t` is `base^(t+1) mod dim`.
    let vocabMods: [[Int]]

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    var embedders: [Embedding]
    var post_projs: [Linear]

    init(_ config: SharpAILongcatFlashConfiguration) {
        let vocab = config.vocabSize
        let hidden = config.hiddenSize
        let ngramVocab = config.ngramVocabSize
        let split = config.embSplitNum
        let neighbors = config.embNeighborNum
        let count = config.ngramEmbedderCount
        let embDim = hidden / count

        self.vocabSize = vocab
        self.ngramVocabSize = ngramVocab
        self.splitNum = split
        self.neighborNum = neighbors
        self.vocabMods = (0 ..< count).map { index in
            let order = index / split + 2
            let dim = ngramVocab + index * 2 + 1
            var power = 1
            var mods: [Int] = []
            for _ in 0 ..< (order - 1) {
                power = (power * vocab) % dim
                mods.append(power)
            }
            return mods
        }
        self._wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: vocab, dimensions: hidden)
        self.embedders = (0 ..< count).map { index in
            Embedding(embeddingCount: ngramVocab + index * 2 + 1, dimensions: embDim)
        }
        self.post_projs = (0 ..< count).map { _ in
            Linear(embDim, hidden, bias: false)
        }
    }

    /// `cache` holds the last `neighborNum - 1` token ids from earlier calls.
    func callAsFunction(_ inputIds: MLXArray, cache: ArraysCache?) -> MLXArray {
        let seqLen = inputIds.dim(-1)
        let tokens = inputIds.asType(.int64)

        var context = tokens
        if let cache {
            if let previous = cache[0] {
                context = concatenated([previous, tokens], axis: -1)
            }
            let keep = max(0, context.dim(-1) - neighborNum + 1)
            cache[0] = context[.ellipsis, keep ..< context.dim(-1)]
        }

        var x = wordEmbeddings(tokens)

        var shifted: [Int: MLXArray] = [:]
        for order in 2 ... neighborNum {
            shifted[order] = shiftRight(context, by: order - 1)
        }

        let total = context.dim(-1)
        for order in 2 ... neighborNum {
            for split in 0 ..< splitNum {
                let index = (order - 2) * splitNum + split
                let dim = ngramVocabSize + index * 2 + 1
                let mods = vocabMods[index]

                var ngramIds = context
                for k in 2 ... order {
                    ngramIds = ngramIds + shifted[k]! * mods[k - 2]
                }
                let newIds = (ngramIds % dim)[.ellipsis, (total - seqLen) ..< total]
                x = x + post_projs[index](embedders[index](newIds))
            }
        }

        return x / Float(1 + splitNum * (neighborNum - 1))
    }

    /// Shifts a `[batch, seq]` array right by `n` positions, filling with zeros.
    private func shiftRight(_ x: MLXArray, by n: Int) -> MLXArray {
        if n <= 0 {
            return x
        }
        let batch = x.dim(0)
        let seq = x.dim(1)
        if seq <= n {
            return MLXArray.zeros(like: x)
        }
        let pad = MLXArray.zeros([batch, n], dtype: x.dtype)
        return concatenated([pad, x[.ellipsis, ..<(seq - n)]], axis: -1)
    }
}
