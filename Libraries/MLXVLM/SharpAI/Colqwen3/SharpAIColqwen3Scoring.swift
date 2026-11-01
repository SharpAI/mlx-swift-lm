import MLX

/// Late-interaction (MaxSim) scoring over ColQwen3 multi-vector embeddings.
///
/// UNVERIFIED: the score is the ColBERT MaxSim definition (sum over query tokens of the
/// max dot product). The model's own reference scorer is not on disk to compare against.
public enum SharpAIColqwen3Scoring {

    /// MaxSim score for one query and one document.
    ///
    /// Pass only the real rows: `query` is `[queryTokens, dim]` and `document` is
    /// `[documentTokens, dim]`. Padded rows are zero vectors and would win the max.
    public static func maxSim(query: MLXArray, document: MLXArray) -> MLXArray {
        let similarity = MLX.matmul(query, document.transposed(1, 0))
        return similarity.max(axis: 1).sum()
    }

    /// MaxSim scores for every query against every document, as `[queries, documents]`.
    ///
    /// - Parameters:
    ///   - queries: `[Q, Lq, dim]` embeddings, padded to `Lq`.
    ///   - documents: `[N, Ld, dim]` embeddings, padded to `Ld`.
    ///   - queryMask: `[Q, Lq]`, 1 for real query tokens.
    ///   - documentMask: `[N, Ld]`, 1 for real document tokens.
    public static func scores(
        queries: MLXArray,
        documents: MLXArray,
        queryMask: MLXArray,
        documentMask: MLXArray
    ) -> MLXArray {
        let numQueries = queries.dim(0)
        let queryLength = queries.dim(1)
        let dim = queries.dim(2)
        let numDocuments = documents.dim(0)
        let documentLength = documents.dim(1)

        let flatQueries = queries.reshaped([numQueries * queryLength, dim])
        let flatDocuments = documents.reshaped([numDocuments * documentLength, dim])
        let similarity = MLX.matmul(flatQueries, flatDocuments.transposed(1, 0))
            .reshaped([numQueries, queryLength, numDocuments, documentLength])

        let documentValid = documentMask.asType(.bool)
            .reshaped([1, 1, numDocuments, documentLength])
        let masked = `where`(documentValid, similarity, MLXArray(-Float.infinity))
        let bestPerQueryToken = masked.max(axis: 3)

        let queryValid = queryMask.asType(.bool).reshaped([numQueries, queryLength, 1])
        let kept = `where`(queryValid, bestPerQueryToken, MLXArray(Float(0)))
        return kept.sum(axis: 1)
    }
}
