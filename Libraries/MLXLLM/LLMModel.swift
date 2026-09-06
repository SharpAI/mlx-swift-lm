// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

// MARK: - Prefill Progress Hook
//
// This global closure is set by Server.swift before each generate() call and
// cleared when the first decode token arrives. It mirrors llama-server's
// slot_update progress reporting: called after each 512-token prefill chunk
// with (n_past, n_total).
//
// Thread model: written by the server async task before generation starts
// (happens-before the generation Task reads it), and read only from the
// synchronous MLX evaluation thread inside prepare(). This is safe without
// a lock because writes precede all reads in time.
public nonisolated(unsafe) var activePrefillProgressHook: ((Int, Int) -> Void)? = nil

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// Evaluates the prompt into the cache in chunks of at most
    /// `PrefillParameters.stepSize` (default 512), leaving one token for the
    /// `TokenIterator`'s first forward. With `PrefillParameters.Chunking.balanced`
    /// (the default) the chunks are equal-sized, so no forward is a small
    /// remainder paying full attention cost against the whole prompt.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        let totalTokens = input.text.tokens.size
        var y = input.text
        var processed = 0

        // Prepare the prompt in chunks if larger than the prefill size.
        // After each chunk, call the progress hook so the server can emit
        // llama-server-style slot_update SSE events with real n_past.
        while y.tokens.size > prefillStepSize {
            let input = y[.newAxis, ..<prefillStepSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            y = y[prefillStepSize...]
            processed += prefillStepSize
            activePrefillProgressHook?(processed, totalTokens)
        }

        return .tokens(y[processed...])
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
