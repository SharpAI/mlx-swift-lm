// Copyright © 2025 Apple Inc.

import MLX
import MLXLMCommon
import MLXVLM
import XCTest

/// The equivalences every M-RoPE continuation model must satisfy, independent of which model
/// it is.
///
/// These models place tokens from a continuation anchor — the cache offset plus the rope delta
/// the cached images accumulated — so the properties worth pinning are all *equivalences*:
/// prefilling a prompt in pieces, or in windows, must land where prefilling it whole does. Each
/// assertion runs the same prompt two ways and compares final-position logits.
///
/// Configure it with the model's special token ids; everything else (prompt layout, seeds,
/// tolerances) is fixed here so all models are held to the same bar. Models with extra structure
/// — deepstack features, batch guards, prompt-cache round trips — keep those tests in their own
/// suite.
struct ContinuationAssertions {
    /// The id repeated to form an image's placeholder run in the text.
    let imageTokenId: Int32

    /// The id that precedes an image run, for models whose templates emit one. GLM-OCR has none.
    let visionStartTokenId: Int32?

    init(imageTokenId: Int32, visionStartTokenId: Int32? = nil) {
        self.imageTokenId = imageTokenId
        self.visionStartTokenId = visionStartTokenId
    }

    // MARK: - Fixtures

    /// Deterministic pseudo-random plain-text tokens, away from the special ids (500...504).
    func textTokens(_ count: Int, seed: Int32 = 0) -> MLXArray {
        let values = (0 ..< count).map { Int32(($0 * 13 + 7 + Int(seed)) % 480) }
        return MLXArray(values).expandedDimensions(axis: 0)
    }

    /// One image: grid THW (1, 4, 4), merge 2 → 4 merged tokens in the text.
    func image() -> LMInput.ProcessedImage {
        LMInput.ProcessedImage(
            pixels: MLXRandom.normal([16, 3 * 2 * 16 * 16]), frames: [THW(1, 4, 4)])
    }

    /// The token run standing in for one image, including the vision-start marker if the model
    /// uses one. Four merged image tokens, matching ``image()``.
    func imageRun() -> MLXArray {
        var ids = [Int32](repeating: imageTokenId, count: 4)
        if let visionStartTokenId {
            ids.insert(visionStartTokenId, at: 0)
        }
        return MLXArray(ids).expandedDimensions(axis: 0)
    }

    func lastLogits(_ result: PrepareResult) throws -> (MLXArray, LMOutput.State?) {
        guard case .logits(let out) = result else {
            throw XCTSkip("expected .logits from prepare")
        }
        return (out.logits[0..., -1, 0...], out.state)
    }

    func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    /// Prefill `tokens` on a fresh cache and return the final-position logits and resume state.
    private func prefill<M: LanguageModel>(
        _ model: M, _ tokens: MLXArray, image: LMInput.ProcessedImage? = nil,
        cache: [any KVCache], state: LMOutput.State? = nil, stepSize: Int? = nil
    ) throws -> (MLXArray, LMOutput.State?) {
        let prefill = stepSize.map { PrefillParameters(stepSize: $0) } ?? PrefillParameters()
        return try lastLogits(
            model.prepare(
                LMInput(text: .init(tokens: tokens), image: image),
                cache: cache, state: state, prefill: prefill))
    }

    // MARK: - Assertions

    /// A warm continuation (prefix in the cache, remainder prefilled on top — the ChatSession
    /// cross-turn flow) must produce the same next-token logits as one cold prefill of the
    /// concatenation.
    ///
    /// The decode path (token by token, state threaded) is the offset-correct control: it is
    /// known to position correctly, so the difference between it and the cold prefill is the
    /// numerical noise inherent to splitting the forward, and bounds what the warm path may drift.
    func assertWarmTextContinuation<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        // Task-local rather than MLXRandom.seed: parallel tests must not share
        // (or perturb) the global random stream.
        try withRandomState(MLXRandom.RandomState(seed: 7)) {
            let t1 = textTokens(40)
            let t2 = textTokens(8, seed: 3)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try prefill(model, concatenated([t1, t2], axis: 1), cache: cacheF)

            let cacheD = try model.newCache(parameters: nil)
            let (_, s0) = try prefill(model, t1, cache: cacheD)
            var state = s0
            var logitsD = MLXArray(0)
            for j in 0 ..< t2.dim(1) {
                let out = model(
                    LMInput.Text(tokens: t2[0..., j ..< (j + 1)]), cache: cacheD, state: state)
                state = out.state
                logitsD = out.logits[0..., -1, 0...]
            }
            let noiseFloor = maxAbsDiff(logitsD, logitsF)

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, cache: cacheW)
            let (logitsW, _) = try prefill(model, t2, cache: cacheW, state: s1)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsW, logitsF), max(noiseFloor * 10, 1e-3),
                "warm continuation diverged from full prefill (noise floor \(noiseFloor))",
                file: file, line: line)
        }
    }

    /// With an image in turn 1, the rope delta that image accumulated must be carried into
    /// turn 2's prefill: two turns with threaded state ≡ one full prefill.
    func assertWarmImageContinuation<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 5)) {
            let image = image()
            let t1 = concatenated([textTokens(10), imageRun(), textTokens(8, seed: 5)], axis: 1)
            let t2 = textTokens(8, seed: 9)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try prefill(
                model, concatenated([t1, t2], axis: 1), image: image, cache: cacheF)

            // The decode path (token by token, state threaded) is the offset-correct
            // control, same as in assertWarmTextContinuation: the vision tower makes
            // this floor larger than the text-only case, so it must be measured rather
            // than assumed away.
            let cacheD = try model.newCache(parameters: nil)
            let (_, s0) = try prefill(model, t1, image: image, cache: cacheD)
            var state = s0
            var logitsD = MLXArray(0)
            for j in 0 ..< t2.dim(1) {
                let out = model(
                    LMInput.Text(tokens: t2[0..., j ..< (j + 1)]), cache: cacheD, state: state)
                state = out.state
                logitsD = out.logits[0..., -1, 0...]
            }
            let noiseFloor = maxAbsDiff(logitsD, logitsF)

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, image: image, cache: cacheW)
            let (logitsW, _) = try prefill(model, t2, cache: cacheW, state: s1)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsW, logitsF), max(noiseFloor * 10, 1e-3),
                "state-threaded warm continuation diverged from full prefill (noise floor \(noiseFloor))",
                file: file, line: line)
        }
    }

    /// An append-only media turn: turn 2 adds its own image to an unchanged turn 1.
    /// `split` carves the suffix out of the full prepared input, and prefilling that
    /// suffix on turn 1's cache must land where a cold prefill of the whole prompt
    /// does -- but only if the vision encoder computes each image independently.
    ///
    /// `expectsIsolation` says which model this is. `Qwen25VL` masks each frame to
    /// itself, so the split is exact; `Qwen2VL` attends across the concatenated
    /// buffer, so dropping the cached image changes the kept image's features and
    /// the logits must differ. Asserting the divergence is what makes the
    /// conformance a measured claim rather than a stated one.
    func assertAppendOnlyMediaSplit<M: LanguageModel>(
        _ model: M, split: (LMInput, Int) -> LMInput?, expectsIsolation: Bool,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 17)) {
            // A bigger grid than `image()`'s (8x8 instead of 4x4 patches, merging to
            // 16 image tokens instead of 4) gives unmasked cross-image attention many
            // more patch pairs to mix over.
            func bigImage() -> LMInput.ProcessedImage {
                LMInput.ProcessedImage(
                    pixels: MLXRandom.normal([64, 3 * 2 * 16 * 16]), frames: [THW(1, 8, 8)])
            }
            func bigImageRun() -> MLXArray {
                var ids = [Int32](repeating: imageTokenId, count: 16)
                if let visionStartTokenId {
                    ids.insert(visionStartTokenId, at: 0)
                }
                return MLXArray(ids).expandedDimensions(axis: 0)
            }
            let imageA = bigImage()
            let imageB = bigImage()
            let t1 = concatenated(
                [textTokens(10), bigImageRun(), textTokens(8, seed: 5)], axis: 1)
            let t2 = concatenated(
                [textTokens(6, seed: 2), bigImageRun(), textTokens(4, seed: 9)], axis: 1)

            // The prepared input a VL processor hands over for the whole transcript:
            // both images' patch rows concatenated, one grid each.
            let bothImages = LMInput.ProcessedImage(
                pixels: concatenated([imageA.pixels, imageB.pixels], axis: 0),
                frames: (imageA.frames ?? []) + (imageB.frames ?? []))
            let fullInput = LMInput(
                text: .init(tokens: concatenated([t1, t2], axis: 1)), image: bothImages)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try lastLogits(
                model.prepare(
                    fullInput, cache: cacheF, state: nil, prefill: PrefillParameters()))

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, image: imageA, cache: cacheW)
            // Captured before the suffix prepare call below mutates `cacheW` in
            // place, extending it past t1's own positions.
            let wState = cacheW.map(\.state)

            let suffix = try XCTUnwrap(
                split(fullInput, t1.dim(1)), "the split was declined", file: file, line: line)
            XCTAssertEqual(suffix.text.tokens.dim(-1), t2.dim(1), file: file, line: line)
            XCTAssertEqual(
                suffix.image?.frames?.count, 1, "the suffix must carry only the new image",
                file: file, line: line)

            let (logitsW, _) = try lastLogits(
                model.prepare(
                    suffix, cache: cacheW, state: s1, prefill: PrefillParameters()))

            let diff = maxAbsDiff(logitsW, logitsF)

            // Whether the vision tower mixed imageB's patches into t1's image is a
            // claim about the attention computation itself, not about the final
            // logits: by the time a cross-image signal has propagated through the
            // rest of a tiny, untrained model's decoder stack, it is the same order
            // of magnitude as ordinary floating point noise from splitting one
            // forward into two calls -- confirmed empirically, the final-logits gap
            // between the diverging and isolating cases could not be reliably pulled
            // clear of a same-image split-noise floor, however the fixtures or model
            // size were tuned (larger images, wider or deeper vision towers, and
            // more contrastive image pairs all left the two the same order of
            // magnitude).
            //
            // Compare the decoder's cached keys/values for t1's own positions
            // instead: `cacheW` computed them with only imageA in the vision
            // tower's batch, `cacheF` computed the same positions with both images
            // concatenated in that batch. This is the exact quantity the isolation
            // guarantee is about, measured before it has had a chance to be
            // renormalized away by later layers -- unmasked cross-attention
            // (Qwen2VL) makes these differ by orders of magnitude more than kernel
            // noise, while masking each frame to itself (Qwen2.5-VL) makes them
            // agree almost exactly, since imageB's patches never reach imageA's
            // features.
            let t1Length = t1.dim(1)
            let cacheDivergence =
                zip(cacheF.map(\.state), wState)
                .map { fLayerState, wLayerState in
                    zip(fLayerState, wLayerState)
                        .map { fState, wState in
                            maxAbsDiff(fState[.ellipsis, ..<t1Length, 0...], wState)
                        }
                        .max() ?? 0
                }
                .max() ?? 0

            if expectsIsolation {
                XCTAssertLessThanOrEqual(
                    cacheDivergence, 1e-3,
                    "masked-per-frame vision attention should not have mixed the two images' features (cache divergence \(cacheDivergence))",
                    file: file, line: line)
                XCTAssertLessThanOrEqual(
                    diff, 1e-2,
                    "split-suffix prefill diverged from full prefill",
                    file: file, line: line)
            } else {
                XCTAssertGreaterThan(
                    cacheDivergence, 1e-2,
                    "unmasked cross-image vision attention should have changed t1's cached features when computed alongside imageB (cache divergence \(cacheDivergence))",
                    file: file, line: line)
            }
        }
    }

    /// Three turns, with the image in the middle one. The continuation must place that image at
    /// the anchor *and* hand back a resume state that positions the following turn — the two
    /// halves of the anchor invariant, which a single-turn test cannot separate.
    func assertImageMidContinuationResumeState<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 3)) {
            let image = image()
            let t1 = textTokens(12)
            let t2 = concatenated(
                [textTokens(4, seed: 2), imageRun(), textTokens(6, seed: 4)], axis: 1)
            let t3 = textTokens(8, seed: 6)

            let cacheF = try model.newCache(parameters: nil)
            let (logitsF, _) = try prefill(
                model, concatenated([t1, t2, t3], axis: 1), image: image, cache: cacheF)

            // Decode-step control (see assertWarmTextContinuation): the same warm
            // prefix, but t3 threaded token by token instead of split as a whole
            // prefill. This measures the floor that any post-image split carries,
            // independent of whether the resume state is positioned correctly.
            let cacheD = try model.newCache(parameters: nil)
            let (_, d1) = try prefill(model, t1, cache: cacheD)
            let (_, d2) = try prefill(model, t2, image: image, cache: cacheD, state: d1)
            var state = d2
            var logitsD = MLXArray(0)
            for j in 0 ..< t3.dim(1) {
                let out = model(
                    LMInput.Text(tokens: t3[0..., j ..< (j + 1)]), cache: cacheD, state: state)
                state = out.state
                logitsD = out.logits[0..., -1, 0...]
            }
            let noiseFloor = maxAbsDiff(logitsD, logitsF)

            let cacheW = try model.newCache(parameters: nil)
            let (_, s1) = try prefill(model, t1, cache: cacheW)
            let (_, s2) = try prefill(model, t2, image: image, cache: cacheW, state: s1)
            let (logitsW, _) = try prefill(model, t3, cache: cacheW, state: s2)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsW, logitsF), max(noiseFloor * 10, 1e-3),
                "post-image resume state positioned the following turn wrong (noise floor \(noiseFloor))",
                file: file, line: line)
        }
    }

    /// Windowed (chunked) prefill must agree with the single-shot forward on plain text.
    func assertWindowedTextPrefill<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 11)) {
            let prompt = textTokens(40)

            let cacheS = try model.newCache(parameters: nil)
            let (logitsS, _) = try prefill(model, prompt, cache: cacheS)

            // Decode-step control (see assertWarmTextContinuation): the same
            // prompt, but its last chunk threaded token by token instead of
            // prefilled as a chunked window.
            let head = prompt[0..., 0 ..< (prompt.dim(1) - 8)]
            let tail = prompt[0..., (prompt.dim(1) - 8)...]
            let cacheD = try model.newCache(parameters: nil)
            let (_, d1) = try prefill(model, head, cache: cacheD)
            var state = d1
            var logitsD = MLXArray(0)
            for j in 0 ..< tail.dim(1) {
                let out = model(
                    LMInput.Text(tokens: tail[0..., j ..< (j + 1)]), cache: cacheD, state: state)
                state = out.state
                logitsD = out.logits[0..., -1, 0...]
            }
            let noiseFloor = maxAbsDiff(logitsD, logitsS)

            let cacheC = try model.newCache(parameters: nil)
            let (logitsC, _) = try prefill(model, prompt, cache: cacheC, stepSize: 8)

            XCTAssertLessThanOrEqual(
                maxAbsDiff(logitsC, logitsS), max(noiseFloor * 10, 1e-3),
                "windowed prefill diverged from single-shot (noise floor \(noiseFloor))",
                file: file, line: line)
        }
    }

    /// Windowed prefill on an image-bearing prompt must agree with the single-shot forward — the
    /// hard case for chunked slicing, since embeddings, positions, and any per-layer visual
    /// features have to be sliced in lockstep.
    ///
    /// Swept across several step sizes rather than fixed at one. Balanced chunking derives its
    /// actual boundaries from the total length, so no single step size reliably lands a boundary
    /// *inside* the image run — and that split is the case worth covering. The small sizes here
    /// guarantee some run does.
    func assertWindowedImagePrefill<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 13)) {
            let image = image()
            let head = textTokens(10)
            let mid = imageRun()
            let tail = textTokens(12, seed: 7)
            let prompt = concatenated([head, mid, tail], axis: 1)

            let cacheS = try model.newCache(parameters: nil)
            let (logitsS, _) = try prefill(model, prompt, image: image, cache: cacheS)

            // Decode-step control (see assertWarmTextContinuation): the same
            // head+image prefix, but the tail threaded token by token instead of
            // prefilled as a chunked window. Computed once, since it does not depend
            // on stepSize — it measures the floor any split of this prompt carries.
            let cacheD = try model.newCache(parameters: nil)
            let (_, d1) = try prefill(
                model, concatenated([head, mid], axis: 1), image: image, cache: cacheD)
            var state = d1
            var logitsD = MLXArray(0)
            for j in 0 ..< tail.dim(1) {
                let out = model(
                    LMInput.Text(tokens: tail[0..., j ..< (j + 1)]), cache: cacheD, state: state)
                state = out.state
                logitsD = out.logits[0..., -1, 0...]
            }
            let noiseFloor = maxAbsDiff(logitsD, logitsS)

            for stepSize in [3, 5, 8] {
                let cacheC = try model.newCache(parameters: nil)
                let (logitsC, _) = try prefill(
                    model, prompt, image: image, cache: cacheC, stepSize: stepSize)

                XCTAssertLessThanOrEqual(
                    maxAbsDiff(logitsC, logitsS), max(noiseFloor * 10, 1e-3),
                    "windowed image prefill diverged from single-shot at stepSize \(stepSize) (noise floor \(noiseFloor))",
                    file: file, line: line)
            }
        }
    }
}
