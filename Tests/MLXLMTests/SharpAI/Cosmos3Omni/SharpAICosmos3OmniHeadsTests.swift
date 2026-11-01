// SharpAI fork overlay tests for the cosmos3_omni generation heads. Not part of upstream.
//
// Air-gapped: tiny synthetic configs and deterministic weights only.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXVLM

final class SharpAICosmos3OmniHeadsTests: XCTestCase {

    private func ramp(_ shape: [Int], scale: Float = 0.05, modulo: Int = 11) -> MLXArray {
        let count = shape.reduce(1, *)
        let values = (0 ..< count).map { Float(($0 * 7) % modulo - modulo / 2) * scale }
        return MLXArray(values, shape)
    }

    private func tinyConfig() -> SharpAICosmos3OmniHeadsConfiguration {
        SharpAICosmos3OmniHeadsConfiguration(
            hiddenSize: 8, latentChannel: 3, latentPatchSize: 2, soundDim: 4, actionDim: 5,
            numEmbodimentDomains: 3, timestepScale: 0.001, timeProjChannels: 16)
    }

    func testParameterNamesMatchContractExactly() {
        let heads = SharpAICosmos3OmniHeads(tinyConfig())
        let names = Set(heads.parameters().flattened().map { $0.0 })
        let expected: Set<String> = [
            "proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias",
            "time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight", "time_embedder.linear_2.bias",
            "audio_modality_embed", "audio_proj_in.weight", "audio_proj_in.bias",
            "audio_proj_out.weight", "audio_proj_out.bias",
            "action_modality_embed",
            "action_proj_in.fc.weight", "action_proj_in.bias.weight",
            "action_proj_out.fc.weight", "action_proj_out.bias.weight",
        ]
        XCTAssertEqual(expected.count, 18)
        XCTAssertEqual(names, expected)
    }

    func testParameterShapesMatchCheckpointLayout() {
        let c = tinyConfig()
        let heads = SharpAICosmos3OmniHeads(c)
        let shapes = Dictionary(
            uniqueKeysWithValues: heads.parameters().flattened().map { ($0.0, $0.1.shape) })
        XCTAssertEqual(c.patchLatentDim, 12)
        XCTAssertEqual(shapes["proj_in.weight"], [8, 12])
        XCTAssertEqual(shapes["proj_out.weight"], [12, 8])
        XCTAssertEqual(shapes["time_embedder.linear_1.weight"], [8, 16])
        XCTAssertEqual(shapes["time_embedder.linear_2.weight"], [8, 8])
        XCTAssertEqual(shapes["audio_proj_in.weight"], [8, 4])
        XCTAssertEqual(shapes["audio_proj_out.weight"], [4, 8])
        XCTAssertEqual(shapes["audio_modality_embed"], [8])
        XCTAssertEqual(shapes["action_modality_embed"], [8])
        XCTAssertEqual(shapes["action_proj_in.fc.weight"], [3, 40])
        XCTAssertEqual(shapes["action_proj_in.bias.weight"], [3, 8])
        XCTAssertEqual(shapes["action_proj_out.fc.weight"], [3, 40])
        XCTAssertEqual(shapes["action_proj_out.bias.weight"], [3, 5])
    }

    func testNanoDefaultsGivePatchLatentDim192() {
        XCTAssertEqual(SharpAICosmos3OmniHeadsConfiguration().patchLatentDim, 192)
    }

    func testDomainAwareLinearMatchesNaivePerDomainLoop() throws {
        let inSize = 3
        let outSize = 5
        let layer = SharpAICosmos3OmniDomainAwareLinear(
            inputSize: inSize, outputSize: outSize, numDomains: 3)
        let fc = ramp([3, inSize * outSize], scale: 0.1, modulo: 13)
        let bias = ramp([3, outSize], scale: 0.2, modulo: 5)
        try layer.update(
            parameters: ModuleParameters.unflattened(["fc.weight": fc, "bias.weight": bias]),
            verify: .all)

        let x = ramp([4, inSize], scale: 0.3, modulo: 7)
        let ids: [Int32] = [2, 0, 2, 1]
        let y = layer(x, domainIds: MLXArray(ids))
        XCTAssertEqual(y.shape, [4, outSize])

        let fcV = fc.asArray(Float.self)
        let biasV = bias.asArray(Float.self)
        let xV = x.asArray(Float.self)
        var expected = [Float](repeating: 0, count: 4 * outSize)
        for n in 0 ..< 4 {
            let d = Int(ids[n])
            for o in 0 ..< outSize {
                var acc = biasV[d * outSize + o]
                for i in 0 ..< inSize {
                    // fc[domain] is viewed as [in, out].
                    acc += xV[n * inSize + i] * fcV[d * inSize * outSize + i * outSize + o]
                }
                expected[n * outSize + o] = acc
            }
        }
        let got = y.asArray(Float.self)
        for (a, b) in zip(got, expected) { XCTAssertEqual(a, b, accuracy: 1e-4) }

        // Rank 3: [N, L, in] with the same per-row domain.
        let x3 = ramp([4, 2, inSize], scale: 0.3, modulo: 7)
        let y3 = layer(x3, domainIds: MLXArray(ids))
        XCTAssertEqual(y3.shape, [4, 2, outSize])
        let x3V = x3.asArray(Float.self)
        let got3 = y3.asArray(Float.self)
        for n in 0 ..< 4 {
            let d = Int(ids[n])
            for l in 0 ..< 2 {
                for o in 0 ..< outSize {
                    var acc = biasV[d * outSize + o]
                    for i in 0 ..< inSize {
                        acc +=
                            x3V[(n * 2 + l) * inSize + i]
                            * fcV[d * inSize * outSize + i * outSize + o]
                    }
                    XCTAssertEqual(got3[(n * 2 + l) * outSize + o], acc, accuracy: 1e-4)
                }
            }
        }
    }

    func testPatchifyMatchesNaiveOrderingWithPadding() {
        // C=3, T=2, H=5 (padded to 6), W=4, p=2.
        let c = 3, t = 2, h = 5, w = 4, p = 2
        let latent = ramp([c, t, h, w], scale: 0.01, modulo: 97)
        let patches = SharpAICosmos3OmniHeads.patchify(latent, patchSize: p)
        let hp = 3, wp = 2
        XCTAssertEqual(patches.shape, [t * hp * wp, p * p * c])

        let l = latent.asArray(Float.self)
        let got = patches.asArray(Float.self)
        for ti in 0 ..< t {
            for hi in 0 ..< hp {
                for wi in 0 ..< wp {
                    for pi in 0 ..< p {
                        for qi in 0 ..< p {
                            for ci in 0 ..< c {
                                let row = (ti * hp + hi) * wp + wi
                                let col = (pi * p + qi) * c + ci
                                let hh = hi * p + pi
                                let ww = wi * p + qi
                                let value: Float =
                                    hh < h
                                    ? l[((ci * t + ti) * h + hh) * w + ww] : 0
                                XCTAssertEqual(got[row * (p * p * c) + col], value, accuracy: 1e-6)
                            }
                        }
                    }
                }
            }
        }
    }

    func testPatchifyUnpatchifyRoundTrip() {
        let c = 3, t = 3, h = 5, w = 4, p = 2
        let latent = ramp([c, t, h, w], scale: 0.01, modulo: 97)
        let patches = SharpAICosmos3OmniHeads.patchify(latent, patchSize: p)
        let back = SharpAICosmos3OmniHeads.unpatchify(
            patches, patchSize: p, channels: c, height: h, width: w, totalFrames: t,
            noisyFrameIndexes: [0, 1, 2])
        XCTAssertEqual(back.shape, [c, t, h, w])
        XCTAssertEqual(abs(back - latent).max().item(Float.self), 0, accuracy: 1e-6)
    }

    func testUnpatchifyPlacesNoisyFramesAndZeroesTheRest() {
        let c = 3, t = 3, h = 4, w = 4, p = 2
        let latent = ramp([c, t, h, w], scale: 0.01, modulo: 97)
        // Only frames 1 and 2 are predicted: take their patches.
        let all = SharpAICosmos3OmniHeads.patchify(latent, patchSize: p)
        let perFrame = (h / p) * (w / p)
        let noisy = all[perFrame...]
        let back = SharpAICosmos3OmniHeads.unpatchify(
            noisy, patchSize: p, channels: c, height: h, width: w, totalFrames: t,
            noisyFrameIndexes: [1, 2])
        XCTAssertEqual(abs(back[0..., 0]).max().item(Float.self), 0, accuracy: 1e-9)
        XCTAssertEqual(
            abs(back[0..., 1...] - latent[0..., 1...]).max().item(Float.self), 0, accuracy: 1e-6)
    }

    func testSinusoidalProjectionMatchesFormulaAndFlip() {
        let channels = 16
        let ts: [Float] = [0, 0.5, 1.0]
        let proj = SharpAICosmos3OmniHeads.sinusoidalTimestepProjection(
            MLXArray(ts), channels: channels)
        XCTAssertEqual(proj.shape, [3, channels])
        let got = proj.asArray(Float.self)
        let half = channels / 2
        for (row, t) in ts.enumerated() {
            for i in 0 ..< half {
                let f = exp(-log(10000.0) * Double(i) / Double(half))
                let angle = Double(t) * f
                // flip_sin_to_cos: cos first, then sin.
                XCTAssertEqual(got[row * channels + i], Float(cos(angle)), accuracy: 1e-5)
                XCTAssertEqual(got[row * channels + half + i], Float(sin(angle)), accuracy: 1e-5)
            }
        }
    }

    func testTimestepEmbeddingShapeFinitenessAndScale() {
        let heads = SharpAICosmos3OmniHeads(tinyConfig())
        let emb = heads.timestepEmbedding(MLXArray([Float(0), 500, 1000]))
        XCTAssertEqual(emb.shape, [3, 8])
        XCTAssertTrue(all(isFinite(emb)).item(Bool.self))
        // A raw timestep of 1000 equals a pre-scaled 1.0 passed through the same path.
        let direct = heads.timeEmbedder(
            SharpAICosmos3OmniHeads.sinusoidalTimestepProjection(MLXArray([Float(1.0)]), channels: 16)
        )
        XCTAssertEqual(abs(emb[2] - direct[0]).max().item(Float.self), 0, accuracy: 1e-5)
    }

    func testNoisyTokenIndexes() {
        XCTAssertEqual(
            SharpAICosmos3OmniHeads.noisyTokenIndexes(
                noisyFrameIndexes: [1, 3], spatialTokenCount: 2, start: 10),
            [12, 13, 16, 17])
    }

    func testTimestepEmbeddingAddedToNoisyTokensOnly() {
        let heads = SharpAICosmos3OmniHeads(tinyConfig())
        let tokens = ramp([6, 8], scale: 0.1, modulo: 9)
        let idx = SharpAICosmos3OmniHeads.noisyTokenIndexes(
            noisyFrameIndexes: [1, 2], spatialTokenCount: 2)  // tokens 2...5
        let steps = MLXArray([Float(800), 800, 300, 300])
        let out = heads.addTimestepEmbeddings(
            to: tokens, rawTimesteps: steps, noisyTokenIndexes: idx)
        XCTAssertEqual(out.shape, [6, 8])
        // Conditioning frame (tokens 0 and 1) is untouched.
        XCTAssertEqual(abs(out[0 ..< 2] - tokens[0 ..< 2]).max().item(Float.self), 0, accuracy: 1e-7)
        let emb = heads.timestepEmbedding(steps)
        let delta = out[2...] - tokens[2...]
        XCTAssertEqual(abs(delta - emb).max().item(Float.self), 0, accuracy: 1e-5)
    }

    func testEmptyNoisySetLeavesTokensUnchanged() {
        let heads = SharpAICosmos3OmniHeads(tinyConfig())
        let tokens = ramp([4, 8])
        let out = heads.addTimestepEmbeddings(
            to: tokens, rawTimesteps: MLXArray([Float]()), noisyTokenIndexes: [])
        XCTAssertEqual(abs(out - tokens).max().item(Float.self), 0, accuracy: 1e-9)
    }

    func testForwardShapesAndFiniteness() {
        let c = tinyConfig()
        let heads = SharpAICosmos3OmniHeads(c)

        let vision = heads.embedVision(ramp([6, c.patchLatentDim]))
        XCTAssertEqual(vision.shape, [6, 8])
        let velocity = heads.predictVision(vision)
        XCTAssertEqual(velocity.shape, [6, c.patchLatentDim])
        XCTAssertTrue(all(isFinite(velocity)).item(Bool.self))

        let sound = heads.embedSound(ramp([5, c.soundDim]))
        XCTAssertEqual(sound.shape, [5, 8])
        XCTAssertEqual(heads.predictSound(sound).shape, [5, c.soundDim])

        let ids = MLXArray([Int32(0), 2, 1])
        let action = heads.embedAction(ramp([3, c.actionDim]), domainIds: ids)
        XCTAssertEqual(action.shape, [3, 8])
        let predicted = heads.predictAction(action, domainIds: ids)
        XCTAssertEqual(predicted.shape, [3, c.actionDim])
        XCTAssertTrue(all(isFinite(predicted)).item(Bool.self))
    }

    // MARK: - Optional sound and action heads

    private func disabledConfig(sound: Bool, action: Bool) -> SharpAICosmos3OmniHeadsConfiguration {
        var c = tinyConfig()
        c.soundGen = sound
        c.actionGen = action
        return c
    }

    func testDisabledHeadsHaveNoParameters() {
        let none = Set(
            SharpAICosmos3OmniHeads(disabledConfig(sound: false, action: false))
                .parameters().flattened().map { $0.0 })
        XCTAssertEqual(none.count, 8)
        XCTAssertFalse(none.contains { $0.hasPrefix("audio_") || $0.hasPrefix("action_") })

        let soundOnly = Set(
            SharpAICosmos3OmniHeads(disabledConfig(sound: true, action: false))
                .parameters().flattened().map { $0.0 })
        XCTAssertEqual(soundOnly.count, 13)
        XCTAssertFalse(soundOnly.contains { $0.hasPrefix("action_") })

        let actionOnly = Set(
            SharpAICosmos3OmniHeads(disabledConfig(sound: false, action: true))
                .parameters().flattened().map { $0.0 })
        XCTAssertEqual(actionOnly.count, 13)
        XCTAssertFalse(actionOnly.contains { $0.hasPrefix("audio_") })
    }

    func testSanitizerMapsDisabledHeadKeysAsUnrecognized() throws {
        let w = MLXArray.zeros([2])
        // Enabled: audio key is mapped.
        let ok = try SharpAICosmos3OmniSanitizer.sanitize(["audio_proj_in.weight": w])
        XCTAssertNotNil(ok["gen_heads.audio_proj_in.weight"])
        // Sound disabled: the key has no parameter to load into, so it is rejected, not dropped.
        XCTAssertThrowsError(
            try SharpAICosmos3OmniSanitizer.sanitize(
                ["audio_proj_in.weight": w], soundGen: false))
        XCTAssertThrowsError(
            try SharpAICosmos3OmniSanitizer.sanitize(
                ["action_proj_in.fc.weight": w], actionGen: false))
        // Video heads stay mapped when the optional heads are off.
        let video = try SharpAICosmos3OmniSanitizer.sanitize(
            ["proj_in.weight": w], soundGen: false, actionGen: false)
        XCTAssertNotNil(video["gen_heads.proj_in.weight"])
    }

    // MARK: - Head configuration decoding

    func testDecodeIgnoresTowerChecksButReadsHeadSizes() throws {
        // mrope_section [16, 8, 8] does not sum to head_dim / 2 (= 4), so the full generation
        // configuration rejects this, but the head sizes are still well defined.
        let json = """
            {"model_type": "cosmos3_omni",
             "text_config": {"hidden_size": 32, "head_dim": 8, "num_attention_heads": 4,
               "rope_parameters": {"mrope_section": [16, 8, 8], "rope_theta": 10000}}}
            """
        XCTAssertThrowsError(
            try SharpAICosmos3OmniGenerationConfiguration.decode(
                rootConfig: Data(json.utf8), transformerConfig: nil))
        let heads = try SharpAICosmos3OmniHeadsConfiguration.decode(
            rootConfig: Data(json.utf8), fallbackHiddenSize: 32)
        XCTAssertEqual(heads.hiddenSize, 32)
        XCTAssertEqual(heads.patchLatentDim, 192)
        XCTAssertFalse(heads.soundGen)
        XCTAssertFalse(heads.actionGen)
    }

    func testDecodeReadsFlagsAndSizesAndRejectsMalformed() throws {
        let transformer = """
            {"hidden_size": 8, "latent_channel": 3, "latent_patch_size": 2, "sound_gen": true,
             "sound_dim": 4, "action_gen": true, "max_action_dim": 5, "num_embodiment_domains": 3}
            """
        let heads = try SharpAICosmos3OmniHeadsConfiguration.decode(
            rootConfig: nil, transformerConfig: Data(transformer.utf8), fallbackHiddenSize: 1)
        XCTAssertEqual(heads.hiddenSize, 8)
        XCTAssertEqual(heads.soundDim, 4)
        XCTAssertEqual(heads.actionDim, 5)
        XCTAssertEqual(heads.numEmbodimentDomains, 3)
        XCTAssertTrue(heads.soundGen && heads.actionGen)

        // An enabled head without its size is a configuration error, not a silent default.
        XCTAssertThrowsError(
            try SharpAICosmos3OmniHeadsConfiguration.decode(
                rootConfig: nil, transformerConfig: Data(#"{"hidden_size": 8, "sound_gen": true}"#.utf8),
                fallbackHiddenSize: 1))
        XCTAssertThrowsError(
            try SharpAICosmos3OmniHeadsConfiguration.decode(
                rootConfig: nil, transformerConfig: Data(#"{"hidden_size": "x"}"#.utf8),
                fallbackHiddenSize: 1))
    }
}
