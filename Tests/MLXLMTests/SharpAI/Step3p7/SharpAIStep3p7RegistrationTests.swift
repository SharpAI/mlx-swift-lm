// SharpAI fork overlay tests for the step3p7 creator. Not part of upstream.
// The creator still refuses to build a model: the PE vision tower is not implemented. The
// forward-pass test runs the step3p5 text backbone (SharpAIStep3p7TextModel) with random weights.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIStep3p7RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIStep3p7Overlay.creators["step3p7"])
    }

    func testWrongArchitectureFailsAtValidation() throws {
        guard let creator = SharpAIStep3p7Overlay.creators["step3p7"] else {
            XCTFail("step3p7 creator missing")
            return
        }
        let json = SharpAIStep3p7Fixtures.fixtureJSON.replacingOccurrences(
            of: "Step3p7ForConditionalGeneration", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIStep3p7Error,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    // Pins the current stub. Update this when the model construction step is implemented.
    func testValidConfigStopsAtConstructionAsUnsupported() throws {
        guard let creator = SharpAIStep3p7Overlay.creators["step3p7"] else {
            XCTFail("step3p7 creator missing")
            return
        }

        XCTAssertThrowsError(
            try creator(Data(SharpAIStep3p7Fixtures.fixtureJSON.utf8))
        ) { error in
            guard let typed = error as? SharpAIStep3p7Error,
                case .unsupported = typed
            else {
                XCTFail("expected unsupported, got \(error)")
                return
            }
        }
    }

    // Tiny step3p5 text backbone with random weights. Covers dense and MoE layers, full and
    // sliding attention with different head counts, partial rotary, head-wise gate, and a
    // clamped shared-expert SwiGLU. Checks logits shape, finiteness, and that one cached decode
    // step matches the same position in a full forward pass.
    func testTinyForwardPassWithRandomWeights() throws {
        let args = SharpAIStep3p7TextArguments(
            hiddenSize: 32,
            hiddenLayers: 4,
            vocabularySize: 64,
            attentionHeads: 4,
            attentionGroups: 2,
            headDim: 8,
            slidingAttentionHeads: 8,
            intermediateSize: 48,
            layerTypes: [
                "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            ],
            slidingWindow: 512,
            partialRotaryFactors: [0.5, 1.0, 1.0, 1.0],
            maxPositionEmbeddings: 4096,
            moeLayers: [1, 2, 3],
            moeNumExperts: 4,
            moeTopK: 2,
            moeIntermediateSize: 16,
            shareExpertDim: 16,
            moeRouterScalingFactor: 3.0,
            normExpertWeight: true,
            swigluLimitsShared: [0, 0, 0, 16]
        )
        let model = SharpAIStep3p7TextModel(args)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7, 8], [1, 8])

        let logits = model(tokens)
        XCTAssertEqual(logits.shape, [1, 8, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })

        let cache: [KVCache] = (0 ..< args.hiddenLayers).map { _ in KVCacheSimple() as KVCache }
        _ = model(tokens[0..., ..<7], cache: cache)
        let step = model(tokens[0..., 7...], cache: cache)
        XCTAssertEqual(step.shape, [1, 1, 64])
        XCTAssertTrue(allClose(step[0, 0], logits[0, 7], rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }
}
