// SharpAI fork overlay tests for the kimi_k3 creator. Not part of upstream.
// The creator refuses to build a model until the vision tower and language stack exist,
// so the forward-pass test is a known-issue skip.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

final class SharpAIKimiK3RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIKimiK3Overlay.creators["kimi_k3"])
    }

    func testWrongArchitectureFailsAtValidation() throws {
        guard let creator = SharpAIKimiK3Overlay.creators["kimi_k3"] else {
            XCTFail("kimi_k3 creator missing")
            return
        }
        let json = SharpAIKimiK3Fixtures.tinyJSON.replacingOccurrences(
            of: "KimiK3ForConditionalGeneration", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIKimiK3Error,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidTextConfigConstructsModel() throws {
        guard let creator = SharpAIKimiK3Overlay.creators["kimi_k3"] else {
            XCTFail("kimi_k3 creator missing")
            return
        }

        // The creator now builds the text backbone (KimiLinear KDA + MLA + MoE) from text_config.
        let model = try creator(Data(SharpAIKimiK3TextFixtures.tinyTextJSON.utf8))
        XCTAssertTrue(model is SharpAIKimiK3Model)
    }

    func testTinyForwardPassWithRandomWeights() throws {
        guard let creator = SharpAIKimiK3Overlay.creators["kimi_k3"] else {
            XCTFail("kimi_k3 creator missing")
            return
        }
        let model = try creator(Data(SharpAIKimiK3TextFixtures.tinyTextJSON.utf8))

        // Three tokens through every layer: KDA (MambaCache-free path), MLA, dense layer 0, MoE, attention residuals.
        let tokens = MLXArray([1, 5, 9] as [Int32]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 3, SharpAIKimiK3TextFixtures.vocabularySize])
        XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
    }

    func testVisionTowerIsKnownGap() throws {
        guard let creator = SharpAIKimiK3Overlay.creators["kimi_k3"],
            let model = try creator(Data(SharpAIKimiK3TextFixtures.tinyTextJSON.utf8)) as? Module
        else {
            XCTFail("kimi_k3 model missing")
            return
        }
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTExpectFailure(
            "KNOWN GAP: MoonViT3d vision tower and the patchmergerv2 projector (vision_tower.*, mm_projector.*) are not ported"
        ) {
            XCTAssertTrue(keys.contains { $0.hasPrefix("vision_tower.") })
        }
    }
}
