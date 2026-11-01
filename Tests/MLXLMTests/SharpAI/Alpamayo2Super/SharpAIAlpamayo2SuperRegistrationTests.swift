// SharpAI fork overlay tests for the alpamayo2_super creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIAlpamayo2SuperRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "alpamayo2_super",
            "vocab_size": 512,
            "vlm_config": { "trajectory_token_start": 400, "trajectory_token_count": 100 }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIAlpamayo2SuperOverlay.creators["alpamayo2_super"])
    }

    func testValidWrappedConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIAlpamayo2SuperOverlay.creators["alpamayo2_super"] else {
            XCTFail("alpamayo2_super creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIAlpamayo2SuperError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "alpamayo2_super")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIAlpamayo2SuperOverlay.creators["alpamayo2_super"] else {
            XCTFail("alpamayo2_super creator missing")
            return
        }
        let json = """
            { "vocab_size": 512, "trajectory_token_start": 510, "trajectory_token_count": 10 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            if case SharpAIAlpamayo2SuperError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    func testTinyForwardPassSkipped() throws {
        guard let creator = SharpAIAlpamayo2SuperOverlay.creators["alpamayo2_super"] else {
            XCTFail("alpamayo2_super creator missing")
            return
        }
        // KNOWN GAP: the model class is not written. Real source exists upstream:
        // NVlabs/alpamayo2 src/alpamayo2_super/models/alpamayo2_super.py (forward,
        // sample_trajectories_from_data) and models/expert.py. Porting the Qwen3-VL VLM,
        // the flow-matching expert, and fuse_traj_tokens is not done in this change.
        // Flips to a hard failure once the creator returns a model.
        XCTExpectFailure(
            "KNOWN GAP: alpamayo2_super forward pass not ported. Reference: NVlabs/alpamayo2 src/alpamayo2_super/models/alpamayo2_super.py, expert.py, action_in_proj.py, config.py. Missing in Swift: the 64-layer hidden-1536 flow-matching expert with Euler diffusion sampler, the action space (unicycle lstsq fit, 64 waypoints), the discrete future tokenizer (3000 bins), the delta history tokenizer, fuse_traj_tokens, and shared-prefill CoT generation. Reusable VLM backbone: MLXVLM/Models/Qwen3VL.swift. Not landed in this batch: too large to write without a build.") {
            let model = try? creator(Data(validJSON.utf8))
            XCTAssertNotNil(model)
        }
    }
}
