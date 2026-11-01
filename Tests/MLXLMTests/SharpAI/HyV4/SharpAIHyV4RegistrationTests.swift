import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHyV4RegistrationTests: XCTestCase {

    func testOverlayRegistersHyV4Key() throws {
        XCTAssertNotNil(SharpAIHyV4Overlay.creators["hy_v4"])
    }

    /// The creator now builds the model. It used to throw `unsupported` while no model class existed.
    func testCreatorBuildsModelAfterValidConfig() throws {
        let creator = try XCTUnwrap(SharpAIHyV4Overlay.creators["hy_v4"])
        // The tiny config keeps the allocation small. The full-size fixture (hidden 6144, 256 experts) is too large for a unit test.
        let data = Data(SharpAIHyV4TinyFixture.json.utf8)

        let model = try creator(data)
        XCTAssertTrue(model is SharpAIHyV4Model)
    }

    func testCreatorRejectsInvalidConfigBeforeConstruction() throws {
        let creator = try XCTUnwrap(SharpAIHyV4Overlay.creators["hy_v4"])
        let data = Data(SharpAIHyV4ConfigurationTests.makeJSON(numExpertsPerTok: 999).utf8)

        XCTAssertThrowsError(try creator(data)) { error in
            guard let typed = error as? SharpAIHyV4Error,
                case .invalidConfiguration = typed
            else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    /// Tiny random-weight forward pass. The logits have the vocabulary shape, and every value is finite.
    func testTinyForwardPassWithRandomWeights() throws {
        let model = try SharpAIHyV4TinyFixture.makeModel()
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 5, 100])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
