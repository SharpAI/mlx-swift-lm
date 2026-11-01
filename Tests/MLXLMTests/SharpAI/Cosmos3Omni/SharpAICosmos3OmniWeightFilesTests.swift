// SharpAI fork overlay tests for cosmos3_omni subfolder weight selection. Not part of upstream.

import MLXVLM
import XCTest

final class SharpAICosmos3OmniWeightFilesTests: XCTestCase {

    func testSelectsOnlyOneFolderDeepSafetensors() {
        let paths = [
            "model.safetensors",
            "transformer/model-00002-of-00002.safetensors",
            "transformer/model-00001-of-00002.safetensors",
            "transformer/config.json",
            "nested/deep/model.safetensors",
        ]

        XCTAssertEqual(
            SharpAICosmos3OmniWeightFiles.subfolderSafetensors(in: paths),
            [
                "transformer/model-00001-of-00002.safetensors",
                "transformer/model-00002-of-00002.safetensors",
            ])
    }

    func testEmptyInputReturnsEmpty() {
        XCTAssertTrue(SharpAICosmos3OmniWeightFiles.subfolderSafetensors(in: []).isEmpty)
    }
}
