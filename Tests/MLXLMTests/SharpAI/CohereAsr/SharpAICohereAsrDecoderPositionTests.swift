// SharpAI fork overlay tests for cohere_asr decoder position handling. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAICohereAsrDecoderPositionTests: XCTestCase {

    func testDecodingPastPositionTableKeepsLogitsFinite() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICohereAsrConfiguration.self,
            from: Data(SharpAICohereAsrSanitizeTests.tinyJSON.utf8))
        let model = try SharpAICohereAsrForConditionalGeneration(config)

        // The tiny configuration has 64 decoder positions. 70 tokens run past the table.
        let tokens = MLXArray.zeros([1, 70], dtype: .int32)
        var state = LMOutput.State()
        state[SharpAICohereAsrStateKeys.encoderStates] = MLXArray.zeros([1, 4, 16])

        let output = model(LMInput.Text(tokens: tokens), cache: nil, state: state)
        eval(output.logits)

        XCTAssertEqual(output.logits.shape, [1, 70, 32])
        XCTAssertTrue(all(isFinite(output.logits)).item(Bool.self))
    }
}
