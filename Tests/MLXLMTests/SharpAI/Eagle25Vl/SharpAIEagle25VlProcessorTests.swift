import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIEagle25VlProcessorTests: XCTestCase {

    func testStringContentIsReturnedAsIs() {
        XCTAssertEqual(SharpAIEagle25VlProcessor.userText(from: "describe this"), "describe this")
    }

    func testTextPartsAreKeptAndNonTextPartsDropped() {
        let parts: [[String: any Sendable]] = [
            ["type": "text", "text": "describe this"],
            ["type": "image"],
            ["type": "text", "text": "in detail"],
        ]
        XCTAssertEqual(
            SharpAIEagle25VlProcessor.userText(from: parts), "describe this\nin detail")
    }

    func testStringDictionaryPartsAreSupported() {
        let parts: [[String: String]] = [["type": "text", "text": "hello"]]
        XCTAssertEqual(SharpAIEagle25VlProcessor.userText(from: parts), "hello")
    }

    func testMissingContentIsEmpty() {
        XCTAssertEqual(SharpAIEagle25VlProcessor.userText(from: nil), "")
    }
}
