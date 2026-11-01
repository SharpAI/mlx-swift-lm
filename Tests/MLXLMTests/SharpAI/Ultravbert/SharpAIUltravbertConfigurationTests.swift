// SharpAI fork overlay tests for the UltraVBERT configuration. Not part of upstream.
//
// Offline checks only: inline JSON, no weights, no network.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIUltravbertConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIUltravbertConfiguration {
        try JSONDecoder.json5().decode(SharpAIUltravbertConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesIdentityAndEmbeddingDimension() throws {
        let config = try decode(
            """
            {
              "architectures": ["ColUltraVBertForRetrieval"],
              "model_type": "ultravbert",
              "embedding_dim": 128
            }
            """)

        XCTAssertEqual(config.modelType, "ultravbert")
        XCTAssertEqual(config.architectures, ["ColUltraVBertForRetrieval"])
        XCTAssertEqual(config.embeddingDimensions, 128)
    }

    func testMissingEmbeddingDimensionStaysNil() throws {
        let config = try decode(#"{ "model_type": "ultravbert" }"#)

        XCTAssertNil(config.embeddingDimensions)
        XCTAssertEqual(config.architectures, [])
    }

    func testMissingModelTypeDefaultsToUltravbert() throws {
        let config = try decode(#"{ "architectures": ["ColUltraVBertForRetrieval"] }"#)

        XCTAssertEqual(config.modelType, "ultravbert")
    }
}
