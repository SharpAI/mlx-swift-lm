// SharpAI fork overlay tests for the exaone4_5 creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIExaone45RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "exaone4_5",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096
            },
            "vision_config": {
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIExaone45Overlay.creators["exaone4_5"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() {
        guard let creator = SharpAIExaone45Overlay.creators["exaone4_5"] else {
            XCTFail("exaone4_5 creator missing")
            return
        }

        do {
            _ = try creator(Data(validJSON.utf8))
            XCTFail("creator must not return a model yet")
        } catch let error as SharpAIExaone45Error {
            guard case .forwardNotImplemented(let type) = error else {
                XCTFail("unexpected SharpAIExaone45Error case")
                return
            }
            XCTAssertEqual(type, "exaone4_5")
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testInvalidConfigFailsBeforeForwardError() {
        guard let creator = SharpAIExaone45Overlay.creators["exaone4_5"] else {
            XCTFail("exaone4_5 creator missing")
            return
        }
        let invalid = Data(
            """
            {
                "model_type": "exaone4_5",
                "text_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 512,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 3,
                    "max_position_embeddings": 4096
                }
            }
            """.utf8)

        XCTAssertThrowsError(try creator(invalid)) { error in
            guard let factoryError = error as? ModelFactoryError,
                case .invalidConfiguration = factoryError
            else {
                XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
                return
            }
        }
    }
}
