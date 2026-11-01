// SharpAI fork overlay tests for the llama_nemotron_vl_rerank creator. Not part of upstream.

import Foundation
import MLXVLM
import XCTest

final class SharpAILlamaNemotronVlRerankRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "llama_nemotron_vl_rerank",
            "architectures": ["LlamaNemotronVLForSequenceClassification"],
            "vocab_size": 1024,
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "rms_norm_eps": 1e-5,
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "patch_size": 16,
                "image_size": 32
            },
            "img_context_token_id": 60
        }
        """

    func testCreatorKeyIsRegistered() {
        XCTAssertNotNil(SharpAILlamaNemotronVlRerankOverlay.creators["llama_nemotron_vl_rerank"])
    }

    /// The creator now builds the model. The earlier not-implemented stub is gone.
    func testValidConfigBuildsModel() throws {
        let creator = SharpAILlamaNemotronVlRerankOverlay.creators["llama_nemotron_vl_rerank"]!

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAILlamaNemotronVlRerankModel)
    }

    func testInvalidConfigFailsValidationBeforeConstruction() {
        let badJSON = validJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let creator = SharpAILlamaNemotronVlRerankOverlay.creators["llama_nemotron_vl_rerank"]!

        XCTAssertThrowsError(try creator(Data(badJSON.utf8)))
    }
}
