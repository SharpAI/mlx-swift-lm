import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct Gemma4Tests {

    /// Create a minimal test configuration for Gemma 4 using upstream's JSON-based init
    private func makeTinyConfigData() -> Data {
        let json = """
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "head_dim": 16,
                "global_head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 2,
                "rope_traditional": false,
                "sliding_window": 128,
                "sliding_window_pattern": 1,
                "max_position_embeddings": 512,
                "num_kv_shared_layers": 0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "hidden_size_per_layer_input": 32,
                "vocab_size_per_layer_input": 10,
                "final_logit_softcapping": 30.0,
                "enable_moe_block": false,
                "attention_k_eq_v": false
            },
            "vocab_size": 100
        }
        """
        return json.data(using: .utf8)!
    }

    @Test("Gemma 4 Configuration Decoding")
    func testGemma4ConfigDecoding() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        // vocabSize is internal, verify via model
        let model = Gemma4Model(config)
        #expect(model.vocabularySize == 100)
    }

    @Test("Gemma 4 Model Instantiation")
    func testGemma4ModelInstantiation() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)
        #expect(model.vocabularySize == 100)
    }

    @Test("Gemma 4 Forward Pass - Shape")
    func testGemma4ForwardPass() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)

        let input = MLXArray(0..<8).reshaped(1, 8)
        let output = model(input, cache: nil)

        #expect(output.shape == [1, 8, model.vocabularySize])
    }

    @Test("Forward Pass Determinism")
    func testDeterminism() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)

        let input = MLXArray(0..<8).reshaped(1, 8)
        let output1 = model(input, cache: nil)
        let output2 = model(input, cache: nil)
        #expect(allClose(output1, output2).item(Bool.self))
    }

    @Test("No NaN/Inf in Output")
    func testNoNaNInf() throws {
        let data = makeTinyConfigData()
        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
        let model = Gemma4Model(config)

        let input = MLXArray(Int32(0)..<Int32(5)).reshaped(1, 5)
        let output = model(input, cache: nil)
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    /// Create a minimal test configuration for Gemma 4 Text MoE
    private func makeTinyTextMoEConfigData() -> Data {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 100,
            "num_key_value_heads": 2,
            "rope_traditional": false,
            "sliding_window": 128,
            "sliding_window_pattern": 1,
            "max_position_embeddings": 512,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": false,
            "tie_word_embeddings": true,
            "hidden_size_per_layer_input": 32,
            "vocab_size_per_layer_input": 10,
            "final_logit_softcapping": 30.0,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 128,
            "attention_k_eq_v": false
        }
        """
        return json.data(using: .utf8)!
    }

    @Test("Gemma 4 Text MoE Instantiation & Forward Pass")
    func testGemma4TextMoEInstantiationAndForward() throws {
        let data = makeTinyTextMoEConfigData()
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
        let model = Gemma4TextModel(config)
        #expect(model.vocabularySize == 100)
        
        // This validates that the conditional MoE logic and SwitchGLU layer initialize properly
        // without crashing, proving we correctly load gemma4_text active MoEs.
        let input = MLXArray(0..<8).reshaped(1, 8)
        let output = model(input, cache: nil)

        // Ensure dimensionality is correct
        #expect(output.shape == [1, 8, model.vocabularySize])
        
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }
}
}
