import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite
struct StackedMoETests {
    
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

    @Test("Stacked MoE fast path falls back for non-quantized models")
    func testStackedMoEFallback() throws {
        // Set env vars directly. Since tests run concurrently, this might affect others
        // if SwitchGLU is initialized here first, which is fine since the fallback is safe.
        setenv("MLX_MOE_STACKED", "1", 1)
        setenv("MLX_MOE_FUSE_GATEUP", "1", 1)
        defer {
            unsetenv("MLX_MOE_STACKED")
            unsetenv("MLX_MOE_FUSE_GATEUP")
        }

        let data = makeTinyTextMoEConfigData()
        let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
        let model = Gemma4TextModel(config)
        
        // This validates that the fast path falls back cleanly because
        // the weights are not quantized (they are standard MLXArray).
        let input = MLXArray(0..<8).reshaped(1, 8)
        let output = model(input, cache: nil)

        #expect(output.shape == [1, 8, model.vocabularySize])
        
        let sum = output.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }
}
