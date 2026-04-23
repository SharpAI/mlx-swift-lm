import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct Qwen3NextTests {

        private func makeTinyConfigData() -> Data {
            let json = """
            {
                "model_type": "qwen3_next",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "decoder_sparse_step": 2,
                "shared_expert_intermediate_size": 64,
                "mlp_only_layers": [],
                "moe_intermediate_size": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "num_key_value_heads": 2,
                "rope_theta": 10000.0,
                "max_position_embeddings": 512,
                "full_attention_interval": 2
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("Qwen3Next callCapturing returns captured layers")
        func testQwen3NextCallCapturing() throws {
            Device.setDefault(device: Device.cpu)
            
            let data = makeTinyConfigData()
            let config = try JSONDecoder().decode(Qwen3NextConfiguration.self, from: data)
            let model = Qwen3NextModel(config)

            let input = MLXArray(0..<8).reshaped(1, 8)
            let captureLayerIDs: Set<Int> = [0, 2]
            
            let (hiddenStates, captured) = model.model.callCapturing(input, captureLayerIDs: captureLayerIDs)

            #expect(hiddenStates.shape == [1, 8, 64])
            #expect(captured.count == 2)
            #expect(captured[0]?.shape == [1, 8, 64])
            #expect(captured[2]?.shape == [1, 8, 64])
        }
    }
}
