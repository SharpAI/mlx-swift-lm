import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct Qwen35Tests {

        private func makeTinyConfigData() -> Data {
            let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 100,
                "rope_theta": 10000.0,
                "max_position_embeddings": 512
            }
            """
            return json.data(using: .utf8)!
        }

        @Test("Qwen35 callCapturing returns captured layers")
        func testQwen35CallCapturing() throws {
            Device.setDefault(device: Device.cpu)
            
            let data = makeTinyConfigData()
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
            let model = Qwen35Model(config)

            let input = MLXArray(0..<8).reshaped(1, 8)
            let captureLayerIDs: Set<Int> = [0, 1]
            
            let (hiddenStates, captured) = model.languageModel.model.callCapturing(input, captureLayerIDs: captureLayerIDs)

            #expect(hiddenStates.shape == [1, 8, 64])
            #expect(captured.count == 2)
            #expect(captured[0]?.shape == [1, 8, 64])
            #expect(captured[1]?.shape == [1, 8, 64])
        }
    }
}
