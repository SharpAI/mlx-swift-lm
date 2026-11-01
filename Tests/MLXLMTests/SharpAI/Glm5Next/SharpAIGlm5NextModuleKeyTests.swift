// SharpAI fork overlay tests for glm5_next module parameter keys. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

final class SharpAIGlm5NextModuleKeyTests: XCTestCase {

    private let textConfigJSON = """
        {
            "model_type": "glm5_next_text",
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 0,
            "v_head_dim": 4,
            "index_topk": 4,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "index_kpool": 2,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 2,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "linear_attn_config": {"num_heads": 2, "head_dim": 4},
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2
        }
        """

    func testEachLayerExposesItsOwnAttentionAndFFNParameters() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGlm5NextTextConfiguration.self, from: Data(textConfigJSON.utf8))
        let model = try SharpAIGlm5NextTextModel(config)
        let keys = Set(model.parameters().flattened().map { $0.0 })

        // Linear-attention layer with a dense MLP.
        XCTAssertTrue(keys.contains("layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("layers.0.self_attn.conv1d.weight"))
        XCTAssertTrue(keys.contains("layers.0.mlp.gate_proj.weight"))
        // Indexed-attention layer with a sparse MoE.
        XCTAssertTrue(keys.contains("layers.1.self_attn.q_a_proj.weight"))
        XCTAssertTrue(keys.contains("layers.1.self_attn.indexer.wk.weight"))
        XCTAssertTrue(keys.contains("layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("layers.1.mlp.gate.weight"))
    }
}
