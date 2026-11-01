// SharpAI fork test helper for the Dory overlay. Not part of upstream.

import Foundation

enum SharpAIDoryTestFixture {

    /// Reference logits for `tinyJSON` and the token ids in the forward-pass test. None exist in
    /// the repo. The reference (vLLM dory.py at b5007e42) needs torch and vLLM, which are not
    /// installed here, and the tiny config has no checkpoint. Stays nil. See the KNOWN GAP in
    /// SharpAIDoryRegistrationTests.
    static let referenceLogits: [Float]? = nil

    /// Small shape for running the real forward pass with random weights. Layers: sliding input,
    /// MLP and full-attention recurrent, MLP output. Not a checkpoint shape.
    static let tinyJSON = """
        {
            "architectures": ["DoryForCausalLM"],
            "model_type": "dory",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "intermediate_size": 48,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false,
            "ngpt_alpha_init": 0.05,
            "attention_output_gate_logit_scale_init": null,
            "mlp_bias": false,
            "attention_bias": false,
            "n_input_layers": 1,
            "n_recurrent_layers": 2,
            "n_output_layers": 1,
            "n_recurrent_loops": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "initializer_range": 0.0177,
            "dtype": "bfloat16",
            "sliding_window": 8,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0
                },
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 1.0
                }
            },
            "layer_types": ["sliding_attention", "mlp", "full_attention", "mlp"]
        }
        """

    /// Even indices hold attention sublayers, odd indices hold "mlp".
    /// Full attention indices copied from the layer_types in the fixture JSON below.
    static let fullAttentionIndices: Set<Int> = [6, 14, 22, 30, 38, 46, 54, 62, 70]

    static var layerTypes: [String] {
        (0 ..< 72).map { index in
            if index % 2 == 1 { return "mlp" }
            return fullAttentionIndices.contains(index) ? "full_attention" : "sliding_attention"
        }
    }

    /// Values copied from scripts/conformance/offline/fixtures/dory/nvidia__NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16.json.
    /// UNVERIFIED: the copy is not checked against the upstream checkpoint, so these tests only
    /// show that the code agrees with this copy.
    static var json: String {
        let layers = layerTypes.map { "\"\($0)\"" }.joined(separator: ", ")
        return """
            {
                "architectures": ["DoryForCausalLM"],
                "model_type": "dory",
                "vocab_size": 131072,
                "hidden_size": 2560,
                "num_hidden_layers": 72,
                "intermediate_size": 7168,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "max_position_embeddings": 262144,
                "tie_word_embeddings": false,
                "ngpt_alpha_init": 0.05,
                "attention_output_gate_logit_scale_init": null,
                "mlp_bias": false,
                "attention_bias": false,
                "n_input_layers": 16,
                "n_recurrent_layers": 40,
                "n_output_layers": 16,
                "n_recurrent_loops": 2,
                "head_dim": 256,
                "hidden_act": "silu",
                "initializer_range": 0.0177,
                "dtype": "bfloat16",
                "sliding_window": 1024,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000,
                        "partial_rotary_factor": 1.0
                    },
                    "full_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000000,
                        "partial_rotary_factor": 1.0
                    }
                },
                "layer_types": [\(layers)]
            }
            """
    }
}
