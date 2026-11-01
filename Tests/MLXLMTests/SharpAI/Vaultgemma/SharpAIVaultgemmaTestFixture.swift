// SharpAI fork overlay test fixtures for vaultgemma. Not part of upstream.
//
// Inline JSON only. Shapes are tiny. Values come from transformers
// configuration_vaultgemma.py, not from a real checkpoint.

import Foundation

enum SharpAIVaultgemmaTestFixture {

    /// Two layers: layer 0 slides with window 4, layer 1 is full attention.
    static func tinyJSON(tied: Bool = true) -> String {
        """
        {
          "architectures": ["VaultGemmaForCausalLM"],
          "model_type": "vaultgemma",
          "vocab_size": 100,
          "hidden_size": 32,
          "intermediate_size": 64,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "hidden_activation": "gelu_pytorch_tanh",
          "max_position_embeddings": 64,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "tie_word_embeddings": \(tied),
          "attention_bias": false,
          "query_pre_attn_scalar": 8,
          "sliding_window": 4,
          "layer_types": ["sliding_attention", "full_attention"],
          "final_logit_softcapping": 30.0,
          "attn_logit_softcapping": 50.0
        }
        """
    }
}
