//
//  SharpAIStep1Sanitize.swift
//  MLXVLM
//
//  Checkpoint key cleanup for `step1`. Keys keep their Hugging Face names.
//  UNVERIFIED: no checkpoint index was available, so only the rotary buffer rule is
//  written. Audio tower key remapping is not written.
//

import Foundation
import MLX

public enum SharpAIStep1Sanitize {

    /// Drops precomputed rotary tables. Other keys pass through unchanged.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
