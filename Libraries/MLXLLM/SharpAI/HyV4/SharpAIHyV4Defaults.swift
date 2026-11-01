//
//  SharpAIHyV4Defaults.swift
//  MLXLLM
//
//  Resolved hy_v4 hyper-parameters. Each default is the value the transformers
//  HYV4Config (modular_hy_v4.py) uses when the checkpoint omits the field.
//  Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/hy_v4/modular_hy_v4.py
//

import Foundation

extension SharpAIHyV4Configuration {

    var resolvedQLoraRank: Int { qLoraRank ?? 1536 }
    var resolvedKvLoraRank: Int { kvLoraRank ?? 512 }
    var resolvedNopeHeadDim: Int { qkNopeHeadDim ?? 192 }
    var resolvedRopeHeadDim: Int { qkRopeHeadDim ?? 64 }
    var resolvedVHeadDim: Int { vHeadDim ?? 256 }

    var resolvedIndexHeadDim: Int { indexHeadDim ?? 128 }
    var resolvedIndexHeads: Int { indexNHeads ?? 16 }
    var resolvedIndexTopK: Int { indexTopk ?? 2048 }

    var resolvedHcMult: Int { hcMult ?? 4 }
    var resolvedHcMagnitude: Float { hcMagnitude ?? 2.0 }
    var resolvedHcEps: Float { hcEps ?? 1e-6 }

    var resolvedSwigluLimit: Float { swigluLimit ?? 10.0 }
    var resolvedSharedExperts: Int { numSharedExperts ?? 1 }
    var resolvedRoutedScalingFactor: Float { routedScalingFactor ?? 2.827 }
    var resolvedNormTopKProb: Bool { normTopkProb ?? true }

    var resolvedRopeTheta: Float { ropeParameters?.ropeTheta ?? 10_000_000 }

    /// `dense` for layer 0 and `sparse` after it, unless the checkpoint says otherwise.
    var resolvedMlpLayerTypes: [String] {
        mlpLayerTypes
            ?? (["dense"] + Array(repeating: "sparse", count: max(hiddenLayers - 1, 0)))
    }

    /// `full` on layer 0 and every fourth layer after it, `shared` elsewhere.
    var resolvedIndexerTypes: [String] {
        indexerTypes
            ?? (0 ..< hiddenLayers).map { layer in
                layer == 0 || (layer - 1) % 4 == 0 ? "full" : "shared"
            }
    }
}
