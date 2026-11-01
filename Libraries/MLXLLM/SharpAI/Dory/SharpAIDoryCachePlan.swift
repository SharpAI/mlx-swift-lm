// SharpAI fork overlay: Dory KV-cache layout. Not part of upstream mlx-swift-lm.
//
// Pure data. It lists which sublayers own a KV cache and in which recurrence loop.
// UNVERIFIED: the recurrent stage is run once per loop with a separate cache per loop, and the
// input and output stages are not looped. This follows n_recurrent_loops and the gap note.
// Confirm against the reference implementation before any attention code uses it.

import Foundation

public enum SharpAIDoryStage: String, Sendable, Equatable {
    case input
    case recurrent
    case output
}

public enum SharpAIDoryAttentionKind: String, Sendable, Equatable {
    case sliding = "sliding_attention"
    case full = "full_attention"
}

public struct SharpAIDoryCacheSlot: Sendable, Equatable {
    /// Index into `layer_types`.
    public let sublayer: Int
    public let stage: SharpAIDoryStage
    /// Recurrence loop, 0-based. Always 0 outside the recurrent stage.
    public let loop: Int
    public let kind: SharpAIDoryAttentionKind
}

public enum SharpAIDoryCachePlan {

    /// One slot per attention sublayer, and one per loop for recurrent attention sublayers.
    /// "mlp" sublayers own no cache and are skipped.
    public static func slots(
        for configuration: SharpAIDoryConfiguration
    ) throws -> [SharpAIDoryCacheSlot] {
        try configuration.validateModelConfiguration()

        let inputEnd = configuration.nInputLayers
        let recurrentEnd = inputEnd + configuration.nRecurrentLayers
        var slots: [SharpAIDoryCacheSlot] = []

        for (index, type) in configuration.layerTypes.enumerated() {
            guard let kind = SharpAIDoryAttentionKind(rawValue: type) else { continue }

            if index < inputEnd {
                slots.append(
                    SharpAIDoryCacheSlot(sublayer: index, stage: .input, loop: 0, kind: kind))
            } else if index < recurrentEnd {
                for loop in 0 ..< configuration.nRecurrentLoops {
                    slots.append(
                        SharpAIDoryCacheSlot(
                            sublayer: index, stage: .recurrent, loop: loop, kind: kind))
                }
            } else {
                slots.append(
                    SharpAIDoryCacheSlot(sublayer: index, stage: .output, loop: 0, kind: kind))
            }
        }
        return slots
    }
}
