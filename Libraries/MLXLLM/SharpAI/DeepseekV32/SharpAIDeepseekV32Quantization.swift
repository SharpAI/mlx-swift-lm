// SharpAI fork overlay: deepseek_v32 quantization guard. Not part of upstream mlx-swift-lm.
//
// NVIDIA ModelOpt checkpoints are rejected at load time. Their scale layout (NVFP4 and
// per-tensor FP8 scales) is not mapped yet, and no ModelOpt checkpoint was available to verify it.
// UNVERIFIED: that ModelOpt exports put `quant_method: "modelopt"` in `quantization_config`.

import Foundation
import MLXLMCommon

public enum SharpAIDeepseekV32Error: Error, LocalizedError {
    case modelOptNotSupported(String)

    public var errorDescription: String? {
        switch self {
        case .modelOptNotSupported(let algo):
            return
                "deepseek_v32 ModelOpt checkpoints (quant_algo \(algo)) are not supported yet."
        }
    }
}

struct SharpAIDeepseekV32ModelOptProbe: Decodable {
    struct QuantizationConfig: Decodable {
        let quantMethod: String?
        let quantAlgo: String?

        enum CodingKeys: String, CodingKey {
            case quantMethod = "quant_method"
            case quantAlgo = "quant_algo"
        }
    }

    let quantizationConfig: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case quantizationConfig = "quantization_config"
    }

    /// Throws for a ModelOpt config. Other configs pass through unchanged.
    static func rejectIfModelOpt(_ data: Data) throws {
        let probe = try JSONDecoder.json5().decode(SharpAIDeepseekV32ModelOptProbe.self, from: data)
        guard probe.quantizationConfig?.quantMethod?.lowercased() == "modelopt" else { return }
        throw SharpAIDeepseekV32Error.modelOptNotSupported(
            probe.quantizationConfig?.quantAlgo ?? "unknown")
    }
}
