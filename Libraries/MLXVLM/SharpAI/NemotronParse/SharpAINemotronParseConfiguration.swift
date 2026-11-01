// SharpAI fork overlay: nemotron_parse configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow the checkpoint's config.json (nvidia/NVIDIA-Nemotron-Parse-v1.1, read from
// hf_nemotron_parse_config.py and config.json; Verified against:
// https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/resolve/main/config.json and
// https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/resolve/main/hf_nemotron_parse_modeling.py).
//
// Encoder: the RADIO block only names `nvidia/C-RADIOv2-H`. Its ViT-H dimensions come from the
// `vit_huge_patch16_224` model name in `encoder.args`, so they are constants below. The keys
// `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and `mlp_dim` are not in the real
// checkpoint. They are read only when present, so a reduced-size model can be built for tests.
// Summary-token layout (`num_cls_tokens`, `summary_idxs`), registers, and the CPE max size are
// read from `encoder.args`, matching `radio_model.py` `create_model_from_args`.

import Foundation
import MLXLMCommon

public enum SharpAINemotronParseError: LocalizedError {
    /// Retained for source compatibility. The creator no longer throws it: the model is built.
    case forwardNotImplemented(String)
    case missingImage
    case unsupportedResolution(height: Int, width: Int, patchSize: Int)
    case cpeGridTooSmall(rows: Int, cols: Int, maxRows: Int, maxCols: Int)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return "Model type \(type) forward pass is not implemented."
        case .missingImage:
            return "nemotron_parse requires pixel values (the RADIO encoder has no text-only path)."
        case .unsupportedResolution(let height, let width, let patchSize):
            return
                "nemotron_parse input \(height)x\(width) must be a multiple of the patch size \(patchSize)."
        case .cpeGridTooSmall(let rows, let cols, let maxRows, let maxCols):
            return
                "nemotron_parse input grid \(rows)x\(cols) exceeds the position table \(maxRows)x\(maxCols)."
        }
    }
}

/// Shape of the RADIO (C-RADIOv2-H) encoder as used by the nemotron_parse port.
public struct SharpAINemotronParseEncoderShape: Sendable, Equatable {
    public var patchSize: Int
    public var hiddenSize: Int
    public var layers: Int
    public var attentionHeads: Int
    public var mlpDim: Int
    /// `args.cpe_max_size` (2048): the position table covers `cpeMaxSize / patchSize` per side.
    public var cpeMaxSize: Int
    /// Number of class tokens: one per unique teacher when `cls_token_per_teacher`, else 1.
    public var numClsTokens: Int
    /// `args.register_multiple`, used by `ClsToken` when `num_registers` is unset.
    public var registerMultiple: Int
    /// Indices into the class tokens whose summaries are concatenated (`use_summary`).
    public var summaryIndices: [Int]

    /// C-RADIOv2-H defaults: ViT-H/16 with 4 teacher class tokens, 4 registers, and summaries from
    /// teachers 0, 1, 2 (CLIP, SigLIP, DINOv2); SAM (index 3) has `use_summary: false`.
    public static let radioViTH = SharpAINemotronParseEncoderShape(
        patchSize: 16, hiddenSize: 1280, layers: 32, attentionHeads: 16, mlpDim: 5120,
        cpeMaxSize: 2048, numClsTokens: 4, registerMultiple: 8, summaryIndices: [0, 1, 2])

    public var numRegisters: Int {
        registerMultiple > 0 ? registerMultiple - (numClsTokens % registerMultiple) : 0
    }

    /// Tokens before the patch tokens: class tokens followed by registers.
    public var numSkipTokens: Int { numClsTokens + numRegisters }

    /// Width of the concatenated summaries fed to `sum_proj` (3 x 1280 = 3840 for this checkpoint).
    public var summaryDim: Int { summaryIndices.count * hiddenSize }

    /// Position-table side length in patches.
    public var cpeGridSide: Int { cpeMaxSize / patchSize }
}

public struct SharpAINemotronParseConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let architectures: [String]
    public let encoderModelType: String?
    public let encoder: SharpAINemotronParseEncoderShape
    public let decoderHiddenSize: Int
    public let decoderLayers: Int
    public let decoderAttentionHeads: Int
    /// `decoder.decoder_ffn_dim`. Kept for reference. The layers use `encoderFFNDim`.
    public let decoderFFNDim: Int?
    /// `decoder.encoder_ffn_dim`. HF `MBartDecoderLayer.fc1` uses this field, not `decoder_ffn_dim`.
    public let ffnDim: Int
    public let vocabularySize: Int
    /// Top-level `decoder_start_token_id` (2). Callers must start the decoder input with it.
    public let decoderStartTokenId: Int?
    /// Top-level `pad_token_id` (1).
    public let padTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case encoder
        case decoder
        case decoderStartTokenId = "decoder_start_token_id"
        case padTokenId = "pad_token_id"
    }

    private struct EncoderFields: Decodable, Sendable {
        let modelType: String?
        let patchSize: Int?
        let hiddenSize: Int?
        let layers: Int?
        let attentionHeads: Int?
        let mlpDim: Int?
        let args: Args?

        struct Args: Decodable, Sendable {
            let cpeMaxSize: Int?
            let registerMultiple: Int?
            let clsTokenPerTeacher: Bool?
            let teachers: [Teacher]?

            enum CodingKeys: String, CodingKey {
                case cpeMaxSize = "cpe_max_size"
                case registerMultiple = "register_multiple"
                case clsTokenPerTeacher = "cls_token_per_teacher"
                case teachers
            }
        }

        struct Teacher: Decodable, Sendable {
            let name: String?
            let useSummary: Bool?

            enum CodingKeys: String, CodingKey {
                case name
                case useSummary = "use_summary"
            }
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case patchSize = "patch_size"
            case hiddenSize = "hidden_size"
            case layers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case mlpDim = "mlp_dim"
            case args
        }
    }

    private struct DecoderFields: Decodable, Sendable {
        let hiddenSize: Int
        let layers: Int
        let attentionHeads: Int
        let ffnDim: Int?
        let encoderFFNDim: Int?
        let vocabularySize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "d_model"
            case layers = "decoder_layers"
            case attentionHeads = "decoder_attention_heads"
            case ffnDim = "decoder_ffn_dim"
            case encoderFFNDim = "encoder_ffn_dim"
            case vocabularySize = "vocab_size"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        decoderStartTokenId = try container.decodeIfPresent(Int.self, forKey: .decoderStartTokenId)
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)

        let encoderFields = try container.decodeIfPresent(EncoderFields.self, forKey: .encoder)
        encoderModelType = encoderFields?.modelType
        encoder = Self.makeEncoderShape(encoderFields)

        let fields = try container.decode(DecoderFields.self, forKey: .decoder)
        decoderHiddenSize = fields.hiddenSize
        decoderLayers = fields.layers
        decoderAttentionHeads = fields.attentionHeads
        decoderFFNDim = fields.ffnDim
        ffnDim = fields.encoderFFNDim ?? fields.ffnDim ?? 4 * fields.hiddenSize
        vocabularySize = fields.vocabularySize
    }

    /// Applies the RADIO defaults to whatever the encoder block provides.
    private static func makeEncoderShape(_ fields: EncoderFields?) -> SharpAINemotronParseEncoderShape {
        var shape = SharpAINemotronParseEncoderShape.radioViTH
        guard let fields else { return shape }
        shape.patchSize = fields.patchSize ?? shape.patchSize
        shape.hiddenSize = fields.hiddenSize ?? shape.hiddenSize
        shape.layers = fields.layers ?? shape.layers
        shape.attentionHeads = fields.attentionHeads ?? shape.attentionHeads
        shape.mlpDim = fields.mlpDim ?? shape.mlpDim
        if let args = fields.args {
            shape.cpeMaxSize = args.cpeMaxSize ?? shape.cpeMaxSize
            shape.registerMultiple = args.registerMultiple ?? shape.registerMultiple
            if let teachers = args.teachers {
                // create_model_from_args: num_cls_tokens = len(set(names)) if cls_token_per_teacher.
                let uniqueNames = Set(teachers.map { $0.name ?? "" })
                shape.numClsTokens =
                    (args.clsTokenPerTeacher ?? true) ? uniqueNames.count : 1
                // RADIOModel: summary_idxs = indices of teachers with use_summary (default true).
                shape.summaryIndices = teachers.enumerated()
                    .filter { $0.element.useSummary ?? true }
                    .map { $0.offset }
            }
        }
        return shape
    }

    public func validateModelConfiguration() throws {
        guard decoderHiddenSize > 0, decoderLayers > 0, decoderAttentionHeads > 0,
            vocabularySize > 0, ffnDim > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "nemotron_parse: decoder hidden, layer, head, FFN and vocab sizes must be positive")
        }
        guard decoderHiddenSize % decoderAttentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "nemotron_parse: decoder d_model must be divisible by decoder_attention_heads")
        }
        let enc = encoder
        guard enc.patchSize > 0, enc.hiddenSize > 0, enc.layers > 0, enc.mlpDim > 0,
            enc.attentionHeads > 0, enc.cpeMaxSize >= enc.patchSize,
            enc.cpeMaxSize % enc.patchSize == 0, enc.numClsTokens > 0,
            !enc.summaryIndices.isEmpty,
            enc.summaryIndices.allSatisfy({ $0 >= 0 && $0 < enc.numClsTokens })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "nemotron_parse: encoder patch, width, depth, position, or summary settings are invalid")
        }
        guard enc.hiddenSize % enc.attentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "nemotron_parse: encoder hidden_size must be divisible by num_attention_heads")
        }
    }
}
