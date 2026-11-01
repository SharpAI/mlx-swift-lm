//
//  SharpAIKimiK3MtpConfiguration.swift
//  MLXLLM
//
//  Config decoding for `kimi_k3_mtp`. Not part of upstream.
//  Verified against: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/kimi_k3/nvidia/mtp.py
//  (KimiK3MultiTokenPredictor.__init__ reads hf_text_config.num_hidden_layers and
//  hf_text_config.num_nextn_predict_layers; mtp_start_layer_idx = num_hidden_layers,
//  num_mtp_layers = num_nextn_predict_layers; layers are keyed "model.layers.{idx}").
//  Only the text_config fields the predictor reads are typed here.
//

import Foundation

public struct SharpAIKimiK3MtpConfiguration: Codable, Sendable {

    /// The `text_config` object of the base KimiK3 `config.json`.
    public struct TextConfig: Codable, Sendable {
        public let numHiddenLayers: Int
        public let numNextnPredictLayers: Int
        public let vocabSize: Int
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case numHiddenLayers = "num_hidden_layers"
            case numNextnPredictLayers = "num_nextn_predict_layers"
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
        }
    }

    public let textConfig: TextConfig

    /// vLLM: `mtp_start_layer_idx = config.num_hidden_layers`.
    public var mtpStartLayerIdx: Int { textConfig.numHiddenLayers }

    /// vLLM: `num_mtp_layers = config.num_nextn_predict_layers`.
    public var numMtpLayers: Int { textConfig.numNextnPredictLayers }

    /// Layer indices of the MTP layers, `[mtpStartLayerIdx, mtpStartLayerIdx + numMtpLayers)`.
    public var mtpLayerIndices: Range<Int> {
        mtpStartLayerIdx ..< (mtpStartLayerIdx + numMtpLayers)
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }
}
