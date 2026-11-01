// SharpAI fork overlay: nemo_speechlm creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet. Ported so far: the ASR transformer encoder
// (SharpAINemoSpeechlmAudioEncoder) and the checkpoint key mapping (SharpAINemoSpeechlmSanitizer).
// Still missing, so the creator keeps failing at load time:
//  - the log-mel preprocessor (perception.preprocessor, STFT and filterbank from the checkpoint),
//  - the Sortformer speaker branch and the ParallelExpertEncoder fuse, needed for the audio embedding,
//  - an embedding-input entry point on NemotronHModel (it takes token ids only, and its backbone is private;
//    NemotronH.swift is outside this overlay),
//  - the MTP head (llm.mtp.*).
// This creator decodes and validates the config, then fails with a clear error at load time.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAINemoSpeechlmOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "nemo_speechlm": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAINemoSpeechlmConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAINemoSpeechlmError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
