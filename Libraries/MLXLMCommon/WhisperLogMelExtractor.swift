// Copyright © 2026 SharpAI.
//
// Shared Whisper-style log-mel front end for audio-capable models.
//
// Status: infrastructure only. No model in this repository consumes it yet, and it does not
// change `AudioProcessor`, `UserInput.Audio` or any model processor. Existing per-model
// extractors (Gemma4, GLM-ASR, ...) are untouched.
//
// Algorithm (mirrors transformers' `WhisperFeatureExtractor`, the front end Qwen3-Omni,
// Qwen2-Audio and Whisper itself use):
//   1. optionally zero-pad the waveform to a whole number of `chunkSeconds` chunks (30 s);
//   2. reflect-pad `nFft / 2` samples at both ends (centred frames, like `torch.stft(center=True)`);
//   3. frame with `hopLength`, multiply by a periodic Hann window of length `nFft`;
//   4. power spectrum of the real DFT (`nFft / 2 + 1` bins), dropping the last frame;
//   5. Slaney-scale, Slaney-normalised triangular mel filters between 0 Hz and `sampleRate / 2`;
//   6. `log10(max(mel, 1e-10))`, clamp to `peak - 8`, then `(x + 4) / 4`.
//
// Defaults: 16 kHz, nFft 400, hop 160, 80 mels (128 for Qwen3-Omni / large-v3), 30 s chunks.
// The DFT is a plain matmul against cos/sin tables, so it needs no Accelerate and runs on
// any platform MLX supports.
//
// Limit: verified here only against analytic properties (shapes, tone-to-bin placement,
// the floor and dynamic range, window and mel-scale identities). It has NOT been compared
// numerically with the Python reference on real audio.

import Foundation
import MLX

/// Error thrown by audio feature extraction.
public struct AudioFeatureError: Error, Equatable, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { "AudioFeatureError: \(message)" }
}

/// Parameters of ``WhisperLogMelExtractor``. The defaults are Whisper's.
public struct WhisperLogMelConfiguration: Sendable, Equatable {
    public var sampleRate: Int
    public var nFft: Int
    public var hopLength: Int
    public var nMels: Int
    /// Chunk length the waveform is zero-padded to when padding is requested. `nil` never pads.
    public var chunkSeconds: Int?

    public init(
        sampleRate: Int = 16_000, nFft: Int = 400, hopLength: Int = 160, nMels: Int = 80,
        chunkSeconds: Int? = 30
    ) {
        self.sampleRate = sampleRate
        self.nFft = nFft
        self.hopLength = hopLength
        self.nMels = nMels
        self.chunkSeconds = chunkSeconds
    }
}

/// Log-mel spectrogram extractor for mono waveforms at `configuration.sampleRate`.
public struct WhisperLogMelExtractor: Sendable {

    public let configuration: WhisperLogMelConfiguration

    public init(_ configuration: WhisperLogMelConfiguration = .init()) {
        self.configuration = configuration
    }

    /// Log-mel features shaped `[nMels, frames]`, where `frames = paddedSamples / hopLength`.
    ///
    /// - Parameters:
    ///   - samples: mono PCM in -1 ... 1 at `configuration.sampleRate`.
    ///   - padToChunk: zero-pad to a whole number of `chunkSeconds` chunks first (Whisper's
    ///     behaviour). Ignored when `chunkSeconds` is `nil`.
    public func logMel(samples: [Float], padToChunk: Bool = true) throws -> MLXArray {
        let c = configuration
        guard c.nFft > 0, c.hopLength > 0, c.nMels > 0, c.sampleRate > 0 else {
            throw AudioFeatureError("nFft, hopLength, nMels and sampleRate must be positive")
        }
        guard !samples.isEmpty else { throw AudioFeatureError("audio has no samples") }

        var waveform = samples
        if padToChunk, let seconds = c.chunkSeconds, seconds > 0 {
            let chunk = seconds * c.sampleRate
            let target = ((samples.count + chunk - 1) / chunk) * chunk
            waveform.append(contentsOf: [Float](repeating: 0, count: target - samples.count))
        }
        let pad = c.nFft / 2
        guard waveform.count > pad else {
            throw AudioFeatureError(
                "audio has \(waveform.count) samples; at least \(pad + 1) are needed for reflect padding"
            )
        }
        // Frames before the last one is dropped, as in the Whisper extractor.
        let frameCount = waveform.count / c.hopLength
        guard frameCount > 0 else {
            throw AudioFeatureError("audio is shorter than one hop (\(c.hopLength) samples)")
        }

        let padded = Self.reflectPad(waveform, pad: pad)
        let window = Self.hannWindow(nFft: c.nFft)
        let bins = c.nFft / 2 + 1

        // Frame i covers padded[i*hop ..< i*hop + nFft]; frame `frameCount` (dropped) is skipped.
        var frameValues = [Float](repeating: 0, count: frameCount * c.nFft)
        for frame in 0 ..< frameCount {
            let start = frame * c.hopLength
            for n in 0 ..< c.nFft {
                frameValues[frame * c.nFft + n] = padded[start + n] * window[n]
            }
        }
        let frames = MLXArray(frameValues, [frameCount, c.nFft])

        let (cosBasis, sinBasis) = Self.dftBasis(nFft: c.nFft)
        let real = matmul(frames, MLXArray(cosBasis, [c.nFft, bins]))
        let imag = matmul(frames, MLXArray(sinBasis, [c.nFft, bins]))
        let power = real * real + imag * imag

        let filters = MLXArray(Self.melFilterBank(c), [bins, c.nMels])
        let mel = matmul(power, filters).transposed(1, 0)

        var logSpec = log10(maximum(mel, MLXArray(Float(1e-10))))
        let peak = logSpec.max().item(Float.self)
        logSpec = maximum(logSpec, MLXArray(peak - 8))
        return (logSpec + Float(4)) / Float(4)
    }

    // MARK: - Tables (CPU)

    /// Reflect-pads both ends by `pad` samples, like numpy `np.pad(mode: "reflect")`.
    /// Requires `pad < values.count`.
    static func reflectPad(_ values: [Float], pad: Int) -> [Float] {
        let count = values.count
        var out = [Float](repeating: 0, count: count + 2 * pad)
        for index in 0 ..< out.count {
            var source = index - pad
            if source < 0 { source = -source }
            if source >= count { source = 2 * (count - 1) - source }
            out[index] = values[source]
        }
        return out
    }

    /// Periodic Hann window (`torch.hann_window(periodic: true)`).
    static func hannWindow(nFft: Int) -> [Float] {
        (0 ..< nFft).map { n in
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(n) / Double(nFft)))
        }
    }

    /// Real DFT cos/sin tables, row-major `[nFft, nFft / 2 + 1]`.
    static func dftBasis(nFft: Int) -> ([Float], [Float]) {
        let bins = nFft / 2 + 1
        var cosBasis = [Float](repeating: 0, count: nFft * bins)
        var sinBasis = [Float](repeating: 0, count: nFft * bins)
        for n in 0 ..< nFft {
            for k in 0 ..< bins {
                // (n * k) mod nFft keeps the argument small and the tables accurate.
                let angle = 2.0 * Double.pi * Double((n * k) % nFft) / Double(nFft)
                cosBasis[n * bins + k] = Float(cos(angle))
                sinBasis[n * bins + k] = Float(sin(angle))
            }
        }
        return (cosBasis, sinBasis)
    }

    /// Slaney mel scale: linear below 1 kHz (3 mel per 200 Hz), logarithmic above.
    static func hzToMel(_ hz: Double) -> Double {
        let minLogHz = 1000.0
        let logStep = 27.0 / log(6.4)
        if hz >= minLogHz { return 15.0 + log(hz / minLogHz) * logStep }
        return 3.0 * hz / 200.0
    }

    static func melToHz(_ mel: Double) -> Double {
        let minLogHz = 1000.0
        let logStep = 27.0 / log(6.4)
        if mel >= 15.0 { return minLogHz * exp((mel - 15.0) / logStep) }
        return 200.0 * mel / 3.0
    }

    /// Slaney-normalised triangular filters between 0 Hz and Nyquist, row-major `[bins, nMels]`.
    static func melFilterBank(_ c: WhisperLogMelConfiguration) -> [Float] {
        let bins = c.nFft / 2 + 1
        let melMin = hzToMel(0)
        let melMax = hzToMel(Double(c.sampleRate) / 2)
        let edgesHz = (0 ..< c.nMels + 2).map { index -> Double in
            melToHz(melMin + (melMax - melMin) * Double(index) / Double(c.nMels + 1))
        }
        var filters = [Float](repeating: 0, count: bins * c.nMels)
        for mel in 0 ..< c.nMels {
            let left = edgesHz[mel]
            let center = edgesHz[mel + 1]
            let right = edgesHz[mel + 2]
            let enorm = 2.0 / (right - left)
            for k in 0 ..< bins {
                let hz = Double(k) * Double(c.sampleRate) / Double(c.nFft)
                let down = (hz - left) / (center - left)
                let up = (right - hz) / (right - center)
                filters[k * c.nMels + mel] = Float(max(0, min(down, up)) * enorm)
            }
        }
        return filters
    }
}
