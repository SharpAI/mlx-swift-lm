// SharpAI fork overlay: glmasr log-mel front end. Not part of upstream mlx-swift-lm.
// Follows transformers' WhisperFeatureExtractor: 16 kHz mono, n_fft 400, hop 160, periodic Hann,
// reflect-padded centre frames, Slaney mel filters, log10 with an 8-decade floor, (x + 4) / 4.
// Waveforms are zero-padded to whole 30 s chunks, as the Whisper extractor pads to 30 s.
// UNVERIFIED: the GLM-ASR preprocessor_config is not on disk; these values are inferred.

import Foundation
import MLX

public struct SharpAIGlmasrFeatureExtractor: Sendable {
    public static let sampleRate = 16_000
    public static let nFft = 400
    public static let hopLength = 160
    public static let chunkSamples = 30 * 16_000

    public let numMelBins: Int

    public init(numMelBins: Int = 128) {
        self.numMelBins = numMelBins
    }

    /// Log-mel features shaped `[1, numMelBins, frames]` for one mono waveform.
    public func inputFeatures(samples: [Float]) throws -> MLXArray {
        let features = try logMel(samples: samples)
        return features.expandedDimensions(axis: 0)
    }

    /// Log-mel features shaped `[numMelBins, frames]`. Frames = padded samples / hop.
    public func logMel(samples: [Float]) throws -> MLXArray {
        guard !samples.isEmpty else {
            throw SharpAIGlmasrError.invalidInput("audio has no samples")
        }
        let chunks = (samples.count + Self.chunkSamples - 1) / Self.chunkSamples
        let targetLength = chunks * Self.chunkSamples
        var waveform = samples
        waveform.append(contentsOf: [Float](repeating: 0, count: targetLength - samples.count))

        let padded = Self.reflectPad(waveform, pad: Self.nFft / 2)
        // Frames before the last one is dropped, as in the Whisper extractor.
        let frameCount = 1 + targetLength / Self.hopLength
        let bins = Self.nFft / 2 + 1

        let window = Self.hannWindow()
        var frameValues = [Float](repeating: 0, count: frameCount * Self.nFft)
        for frame in 0 ..< frameCount {
            let start = frame * Self.hopLength
            for n in 0 ..< Self.nFft {
                frameValues[frame * Self.nFft + n] = padded[start + n] * window[n]
            }
        }
        let frames = MLXArray(frameValues, [frameCount, Self.nFft])

        let (cosBasis, sinBasis) = Self.dftBasis(bins: bins)
        let real = matmul(frames, MLXArray(cosBasis, [Self.nFft, bins]))
        let imag = matmul(frames, MLXArray(sinBasis, [Self.nFft, bins]))
        let power = real * real + imag * imag

        let filters = MLXArray(Self.melFilterBank(numMelBins: numMelBins), [bins, numMelBins])
        let mel = matmul(power, filters).transposed(1, 0)

        var logSpec = log10(maximum(mel, MLXArray(Float(1e-10))))
        logSpec = logSpec[0..., 0 ..< (frameCount - 1)]
        let peak = logSpec.max().item(Float.self)
        logSpec = maximum(logSpec, MLXArray(peak - 8))
        return (logSpec + Float(4)) / Float(4)
    }

    // MARK: - Helpers (CPU tables)

    /// Reflect-pads both ends by `pad` samples, matching numpy `np.pad(mode="reflect")`.
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

    /// Periodic Hann window of length `nFft`.
    static func hannWindow() -> [Float] {
        (0 ..< nFft).map { n in
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(n) / Double(nFft)))
        }
    }

    /// Real DFT basis `[nFft, bins]` for `bins = nFft / 2 + 1`, as cosine and sine tables.
    static func dftBasis(bins: Int) -> ([Float], [Float]) {
        var cosBasis = [Float](repeating: 0, count: nFft * bins)
        var sinBasis = [Float](repeating: 0, count: nFft * bins)
        for n in 0 ..< nFft {
            for k in 0 ..< bins {
                let angle = 2.0 * Double.pi * Double(n * k) / Double(nFft)
                cosBasis[n * bins + k] = Float(cos(angle))
                sinBasis[n * bins + k] = Float(sin(angle))
            }
        }
        return (cosBasis, sinBasis)
    }

    static func hzToMel(_ hz: Double) -> Double {
        let minLogHz = 1000.0
        let logStep = 27.0 / log(6.4)
        if hz >= minLogHz {
            return 15.0 + log(hz / minLogHz) * logStep
        }
        return 3.0 * hz / 200.0
    }

    static func melToHz(_ mel: Double) -> Double {
        let minLogHz = 1000.0
        let logStep = 27.0 / log(6.4)
        if mel >= 15.0 {
            return minLogHz * exp((mel - 15.0) / logStep)
        }
        return 200.0 * mel / 3.0
    }

    /// Slaney-normalised triangular filters, row-major `[bins, numMelBins]`.
    static func melFilterBank(numMelBins: Int) -> [Float] {
        let bins = nFft / 2 + 1
        let melMin = hzToMel(0)
        let melMax = hzToMel(8_000)
        let filterHz = (0 ..< numMelBins + 2).map { index -> Double in
            melToHz(melMin + (melMax - melMin) * Double(index) / Double(numMelBins + 1))
        }

        var filters = [Float](repeating: 0, count: bins * numMelBins)
        for mel in 0 ..< numMelBins {
            let left = filterHz[mel]
            let center = filterHz[mel + 1]
            let right = filterHz[mel + 2]
            let enorm = 2.0 / (right - left)
            for k in 0 ..< bins {
                let hz = Double(k) * Double(sampleRate) / Double(nFft)
                let down = (hz - left) / (center - left)
                let up = (right - hz) / (right - center)
                filters[k * numMelBins + mel] = Float(max(0, min(down, up)) * enorm)
            }
        }
        return filters
    }
}
