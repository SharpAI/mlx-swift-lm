// Gemma4AudioFeatureExtractor.swift
// USM (Universal Speech Model) audio feature extractor for Gemma 4.
//
// Ported from:
//   3rd_party/mlx-vlm/mlx_vlm/models/gemma4/audio_feature_extractor.py
//
// Pipeline:
//   raw waveform → semicausal pad → unfold frames → preemphasis → window → FFT →
//   magnitude → mel filter bank → log → (optional per-bin normalization)

import Accelerate
import Foundation
import MLX

// MARK: - Mel Filter Bank

/// Create an HTK-scale mel filter bank matrix of shape (numFreqBins, numMelFilters).
private func melFilterBank(
    numFrequencyBins: Int,
    numMelFilters: Int,
    minFrequency: Float,
    maxFrequency: Float,
    samplingRate: Int
) -> [[Float]] {
    func hzToMel(_ freq: Float) -> Float {
        2595.0 * log10(1.0 + freq / 700.0)
    }
    func melToHz(_ mel: Float) -> Float {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    let melMin = hzToMel(minFrequency)
    let melMax = hzToMel(maxFrequency)

    // numMelFilters + 2 evenly spaced points in mel space
    var melPoints = [Float](repeating: 0, count: numMelFilters + 2)
    for i in 0..<(numMelFilters + 2) {
        melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(numMelFilters + 1)
    }
    let freqPoints = melPoints.map { melToHz($0) }

    // All FFT bin center frequencies
    var allFreqs = [Float](repeating: 0, count: numFrequencyBins)
    for i in 0..<numFrequencyBins {
        allFreqs[i] = Float(i) * Float(samplingRate) / Float(2 * (numFrequencyBins - 1))
    }

    // Build triangular filters
    var filterBank = [[Float]](repeating: [Float](repeating: 0, count: numMelFilters), count: numFrequencyBins)
    for i in 0..<numMelFilters {
        let lower = freqPoints[i]
        let center = freqPoints[i + 1]
        let upper = freqPoints[i + 2]

        for j in 0..<numFrequencyBins {
            let rising = (allFreqs[j] - lower) / max(center - lower, 1e-10)
            let falling = (upper - allFreqs[j]) / max(upper - center, 1e-10)
            filterBank[j][i] = max(0, min(rising, falling))
        }
    }

    return filterBank
}

// MARK: - Feature Extractor

/// USM-compatible audio feature extractor for Gemma 4.
/// Extracts log-mel spectrograms from raw 16kHz mono PCM waveforms.
public class Gemma4AudioFeatureExtractor {

    public let featureSize: Int  // Number of mel bins (128)
    public let samplingRate: Int  // Expected sample rate (16000)
    public let frameLength: Int  // Samples per frame (320 @ 20ms)
    public let hopLength: Int  // Hop between frames (160 @ 10ms)
    public let fftLength: Int  // FFT size (512)
    public let melFloor: Float  // Floor for log(mel + floor) (1e-3)

    private let window: [Float]  // Periodic Hann window
    private let melFilters: [[Float]]  // (fftLength/2+1, featureSize) mel filter bank
    private let perBinMean: [Float]?
    private let perBinStddev: [Float]?

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16_000,
        frameLengthMs: Float = 20.0,
        hopLengthMs: Float = 10.0,
        minFrequency: Float = 0.0,
        maxFrequency: Float = 8000.0,
        fftOverdrive: Bool = false,
        melFloor: Float = 1e-3,
        perBinMean: [Float]? = nil,
        perBinStddev: [Float]? = nil
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.melFloor = melFloor
        self.perBinMean = perBinMean
        self.perBinStddev = perBinStddev

        self.frameLength = Int(round(Float(samplingRate) * frameLengthMs / 1000.0))
        self.hopLength = Int(round(Float(samplingRate) * hopLengthMs / 1000.0))

        var fft = 1
        while fft < frameLength { fft *= 2 }
        if fftOverdrive { fft *= 2 }
        self.fftLength = fft

        // Periodic Hann window: w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
        // where N = frameLength (periodic, NOT symmetric which would use N-1)
        var win = [Float](repeating: 0, count: frameLength)
        let twoPiOverN = 2.0 * Float.pi / Float(frameLength)
        for n in 0..<frameLength {
            win[n] = 0.5 - 0.5 * cos(twoPiOverN * Float(n))
        }
        self.window = win

        self.melFilters = melFilterBank(
            numFrequencyBins: fft / 2 + 1,
            numMelFilters: featureSize,
            minFrequency: minFrequency,
            maxFrequency: maxFrequency,
            samplingRate: samplingRate
        )
    }

    /// Extract log-mel spectrogram and attention mask from a raw waveform.
    ///
    /// - Parameter waveform: 1-D Float array of audio samples at `samplingRate` Hz.
    /// - Returns: `(features, mask)` where:
    ///   - `features`: MLXArray of shape `[1, numFrames, featureSize]`
    ///   - `mask`: MLXArray of shape `[1, numFrames]` — `true` = valid audio frame
    public func extract(waveform: [Float]) -> (features: MLXArray, mask: MLXArray) {
        let maxLength = 480_000  // 30s max
        var wav = waveform
        if wav.count > maxLength {
            wav = Array(wav.prefix(maxLength))
        }

        // Pad to multiple of 128 samples
        let padMultiple = 128
        let remainder = wav.count % padMultiple
        let originalLength = wav.count
        if remainder != 0 {
            wav.append(contentsOf: [Float](repeating: 0, count: padMultiple - remainder))
        }

        // Build attention mask for waveform (1 = valid, 0 = padding)
        var attentionMask = [Int32](repeating: 1, count: wav.count)
        for i in originalLength..<wav.count {
            attentionMask[i] = 0
        }

        // Semicausal left-padding: prepend frame_length // 2 zeros
        let padLeft = frameLength / 2
        wav = [Float](repeating: 0, count: padLeft) + wav
        attentionMask = [Int32](repeating: 0, count: padLeft) + attentionMask

        // Frame unfold: window of size (frameLength + 1), step = hopLength
        let frameSizeForUnfold = frameLength + 1
        let numFrames = (wav.count - frameSizeForUnfold) / hopLength + 1
        if numFrames <= 0 {
            // Too short — return empty
            let emptyFeatures = MLXArray.zeros([1, 0, featureSize])
            let emptyMask = MLXArray.zeros([1, 0]).asType(Bool.self)
            return (emptyFeatures, emptyMask)
        }

        // Extract frames and apply preemphasis (preemphasis=0 → just drop last sample)
        // frames = frames_to_process[..., :-1]
        var frames = [[Float]](repeating: [Float](repeating: 0, count: frameLength), count: numFrames)
        for i in 0..<numFrames {
            let start = i * hopLength
            for j in 0..<frameLength {
                frames[i][j] = wav[start + j]
            }
        }

        // Apply window
        for i in 0..<numFrames {
            for j in 0..<frameLength {
                frames[i][j] *= window[j]
            }
        }

        // FFT → magnitude → mel → log
        let numFreqBins = fftLength / 2 + 1
        var melSpectrogram = [[Float]](repeating: [Float](repeating: 0, count: featureSize), count: numFrames)

        // Use Accelerate for FFT
        let log2n = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            // Fallback: return zeros
            let emptyFeatures = MLXArray.zeros([1, numFrames, featureSize])
            let emptyMask = MLXArray.zeros([1, numFrames]).asType(Bool.self)
            return (emptyFeatures, emptyMask)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // fftLength/2-sized split-complex buffers for vDSP_fft_zrip
        var realHalf = [Float](repeating: 0, count: fftLength / 2)
        var imagHalf = [Float](repeating: 0, count: fftLength / 2)
        var magnitudes = [Float](repeating: 0, count: numFreqBins)

        for i in 0..<numFrames {
            // Build zero-padded frame
            let frame = frames[i]

            // Pack real signal into N/2 split-complex: realp[k]=signal[2k], imagp[k]=signal[2k+1]
            // This is the correct packing for vDSP_fft_zrip on a purely real input.
            let halfLen = fftLength / 2
            for k in 0..<halfLen {
                let base = 2 * k
                realHalf[k] = base < frameLength ? frame[base] : 0.0
                imagHalf[k] = (base + 1) < frameLength ? frame[base + 1] : 0.0
            }

            realHalf.withUnsafeMutableBufferPointer { rBuf in
                imagHalf.withUnsafeMutableBufferPointer { iBuf in
                    var splitComplex = DSPSplitComplex(
                        realp: rBuf.baseAddress!,
                        imagp: iBuf.baseAddress!
                    )
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Forward))

                    // vDSP_fft_zrip scales by 2× relative to numpy's rfft.
                    // Multiply by 0.5 to match numpy magnitudes exactly.
                    magnitudes[0] = abs(splitComplex.realp[0]) * 0.5           // DC
                    if numFreqBins > fftLength / 2 {
                        magnitudes[fftLength / 2] = abs(splitComplex.imagp[0]) * 0.5  // Nyquist
                    }
                    for k in 1..<(fftLength / 2) {
                        let r = splitComplex.realp[k]
                        let im = splitComplex.imagp[k]
                        magnitudes[k] = sqrt(r * r + im * im) * 0.5
                    }
                }
            }

            // Apply mel filter bank: mel = magnitudes @ melFilters
            for m in 0..<featureSize {
                var sum: Float = 0
                for k in 0..<numFreqBins {
                    sum += magnitudes[k] * melFilters[k][m]
                }
                var logVal = log(sum + melFloor)
                if let mean = perBinMean, mean.count == featureSize {
                    logVal -= mean[m]
                }
                if let stddev = perBinStddev, stddev.count == featureSize {
                    logVal /= stddev[m]
                }
                melSpectrogram[i][m] = logVal
            }
        }

        // Build frame-level attention mask
        // A frame is valid if the last sample in its unfold window was valid
        var frameMask = [Bool](repeating: false, count: numFrames)
        for i in 0..<numFrames {
            let frameEndIdx = i * hopLength + frameSizeForUnfold - 1
            if frameEndIdx < attentionMask.count {
                frameMask[i] = attentionMask[frameEndIdx] == 1
            }
        }

        // Zero out padded frames (matching Python: spec * mask[..., None])
        for i in 0..<numFrames {
            if !frameMask[i] {
                for m in 0..<featureSize {
                    melSpectrogram[i][m] = 0
                }
            }
        }

        // Convert to MLXArray [1, numFrames, featureSize]
        let flatMel = melSpectrogram.flatMap { $0 }
        let features = MLXArray(flatMel, [1, numFrames, featureSize])

        // Match Hugging Face Gemma4AudioFeatureExtractor: true = valid, false = padding
        let mask = MLXArray(frameMask, [1, numFrames])

        return (features, mask)
    }
}
