// SharpAI fork overlay: qwen3_asr prompt, output parsing, and forced-alignment helpers. Not part
// of upstream mlx-swift-lm.
//
// These are plain functions with no model state. Ports of transformers' Qwen3ASR processor:
// prompt wrapping, "language X<asr_text>" parsing, repetition cleanup, and timestamp repair.
// UNVERIFIED: the chat template is not in the repo. The wrapper below follows vLLM's
// get_generation_prompt, and the processor's language prefill is treated as the same string.

import Foundation

public struct SharpAIQwen3AsrTranscript: Equatable, Sendable {
    /// Language name from the model, or nil when detection gave none.
    public var language: String?
    public var transcription: String

    public init(language: String?, transcription: String) {
        self.language = language
        self.transcription = transcription
    }
}

public struct SharpAIQwen3AsrWordTiming: Equatable, Sendable {
    public var text: String
    public var startTime: Double
    public var endTime: Double

    public init(text: String, startTime: Double, endTime: Double) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
    }
}

public enum SharpAIQwen3AsrTranscription {

    public static let asrTextTag = "<asr_text>"
    public static let audioStartToken = "<|audio_start|>"
    public static let audioPadToken = "<|audio_pad|>"
    public static let audioEndToken = "<|audio_end|>"

    /// Language codes to the names the model was trained with.
    public static let languageNames: [String: String] = [
        "ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech", "da": "Danish",
        "nl": "Dutch", "en": "English", "fil": "Filipino", "fi": "Finnish", "fr": "French",
        "de": "German", "el": "Greek", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "mk": "Macedonian", "ms": "Malay",
        "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
        "es": "Spanish", "sv": "Swedish", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
    ]

    /// Languages the forced aligner supports, by name.
    public static let forcedAlignerLanguages: Set<String> = [
        "Chinese", "English", "Cantonese", "French", "German", "Italian", "Japanese",
        "Korean", "Portuguese", "Russian", "Spanish",
    ]

    /// Maps a code like "en" or a full name like "English" to the trained name.
    public static func languageName(for codeOrName: String) -> String? {
        if let name = languageNames[codeOrName] {
            return name
        }
        if languageNames.values.contains(codeOrName) {
            return codeOrName
        }
        return nil
    }

    /// Chat prompt for one utterance. `audioTokenCount` comes from
    /// `SharpAIQwen3AsrAudioMath.audioTokenCount`. `context` becomes the system turn, for hotwords.
    /// `language` forces the output language through the assistant prefill.
    public static func prompt(
        audioTokenCount: Int, language: String? = nil, context: String? = nil
    ) throws -> String {
        guard audioTokenCount > 0 else {
            throw SharpAIQwen3AsrError.invalidInput("audio token count must be positive")
        }
        var text = ""
        if let context {
            text += "<|im_start|>system\n\(context)<|im_end|>\n"
        }
        let pads = String(repeating: audioPadToken, count: audioTokenCount)
        text += "<|im_start|>user\n\(audioStartToken)\(pads)\(audioEndToken)<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        if let language {
            guard let name = languageName(for: language) else {
                throw SharpAIQwen3AsrError.invalidInput("unknown language \(language)")
            }
            text += "language \(name)\(asrTextTag)"
        }
        return text
    }

    /// Parses one decoded output string, such as "language English<asr_text>Hello".
    public static func parse(_ raw: String) -> SharpAIQwen3AsrTranscript {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty {
            return SharpAIQwen3AsrTranscript(language: nil, transcription: "")
        }
        if let range = text.range(of: "assistant\n") {
            text = String(text[range.upperBound...])
        }
        text = fixRepetitions(text)

        guard let tagRange = text.range(of: asrTextTag) else {
            return SharpAIQwen3AsrTranscript(
                language: nil, transcription: text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let prefix = String(text[..<tagRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let transcription = String(text[tagRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if prefix.lowercased() == "language none" {
            return SharpAIQwen3AsrTranscript(language: nil, transcription: transcription)
        }

        // Only the first non-empty prefix line counts, as in the processor.
        var language: String?
        for rawLine in prefix.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                continue
            }
            let marker = "language "
            if line.lowercased().hasPrefix(marker) {
                let value = String(line.dropFirst(marker.count))
                    .trimmingCharacters(in: .whitespaces)
                if !value.isEmpty {
                    language = value
                }
            } else {
                language = line
            }
            break
        }
        return SharpAIQwen3AsrTranscript(language: language, transcription: transcription)
    }

    /// Word timings from the aligner's bin predictions at each timestamp token.
    /// `predictedBins` holds two bins per word, start then end, in token order.
    public static func alignmentItems(
        words: [String], predictedBins: [Int], segmentTimeMs: Double
    ) throws -> [SharpAIQwen3AsrWordTiming] {
        guard predictedBins.count >= words.count * 2 else {
            throw SharpAIQwen3AsrError.invalidInput(
                "need \(words.count * 2) timestamps for \(words.count) words, got \(predictedBins.count)"
            )
        }
        let raw = predictedBins.map { Double($0) * segmentTimeMs }
        let fixed = fixTimestamps(raw)
        return words.enumerated().map { index, word in
            SharpAIQwen3AsrWordTiming(
                text: word,
                startTime: Double(fixed[index * 2]) / 1000.0,
                endTime: Double(fixed[index * 2 + 1]) / 1000.0)
        }
    }

    /// Makes timestamps non-decreasing. Keeps the longest increasing run, then replaces outliers
    /// by snapping short gaps (<= 2) and interpolating longer ones.
    public static func fixTimestamps(_ data: [Double]) -> [Int] {
        let n = data.count
        if n == 0 {
            return []
        }

        var dp = Array(repeating: 1, count: n)
        var parent = Array(repeating: -1, count: n)
        for current in 1 ..< n {
            for prev in 0 ..< current where data[prev] <= data[current] && dp[prev] + 1 > dp[current] {
                dp[current] = dp[prev] + 1
                parent[current] = prev
            }
        }

        let maxLength = dp.max() ?? 1
        var index = dp.firstIndex(of: maxLength) ?? 0
        var chain: [Int] = []
        while index != -1 {
            chain.append(index)
            index = parent[index]
        }
        var isNormal = Array(repeating: false, count: n)
        for position in chain {
            isNormal[position] = true
        }

        var result = data
        var blockStart = 0
        while blockStart < n {
            if isNormal[blockStart] {
                blockStart += 1
                continue
            }
            var blockEnd = blockStart
            while blockEnd < n && !isNormal[blockEnd] {
                blockEnd += 1
            }
            let anomalyCount = blockEnd - blockStart

            var leftValue: Double?
            var scan = blockStart - 1
            while scan >= 0 && leftValue == nil {
                if isNormal[scan] {
                    leftValue = result[scan]
                }
                scan -= 1
            }
            var rightValue: Double?
            scan = blockEnd
            while scan < n && rightValue == nil {
                if isNormal[scan] {
                    rightValue = result[scan]
                }
                scan += 1
            }

            if anomalyCount <= 2 {
                for pos in blockStart ..< blockEnd {
                    if let left = leftValue, let right = rightValue {
                        let useLeft = (pos - (blockStart - 1)) <= (blockEnd - pos)
                        result[pos] = useLeft ? left : right
                    } else if let left = leftValue {
                        result[pos] = left
                    } else if let right = rightValue {
                        result[pos] = right
                    }
                }
            } else if let left = leftValue, let right = rightValue {
                let step = (right - left) / Double(anomalyCount + 1)
                for pos in blockStart ..< blockEnd {
                    result[pos] = left + step * Double(pos - blockStart + 1)
                }
            } else if let left = leftValue {
                for pos in blockStart ..< blockEnd {
                    result[pos] = left
                }
            } else if let right = rightValue {
                for pos in blockStart ..< blockEnd {
                    result[pos] = right
                }
            }
            blockStart = blockEnd
        }

        return result.map { Int($0) }
    }

    // MARK: - Repetition cleanup (port of the processor's _detect_and_fix_repetitions)

    static func fixRepetitions(_ text: String, threshold: Int = 20) -> String {
        let charFixed = fixCharRepeats(Array(text.unicodeScalars), threshold: threshold)
        let patternFixed = fixPatternRepeats(charFixed, threshold: threshold, maxLength: 20)
        var view = String.UnicodeScalarView()
        view.append(contentsOf: patternFixed)
        return String(view)
    }

    static func fixCharRepeats(_ s: [Unicode.Scalar], threshold: Int) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        var i = 0
        let n = s.count
        while i < n {
            var count = 1
            while i + count < n && s[i + count] == s[i] {
                count += 1
            }
            if count > threshold {
                result.append(s[i])
            } else {
                result.append(contentsOf: s[i ..< i + count])
            }
            i += count
        }
        return result
    }

    static func fixPatternRepeats(
        _ s: [Unicode.Scalar], threshold: Int, maxLength: Int
    ) -> [Unicode.Scalar] {
        let n = s.count
        let minRepeat = threshold * 2
        if n < minRepeat {
            return s
        }

        var i = 0
        var result: [Unicode.Scalar] = []
        var found = false
        while i <= n - minRepeat {
            found = false
            for k in 1 ... maxLength {
                if i + k * threshold > n {
                    break
                }
                let pattern = Array(s[i ..< i + k])
                var valid = true
                for rep in 1 ..< threshold {
                    let start = i + rep * k
                    if Array(s[start ..< start + k]) != pattern {
                        valid = false
                        break
                    }
                }
                if valid {
                    var endIndex = i + threshold * k
                    while endIndex + k <= n && Array(s[endIndex ..< endIndex + k]) == pattern {
                        endIndex += k
                    }
                    result.append(contentsOf: pattern)
                    result.append(
                        contentsOf: fixPatternRepeats(
                            Array(s[endIndex...]), threshold: threshold, maxLength: maxLength))
                    i = n
                    found = true
                    break
                }
            }
            if found {
                break
            }
            result.append(s[i])
            i += 1
        }

        if !found {
            result.append(contentsOf: s[min(i, n)...])
        }
        return result
    }
}
