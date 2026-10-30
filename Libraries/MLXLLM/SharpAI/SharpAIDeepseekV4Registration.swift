// SL-41 (red-phase stub): the overlay registration for deepseek_v4.
// Deliberately wrong placeholder semantics (base wins, no creators) so the SL-41 tests fail
// for the right reason; the next commit replaces it.

import Foundation
import MLXLMCommon

public enum SharpAIDeepseekV4Overlay {
    public typealias Creator = (Data) throws -> any LanguageModel

    public static var creators: [String: Creator] { [:] }

    public static func applying(to base: [String: Creator]) -> [String: Creator] {
        base
    }
}
