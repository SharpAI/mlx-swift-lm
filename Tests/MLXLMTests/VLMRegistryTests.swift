// Copyright © 2026 Apple Inc.

import MLXVLM
import XCTest

final class VLMRegistryTests: XCTestCase {

    func testGemma4VLMRegistryUsesTurnEndToken() {
        // SharpAI fork addition (4f54bcc): Gemma-4 emits <pad> (id 0) instead of a
        // proper EOS once context exceeds the 1024-token sliding window, so <pad>
        // is included alongside the upstream <turn|> end token to prevent an
        // infinite padding loop. Not a plain rename of upstream's expectation.
        for configuration in [
            VLMRegistry.gemma4_E2B_it_4bit,
            VLMRegistry.gemma4_E4B_it_4bit,
            VLMRegistry.gemma4_31B_it_4bit,
            VLMRegistry.gemma4_26BA4B_it_4bit,
        ] {
            XCTAssertEqual(configuration.extraEOSTokens, ["<turn|>", "<pad>"])
        }
    }
}
