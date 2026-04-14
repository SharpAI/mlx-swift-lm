import Foundation
import MLXLMCommon

@main
struct App {
    static func main() async throws {
        let registry = ModelRegistry.shared
        print(registry)
    }
}
