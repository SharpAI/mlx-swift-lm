// Copyright © 2026 Apple Inc.

import Foundation

/// Schema-guided reading of textual arguments, shared by all call dialects.
/// This is intentionally separate from validation: unconvertible or ambiguous
/// values keep their original spelling so the caller can reject or inspect them.
enum ToolArgumentNormalization {
    static func normalize(_ call: ToolCall, tools: [[String: any Sendable]]?) -> ToolCall {
        guard !call.function.arguments.isEmpty, let tools,
            let schema = ToolSchemaValidator.parametersSchema(
                ofToolNamed: call.function.name, in: tools),
            let properties = schema["properties"] as? [String: any Sendable]
        else { return call }
        var arguments = call.function.arguments
        var changed = false
        for (key, value) in call.function.arguments {
            let converted = normalize(value, schema: properties[key] as? [String: any Sendable])
            if converted != value {
                arguments[key] = converted
                changed = true
            }
        }
        guard changed else { return call }
        return ToolCall(
            function: .init(name: call.function.name, arguments: arguments), id: call.id)
    }

    static func normalize(
        _ value: JSONValue, schema: [String: any Sendable]?, depth: Int = 0
    ) -> JSONValue {
        switch value {
        case .string, .array, .object: break
        default: return value
        }
        if schema?["type"] as? String == "string" { return value }
        guard depth < 32, let schema, let types = declaredTypes(schema, depth: depth) else {
            return value
        }
        var nonNull = types.subtracting(["null"])
        if nonNull.contains("number") { nonNull.remove("integer") }
        // An admitted string or multiple interpretations is not an invitation
        // to guess which union branch the model meant (for example "001").
        guard !types.contains("string"), nonNull.count <= 1 else { return value }
        var result = value
        if case .string(let text) = value {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if types.contains("null"), ["null", "none", "nil"].contains(trimmed.lowercased()) {
                return .null
            }
            switch nonNull.first {
            case "integer", "number":
                result = number(trimmed, integer: nonNull.first == "integer") ?? value
            case "boolean":
                switch trimmed.lowercased() {
                case "true", "1", "yes", "on": result = .bool(true)
                case "false", "0", "no", "off": result = .bool(false)
                default: break
                }
            case "object", "array":
                let parsed: (any Sendable)? =
                    tryParseJSON(trimmed) ?? tryParsePythonLiteral(trimmed)
                if let parsed {
                    let json = JSONValue.from(parsed)
                    switch (nonNull.first, json) {
                    case ("object", .object), ("array", .array): result = json
                    default: break
                    }
                }
            default: break
            }
        }
        switch result {
        case .object(let object):
            let properties = schema["properties"] as? [String: any Sendable] ?? [:]
            let extra = schema["additionalProperties"] as? [String: any Sendable]
            guard !properties.isEmpty || extra != nil else { return result }
            var normalized = object
            for (key, value) in object {
                let converted = normalize(
                    value, schema: properties[key] as? [String: any Sendable] ?? extra,
                    depth: depth + 1)
                if converted != value { normalized[key] = converted }
            }
            return .object(normalized)
        case .array(let array):
            guard let items = schema["items"] as? [String: any Sendable] else { return result }
            var normalized = array
            for index in array.indices {
                let converted = normalize(array[index], schema: items, depth: depth + 1)
                if converted != array[index] { normalized[index] = converted }
            }
            return .array(normalized)
        default: return result
        }
    }

    /// Returns only interpretations every branch describes. Unknown branches
    /// (notably references) prevent coercion rather than being silently ignored.
    private static func declaredTypes(_ schema: [String: any Sendable], depth: Int) -> Set<String>?
    {
        guard depth < 32,
            ["$ref", "$dynamicRef", "not", "if", "then", "else"].allSatisfy({ schema[$0] == nil })
        else { return nil }
        var result: Set<String>?
        if let raw = schema["type"] {
            let names = (raw as? String).map { [$0] } ?? (raw as? [String]) ?? []
            let mapped = names.compactMap(canonicalType)
            guard !names.isEmpty, mapped.count == names.count else { return nil }
            result = Set(mapped)
            // Integer is a subtype of number, so intersections such as
            // allOf(number, integer) still describe an integer.
            if result?.contains("number") == true { result?.insert("integer") }
        }
        for key in ["anyOf", "oneOf", "allOf"] where schema[key] != nil {
            guard let branches = schema[key] as? [[String: any Sendable]], !branches.isEmpty else {
                return nil
            }
            let types = branches.compactMap { declaredTypes($0, depth: depth + 1) }
            guard types.count == branches.count else { return nil }
            let combined = types.dropFirst().reduce(types[0]) {
                key == "allOf" ? $0.intersection($1) : $0.union($1)
            }
            result = result.map { $0.intersection(combined) } ?? combined
        }
        return result
    }

    private static func canonicalType(_ name: String) -> String? {
        let type = name.lowercased()
        switch type {
        case "string", "str", "text", "varchar", "char", "enum": return "string"
        case "integer", "int", "uint", "long", "short", "unsigned": return "integer"
        case "number", "float": return "number"
        case "boolean", "bool", "binary": return "boolean"
        case "object", "dict": return "object"
        case "array", "arr", "list": return "array"
        case "null": return "null"
        default:
            // Preserve the aliases already accepted by the textual parsers.
            if ["int", "uint", "long", "short", "unsigned"].contains(where: type.hasPrefix) {
                return "integer"
            }
            if type.hasPrefix("num") || type.hasPrefix("float") { return "number" }
            if type.hasPrefix("dict") { return "object" }
            if type.hasPrefix("list") { return "array" }
            return nil
        }
    }

    private static func number(_ text: String, integer: Bool) -> JSONValue? {
        if let value = Int(text) { return .int(value) }
        guard let value = Double(text), value.isFinite else { return nil }
        return decimalInteger(text).map(JSONValue.int) ?? (integer ? nil : .double(value))
    }

    /// Recognize integral decimal/exponent spellings without floating-point
    /// rounding. No fixed-width conversion occurs until all digits are known.
    private static func decimalInteger(_ text: String) -> Int? {
        var body = text[...]
        let negative = body.first == "-"
        if body.first == "-" || body.first == "+" { body.removeFirst() }
        let parts = body.split(
            omittingEmptySubsequences: false, whereSeparator: { $0 == "e" || $0 == "E" })
        guard parts.count <= 2 else { return nil }
        let exponent: Int
        if parts.count == 2 {
            guard let parsed = Int(parts[1]) else { return nil }
            exponent = parsed
        } else {
            exponent = 0
        }
        let mantissa = parts[0].split(separator: ".", omittingEmptySubsequences: false)
        guard mantissa.count <= 2 else { return nil }
        let fractionalCount = mantissa.count == 2 ? mantissa[1].count : 0
        var digits = mantissa.joined()
        guard !digits.isEmpty, digits.utf8.allSatisfy({ (48 ... 57).contains($0) }) else {
            return nil
        }
        digits = String(digits.drop(while: { $0 == "0" }))
        if digits.isEmpty { return 0 }
        let (scale, overflow) = exponent.subtractingReportingOverflow(fractionalCount)
        guard !overflow else { return nil }
        if scale >= 0 {
            guard scale <= 19, digits.count <= 19 - scale else { return nil }
            digits += String(repeating: "0", count: scale)
        } else {
            guard scale > -digits.count, digits.suffix(-scale).allSatisfy({ $0 == "0" }) else {
                return nil
            }
            digits.removeLast(-scale)
        }
        return Int((negative ? "-" : "") + digits)
    }
}
