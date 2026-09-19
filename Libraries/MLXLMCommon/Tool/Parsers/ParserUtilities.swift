// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - JSON to Sendable Bridge

/// Convert a JSON-deserialized value to `any Sendable`.
///
/// `JSONSerialization` returns `Any`, but all JSON types it produces
/// (String, NSNumber, NSNull, Array, Dictionary) are Sendable.
func asSendable(_ value: Any) -> any Sendable {
    switch value {
    case let s as String: return s
    case let n as NSNumber: return n
    case let a as [Any]: return a.map(asSendable)
    case let d as [String: Any]: return d.mapValues(asSendable)
    case let null as NSNull: return null
    default: return "\(value)"
    }
}

/// Deserialize JSON data, returning a Sendable value.
func deserializeJSON(_ data: Data) -> (any Sendable)? {
    guard let object = try? JSONSerialization.jsonObject(with: data) else { return nil }
    return asSendable(object)
}

// MARK: - Basic Deserialization

/// Deserialize a string value to JSON or return as string.
///
/// Attempts JSON parsing first, falling back to the original string value.
/// Reference: Python's `ast.literal_eval` / `json.loads` pattern
func tryParseJSON(_ value: String) -> (any Sendable)? {
    guard let data = value.data(using: .utf8) else { return nil }
    return deserializeJSON(data)
}

func deserialize(_ value: String) -> any Sendable {
    tryParseJSON(value) ?? value
}

// MARK: - Schema Lookup Functions

/// Check if a parameter is a string type in the tool schema.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
func isStringType(funcName: String, argName: String, tools: [[String: any Sendable]]?) -> Bool {
    guard let type = getParameterType(funcName: funcName, paramName: argName, tools: tools) else {
        return false
    }
    return type == "string"
}

/// Get the parameter type from tool schema for a specific function and parameter.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
func getParameterType(
    funcName: String, paramName: String, tools: [[String: any Sendable]]?
) -> String? {
    guard let tools else { return nil }
    for tool in tools {
        guard let function = tool["function"] as? [String: any Sendable],
            function["name"] as? String == funcName,
            let parameters = function["parameters"] as? [String: any Sendable],
            let properties = parameters["properties"] as? [String: any Sendable],
            let param = properties[paramName] as? [String: any Sendable],
            let type = param["type"] as? String
        else { continue }
        return type
    }
    return nil
}

/// Get parameter configuration for a function from tools schema.
func getParameterConfig(
    funcName: String, tools: [[String: any Sendable]]?
) -> [String: any Sendable] {
    guard let tools else { return [:] }
    for tool in tools {
        guard let function = tool["function"] as? [String: any Sendable],
            function["name"] as? String == funcName,
            let parameters = function["parameters"] as? [String: any Sendable],
            let properties = parameters["properties"] as? [String: any Sendable]
        else { continue }
        return properties
    }
    return [:]
}

// MARK: - Schema Type Extraction

/// Extract types from JSON schema (handles anyOf, oneOf, allOf, enums).
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func extractTypesFromSchema(_ schema: [String: any Sendable]?) -> [String] {
    guard let schema else { return ["string"] }

    var types: Set<String> = []

    // Handle direct "type" field
    if let typeValue = schema["type"] {
        if let typeString = typeValue as? String {
            types.insert(typeString)
        } else if let typeArray = typeValue as? [String] {
            types.formUnion(typeArray)
        }
    }

    // Handle enum - infer types from enum values
    if let enumValues = schema["enum"] as? [any Sendable], !enumValues.isEmpty {
        for value in enumValues {
            switch value {
            case is NSNull: types.insert("null")
            case is Bool: types.insert("boolean")
            case is Int: types.insert("integer")
            case is Double: types.insert("number")
            case is String: types.insert("string")
            case is [any Sendable]: types.insert("array")
            case is [String: any Sendable]: types.insert("object")
            default: break
            }
        }
    }

    // Handle anyOf, oneOf, allOf - recursively extract types
    for choiceField in ["anyOf", "oneOf", "allOf"] {
        if let choices = schema[choiceField] as? [[String: any Sendable]] {
            for choice in choices {
                types.formUnion(extractTypesFromSchema(choice))
            }
        }
    }

    return types.isEmpty ? ["string"] : Array(types)
}

// MARK: - Slicing

extension Substring {
    /// The slice without leading or trailing whitespace, avoiding the copy that
    /// `trimmingCharacters(in:)` makes when the caller only needs a view.
    func trimmingWhitespace() -> Substring {
        var slice = drop(while: \.isWhitespace)
        while let last = slice.last, last.isWhitespace {
            slice = slice.dropLast()
        }
        return slice
    }
}

// MARK: - Type Conversion

/// Whether a generated function name belongs to the caller-provided tool set.
/// An absent or empty schema list preserves parser-only use cases.
func isDeclaredTool(
    _ functionName: String, tools: [[String: any Sendable]]?
) -> Bool {
    guard let tools, !tools.isEmpty else { return true }
    return tools.contains { tool in
        let function = tool["function"] as? [String: any Sendable]
        return function?["name"] as? String == functionName
    }
}

/// Convert parameter value based on multiple possible types.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func convertValueWithTypes(_ value: String, types: [String]) -> any Sendable {
    ToolArgumentNormalization.normalize(.string(value), schema: ["type": types]).sendableValue
}

/// Read a textual parameter using its declared schema; unknown or ambiguous
/// schemas preserve the original text. Validation is a separate boundary.
func convertParameterValue(
    _ value: String, paramName: String, funcName: String, tools: [[String: any Sendable]]?
) -> any Sendable {
    guard let tools,
        let parameters = ToolSchemaValidator.parametersSchema(ofToolNamed: funcName, in: tools),
        let properties = parameters["properties"] as? [String: any Sendable]
    else { return value }
    return ToolArgumentNormalization.normalize(
        .string(value), schema: properties[paramName] as? [String: any Sendable]
    ).sendableValue
}

// MARK: - String Utilities

/// Extract name from a potentially quoted string.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func extractName(_ nameStr: String) -> String {
    let trimmed = nameStr.trimmingCharacters(in: .whitespaces)
    if (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\""))
        || (trimmed.hasPrefix("'") && trimmed.hasSuffix("'"))
    {
        return String(trimmed.dropFirst().dropLast())
    }
    return trimmed
}
