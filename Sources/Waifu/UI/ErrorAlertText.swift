import Foundation

enum ErrorAlertText {
    static func sanitized(_ message: String) -> String {
        let lowercased = message.lowercased()
        let hasDomainCode = message.range(
            of: #"(?i)[a-z0-9_.-]*domain\s+error\s+-?\d+"#,
            options: .regularExpression
        ) != nil
        let isOpaqueOperationFailure = lowercased.contains("the operation couldn't be completed")
            || lowercased.contains("the operation couldn’t be completed")

        guard hasDomainCode || isOpaqueOperationFailure else { return message }
        return "That request couldn’t be completed. Please try again."
    }
}
