import Foundation
import Network

/// Address parsing and validation for the client.
///
/// One gate, in the spirit of `MediaImport.webURL` being the single place a
/// pasted location is inspected: nothing below this opens a socket, and nothing
/// above it has to re-check what came out.
enum RemoteAddress {
    struct Endpoint: Equatable, Sendable {
        var host: String
        var port: UInt16

        var display: String { host.contains(":") ? "[\(host)]:\(port)" : "\(host.lowercased()):\(port)" }
        func networkEndpoint() -> NWEndpoint {
            .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port) ?? .any)
        }
        /// The port as the beacon port, so a beacon sender can reuse the
        /// endpoint construction rather than repeating it.
        static func broadcast(port: UInt16) -> NWEndpoint {
            .hostPort(host: NWEndpoint.Host("255.255.255.255"),
                      port: NWEndpoint.Port(rawValue: port) ?? .any)
        }
        static func multicast(group: String, port: UInt16) -> NWEndpoint {
            .hostPort(host: NWEndpoint.Host(group),
                      port: NWEndpoint.Port(rawValue: port) ?? .any)
        }
    }

    /// Accepts `host`, `host:port`, or a bracketed IPv6 literal with a port.
    ///
    /// A bare host gets the default port. Anything else — an empty host, a port
    /// outside 1…65535, a scheme, a path — is rejected here rather than becoming
    /// a confusing socket error later.
    static func parse(_ text: String, defaultPort: UInt16 = RemoteLimits.defaultPort) -> Endpoint? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // A scheme is a paste accident, not an address. Stripping it would make
        // a mistyped URL silently resolve somewhere unexpected, so refuse it.
        guard !trimmed.contains("://") else { return nil }
        guard !trimmed.contains("/") else { return nil }

        var host = trimmed
        var port = defaultPort
        if trimmed.hasPrefix("[") {
            // Bracketed IPv6 literal: [::1] or [::1]:47410
            guard let close = trimmed.firstIndex(of: "]") else { return nil }
            host = String(trimmed[trimmed.index(after: trimmed.startIndex)..<close])
            let remainder = trimmed[trimmed.index(after: close)...]
            if !remainder.isEmpty {
                guard remainder.hasPrefix(":"), let value = UInt16(remainder.dropFirst()), value > 0 else {
                    return nil
                }
                port = value
            }
        } else if let colon = trimmed.lastIndex(of: ":"), trimmed[..<colon].contains(":") == false {
            host = String(trimmed[..<colon])
            guard let value = UInt16(trimmed[trimmed.index(after: colon)...]), value > 0 else { return nil }
            port = value
        }
        return isValidHost(host) ? Endpoint(host: host, port: port) : nil
    }

    /// A host is a hostname, an IPv4 literal, or an IPv6 literal. Nothing else:
    /// no whitespace, no shell metacharacters, no path separators.
    static func isValidHost(_ host: String) -> Bool {
        guard !host.isEmpty, host.count <= 253 else { return false }
        if host.contains(":") {
            // IPv6 literal, optionally with a zone index. Colons and dots are
            // permitted inside it; every other character is not.
            let address = host.split(separator: "%", maxSplits: 1, omittingEmptySubsequences: false)[0]
            return !address.isEmpty && address.allSatisfy { $0.isHexDigit || $0 == ":" || $0 == "." }
        }
        let labels = host.split(separator: ".", omittingEmptySubsequences: false)
        guard labels.allSatisfy({ !$0.isEmpty }) else { return false }
        return labels.allSatisfy { label in
            !label.isEmpty && label.count <= 63
                && label.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_") }
                && !label.hasPrefix("-") && !label.hasSuffix("-")
        }
    }
}