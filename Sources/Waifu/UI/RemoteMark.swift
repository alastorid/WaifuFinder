import SwiftUI

/// Appends a superscript **R** to an inference control's label when that request
/// would be executed on the connected server.
///
/// The mark appears only when the connection is live *and* the server advertises
/// the specific capability the request needs — so it is a statement about where
/// the work will happen, not a generic "connected" badge. Its accessibility
/// label reads the glyph aloud as remote work, so it is self-explaining rather
/// than something the user has to decode.
struct RemoteMark: ViewModifier {
    let available: Bool

    func body(content: Content) -> some View {
        HStack(spacing: 1) {
            content
            if available {
                Text("R")
                    .font(.system(size: 9, weight: .semibold))
                    .baselineOffset(5)
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("runs on the connected server")
            }
        }
    }
}

extension View {
    /// Marks an inference control as remote-executable. The mark's presence is
    /// decided by `available` alone, so a caller cannot accidentally show a
    /// glyph for a capability the server does not have.
    func remoteMark(_ available: Bool) -> some View {
        modifier(RemoteMark(available: available))
    }
}

/// The labels the three inference controls share.
///
/// Collected in one place so the three buttons cannot disagree about what the
/// server can currently do — which is the sort of drift that makes a feature feel
/// broken rather than merely unavailable.
struct RemoteLabelContext {
    var isConnected: Bool
    var serverName: String?
    var offersTagging: Bool
    var offersAnyUncover: Bool
    /// Per algorithm, so the Reconstruct button marks itself only for the
    /// algorithm the user has actually selected.
    var availableAlgorithms: Set<String>

    @MainActor
    init(client: RemoteClient) {
        self.init(connected: client.status.isConnected,
                  serverName: client.capabilities?.serverName,
                  capabilities: client.capabilities)
    }

    /// The explicit form.
    ///
    /// A `RemoteLabelContext` is a value derived from connection state, and the
    /// rules it encodes — mark only what the server can run, never on the
    /// strength of stale capabilities alone — are worth stating and testing
    /// without a socket.
    init(connected: Bool, serverName: String?, capabilities: RemoteCapabilities?) {
        isConnected = connected
        self.serverName = serverName
        offersTagging = capabilities?.offersRemoteTagging == true
        offersAnyUncover = capabilities?.offersAnyUncover == true
        availableAlgorithms = Set((capabilities?.uncover ?? [])
            .filter(\.available).map(\.algorithm))
    }

    /// Whether a tagging request would run remotely.
    var tagsRemotely: Bool { isConnected && offersTagging }

    /// Whether opening an Uncover workspace has anything to offer remotely.
    var offersRemoteUncover: Bool { isConnected && offersAnyUncover }

    /// Whether *this* algorithm would run remotely.
    func runsRemotely(_ algorithm: UncoverAlgorithm) -> Bool {
        isConnected && availableAlgorithms.contains(algorithm.rawValue)
    }

    var serverDescription: String {
        guard isConnected, let serverName else { return "a remote server" }
        return serverName
    }
}