import AppKit
import Darwin
import SwiftUI

/// Settings → Remote.
///
/// Two independent halves, because a machine may be both a server and a client,
/// and either side may be off. The server half is the owner of the key; the
/// client half is the borrower.
struct RemoteSettingsView: View {
    @ObservedObject private var server = RemoteServer.shared
    @ObservedObject private var client = RemoteClient.shared
    @State private var rotateConfirmation = false
    @State private var revealKey = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                serverSection
                Divider()
                clientSection
            }
            .padding(.top, 4)
        }
    }

    // MARK: Server

    private var serverSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Serve inference from this Mac").font(.headline)
                    Text("Run tagging and Uncover on this Mac for another WaifuFinder to borrow.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("Serve inference", isOn: servingBinding)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
            statusRow(server.state.label, isActive: isServing)

            if isServing {
                labelled("Address") {
                    HStack(spacing: 8) {
                        Text("\(LocalAddress.current):\(port)").font(.caption.monospaced())
                            .foregroundStyle(.secondary).textSelection(.enabled)
                        Button("Copy") {
                            let port = port
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString("\(LocalAddress.current):\(port)",
                                                          forType: .string)
                        }
                        .controlSize(.small)
                    }
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Access key").font(.callout.weight(.medium))
                        Spacer()
                        Button(revealKey ? "Hide" : "Reveal") { revealKey.toggle() }
                            .controlSize(.small)
                        Button("Copy") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(server.keyDisplay, forType: .string)
                        }
                        .controlSize(.small)
                        Button("Rotate…", role: .destructive) { rotateConfirmation = true }
                            .controlSize(.small)
                    }
                    GroupBox {
                        VStack(alignment: .leading, spacing: 5) {
                            Text(server.keyDisplay)
                                .font(.caption.monospaced())
                                .textSelection(.enabled)
                                .redacted(reason: revealKey ? [] : .placeholder)
                            Text(generatedText)
                                .font(.caption2).foregroundStyle(.secondary)
                            Text("Frames are encrypted. Share this key only with Macs you trust.")
                                .font(.caption2).foregroundStyle(.tertiary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                connectedClient
            }
        }
        .confirmationDialog("Generate a new access key?",
                            isPresented: $rotateConfirmation, titleVisibility: .visible) {
            Button("Rotate Key", role: .destructive) { server.rotateKey() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("The current key stops working immediately and any connected client is disconnected. "
                 + "The new key has to be pasted into that client.")
        }
    }

    private var connectedClient: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Connected client").font(.callout.weight(.medium))
            if let client = server.client {
                GroupBox {
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(client.name).font(.callout.weight(.medium))
                            Text(client.address).font(.caption.monospaced()).foregroundStyle(.secondary)
                            Text(clientStatus(client))
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Kick", role: .destructive) { server.kick() }.controlSize(.small)
                    }
                }
                if server.stopAfterCurrentJob {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Will stop when the current job finishes").font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                Text("No client is connected.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private func clientStatus(_ client: RemoteServer.Client) -> String {
        let since = client.connectedSince.formatted(date: .omitted, time: .standard)
        if let job = client.currentJob { return "\(job) · since \(since)" }
        return "Idle · since \(since)"
    }

    private var generatedText: String {
        guard let date = server.keyGeneratedAt else { return "Generated on first start" }
        return "Generated \(date.formatted(date: .abbreviated, time: .shortened))"
    }

    private var isServing: Bool { if case .listening = server.state { return true } else { return false } }
    private var port: UInt16 { if case .listening(let value) = server.state { return value } else { return RemoteLimits.defaultPort } }

    private var servingBinding: Binding<Bool> {
        Binding(get: { isServing }, set: { $0 ? server.start() : server.stop() })
    }

    // MARK: Client

    private var clientSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Use an inference server").font(.headline)
                    Text("Borrow tagging and Uncover from another Mac while keeping this one in charge of the video.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Toggle("Use a server", isOn: Binding(get: { client.isEnabled },
                                                     set: { client.setEnabled($0) }))
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
            HStack {
                statusRow(client.status.label, isActive: client.status.isConnected)
                if client.status.isConnected {
                    Button("Disconnect") { client.disconnect() }.controlSize(.small)
                }
            }

            if client.isEnabled {
                discoveredServers
                manualAddress
                capabilitiesSummary
            }
        }
    }

    private var discoveredServers: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Discovered servers").font(.callout.weight(.medium))
                Spacer()
                Button { client.refreshServers() } label: { Image(systemName: "arrow.clockwise") }
                    .controlSize(.small)
                    .help("Refresh the list")
            }
            if client.servers.isEmpty {
                Text("No WaifuFinder servers are advertising on this network.")
                    .font(.caption).foregroundStyle(.tertiary)
            } else {
                VStack(spacing: 6) {
                    ForEach(client.servers) { server in
                        discoveredRow(server)
                    }
                }
            }
        }
    }

    private func discoveredRow(_ server: DiscoveredServer) -> some View {
        GroupBox {
            HStack(spacing: 8) {
                Image(systemName: server.busy ? "circle.fill" : "circle")
                    .foregroundStyle(server.busy ? Color.orange : Color.green)
                    .imageScale(.small)
                VStack(alignment: .leading, spacing: 2) {
                    Text(server.name).font(.callout.weight(.medium))
                    Text("\(server.address) · \(server.appVersion)").font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(keyLabel(server)).font(.caption).foregroundStyle(.secondary)
                Button("Connect") { client.connect(to: server.address) }
                    .controlSize(.small)
                    .disabled(server.busy && client.status.isConnected)
                Button { client.forget(address: server.address) } label: {
                    Image(systemName: "trash")
                }
                .controlSize(.small).help("Forget this server")
            }
        }
    }

    private func keyLabel(_ server: DiscoveredServer) -> String {
        server.keyState(storedKeyID: client.keyID(for: server.address)).label
    }

    private var manualAddress: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Manual address").font(.callout.weight(.medium))
            HStack(spacing: 8) {
                TextField("host:port", text: $client.manualAddress)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 240)
                Button("Connect") { client.connect(manual: true) }
                    .disabled(RemoteAddress.parse(client.manualAddress) == nil)
            }
            HStack(spacing: 8) {
                SecureField("Access key", text: $client.manualKey)
                    .textFieldStyle(.roundedBorder)
                    .textContentType(nil)
                    .autocorrectionDisabled()
                    .frame(maxWidth: 420)
                Text("Paste the server’s key, then Connect").font(.caption).foregroundStyle(.secondary)
            }
            if let message = client.keyMessage {
                Text(message).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var capabilitiesSummary: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("On this server").font(.callout.weight(.medium))
            if let capabilities = client.capabilities {
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        capabilityRow("Tagging",
                                      ready: capabilities.offersRemoteTagging,
                                      detail: capabilities.tagger == nil
                                        ? "Not installed on the server"
                                        : "Ready · \(capabilities.tagger?.headCount.formatted() ?? "") heads")
                        ForEach(capabilities.uncover, id: \.algorithm) { entry in
                            capabilityRow(UncoverAlgorithm(rawValue: entry.algorithm)?.label ?? entry.algorithm,
                                          ready: entry.available,
                                          detail: entry.available ? entry.message : entry.message)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                Text("Not connected, so nothing is known about the server's models.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private func capabilityRow(_ title: String, ready: Bool, detail: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: ready ? "checkmark.circle.fill" : "circle.dashed")
                .foregroundStyle(ready ? Color.green : Color.secondary).imageScale(.small)
            Text(title).font(.caption)
            Text(detail).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
        }
    }

    private func labelled<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title).font(.caption).frame(width: 70, alignment: .trailing)
            content()
            Spacer()
        }
    }

    private func statusRow(_ text: String, isActive: Bool) -> some View {
        HStack(spacing: 6) {
            Circle().fill(isActive ? Color.green : Color.secondary).frame(width: 8, height: 8)
            Text(text).font(.caption).foregroundStyle(.secondary)
            Spacer()
        }
    }
}

/// The address a server tells a client to use, for the Copy button.
///
/// The listener's own bound port is authoritative; the local address is only
/// needed so the copied string is dialable from another machine.
enum LocalAddress {
    /// The IPv4 address other machines on this network should dial.
    ///
    /// `en0` is Wi-Fi and is preferred when present; the first other
    /// non-loopback interface is the fallback, which covers Ethernet. Returning
    /// `127.0.0.1` is deliberately useless rather than wrong: a copied address
    /// that silently points at the wrong machine would be worse than one that
    /// obviously cannot work.
    static var current: String {
        var preferred: String?
        var fallback: String?
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let first = list else { return "127.0.0.1" }
        defer { freeifaddrs(first) }
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let pointer = cursor {
            let interface = pointer.pointee
            let flags = Int32(interface.ifa_flags)
            let address = interface.ifa_addr
            defer { cursor = interface.ifa_next }
            guard let socket = address, flags & IFF_UP == IFF_UP, flags & IFF_LOOPBACK == 0,
                  socket.pointee.sa_family == UInt8(AF_INET),
                  let named = interface.ifa_name else { continue }
            let name = String(cString: named)
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(socket, socklen_t(socket.pointee.sa_len), &host, socklen_t(host.count),
                              nil, 0, NI_NUMERICHOST) == 0 else { continue }
            let value = String(decoding: host.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
            if name == "en0" { preferred = value } else if fallback == nil { fallback = value }
        }
        return preferred ?? fallback ?? "127.0.0.1"
    }
}
