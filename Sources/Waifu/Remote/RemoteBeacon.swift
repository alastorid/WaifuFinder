import Foundation
import Network

/// The once-per-second advertisement a serving WaifuFinder broadcasts.
///
/// It is a hint, not a credential. Everything it claims is re-verified by the
/// authenticated TCP handshake, so a forged beacon buys an attacker nothing: the
/// `keyID` it carries is a hash prefix of a 256-bit secret, and the connect
/// address is the datagram's *source IP* rather than the name in the payload,
/// because a `.local` name can resolve differently on two machines.
struct RemoteBeacon: Equatable, Sendable {
    static let magic: [UInt8] = [0x57, 0x46, 0x52, 0x42] // "WFRB"
    static let version: UInt8 = 1
    static let fixedBytes = 40
    static let maximumNameBytes = 63
    static let maximumVersionBytes = 31

    var tcpPort: UInt16
    var serverInstance: UInt64
    var keyID: Data
    /// Unix milliseconds, low 32 bits. A staleness readout only.
    var sentAtMilliseconds: UInt32
    var flags: BeaconFlags
    var name: String
    var appVersion: String

    struct BeaconFlags: OptionSet, Sendable {
        let rawValue: UInt16
        init(rawValue: UInt16) { self.rawValue = rawValue }
        static let accepting = BeaconFlags(rawValue: 1 << 0)
        static let clientConnected = BeaconFlags(rawValue: 1 << 1)
        static let taggerReady = BeaconFlags(rawValue: 1 << 2)
    }

    /// `UncoverAlgorithm.allCases` order, one bit each starting at bit 3.
    func offers(_ algorithm: UncoverAlgorithm) -> Bool {
        guard let index = UncoverAlgorithm.allCases.firstIndex(of: algorithm), index < 5 else { return false }
        return flags.contains(BeaconFlags(rawValue: 1 << (3 + index)))
    }

    private static func put<T: FixedWidthInteger>(_ value: T, into bytes: inout [UInt8], at offset: Int) {
        withUnsafeBytes(of: value.littleEndian) { bytes.replaceSubrange(offset..<(offset + MemoryLayout<T>.size), with: $0) }
    }
    func encoded() -> [UInt8] {
        let nameBytes = Array(name.utf8.prefix(Self.maximumNameBytes))
        let versionBytes = Array(appVersion.utf8.prefix(Self.maximumVersionBytes))
        var bytes = [UInt8](repeating: 0, count: Self.fixedBytes + 1 + nameBytes.count + 1 + versionBytes.count)
        bytes.replaceSubrange(0..<4, with: Self.magic)
        bytes[4] = Self.version
        bytes[5] = UInt8(RemoteLimits.protocolVersion)
        Self.put(flags.rawValue, into: &bytes, at: 6)
        Self.put(tcpPort, into: &bytes, at: 8)
        Self.put(serverInstance, into: &bytes, at: 12)
        let id = [UInt8](keyID.prefix(8))
        bytes.replaceSubrange(20..<(20 + id.count), with: id)
        Self.put(sentAtMilliseconds, into: &bytes, at: 28)
        // bytes 32..<36 stay zero, reserved.
        var offset = Self.fixedBytes
        bytes[offset] = UInt8(nameBytes.count); offset += 1
        bytes.replaceSubrange(offset..<(offset + nameBytes.count), with: nameBytes)
        offset += nameBytes.count
        bytes[offset] = UInt8(versionBytes.count); offset += 1
        bytes.replaceSubrange(offset..<(offset + versionBytes.count), with: versionBytes)
        return bytes
    }

    static func decode(_ bytes: [UInt8]) -> RemoteBeacon? {
        guard bytes.count > fixedBytes, bytes[0..<4].elementsEqual(magic), bytes[4] == version else { return nil }
        guard bytes[5] == UInt8(RemoteLimits.protocolVersion) else { return nil }
        let flags = BeaconFlags(rawValue: get(UInt16.self, from: bytes, at: 6))
        let tcpPort = get(UInt16.self, from: bytes, at: 8)
        let instance = get(UInt64.self, from: bytes, at: 12)
        let keyID = Data(bytes[20..<28])
        let sentAt = get(UInt32.self, from: bytes, at: 28)
        guard get(UInt32.self, from: bytes, at: 32) == 0 else { return nil }
        var offset = fixedBytes
        guard offset < bytes.count else { return nil }
        let nameLength = Int(bytes[offset]); offset += 1
        guard offset + nameLength <= bytes.count else { return nil }
        let nameBytes = bytes[offset..<(offset + nameLength)]
        offset += nameLength
        guard offset < bytes.count else { return nil }
        let versionLength = Int(bytes[offset]); offset += 1
        guard offset + versionLength <= bytes.count else { return nil }
        let versionBytes = bytes[offset..<(offset + versionLength)]
        guard let name = String(bytes: nameBytes, encoding: .utf8),
              let appVersion = String(bytes: versionBytes, encoding: .utf8) else { return nil }
        return RemoteBeacon(tcpPort: tcpPort, serverInstance: instance, keyID: keyID,
                            sentAtMilliseconds: sentAt, flags: flags, name: name, appVersion: appVersion)
    }

    private static func get<T: FixedWidthInteger>(_ type: T.Type, from bytes: [UInt8], at offset: Int) -> T {
        var value: T = 0
        withUnsafeMutableBytes(of: &value) { destination in
            for index in 0..<MemoryLayout<T>.size { destination[index] = bytes[offset + index] }
        }
        return T(littleEndian: value)
    }
}

/// One discovered server, as the client remembers it between beacons.
struct DiscoveredServer: Identifiable, Equatable, Sendable {
    var id: String { address }
    var address: String
    var name: String
    var appVersion: String
    var tcpPort: UInt16
    var keyID: Data
    var offersTagging: Bool
    var availableAlgorithms: Set<String>
    var busy: Bool
    var lastSeen: Date

    func keyState(storedKeyID: Data?) -> RemoteKeyState {
        guard let storedKeyID else { return .none }
        return storedKeyID == keyID ? .matches : .mismatched
    }
}

/// Collects beacons and expires the ones that stopped arriving.
actor RemoteBeaconCache {
    private var servers: [String: DiscoveredServer] = [:]
    private let expiry: TimeInterval

    init(expiry: TimeInterval = 3) { self.expiry = expiry }

    func record(_ beacon: RemoteBeacon, from sourceAddress: String, now: Date = Date()) {
        // The source address is authoritative. A beacon that claims a different
        // port than the one it arrived on is still usable, because the datagram
        // came from that host — but a name is not an address, so it is ignored.
        let address = "\(sourceAddress):\(beacon.tcpPort)"
        servers[address] = DiscoveredServer(
            address: address, name: beacon.name, appVersion: beacon.appVersion, tcpPort: beacon.tcpPort,
            keyID: beacon.keyID, offersTagging: beacon.flags.contains(.taggerReady),
            availableAlgorithms: Set(UncoverAlgorithm.allCases.filter { beacon.offers($0) }.map(\.rawValue)),
            busy: beacon.flags.contains(.clientConnected), lastSeen: now)
    }

    /// Returns the live servers, dropping any that have missed three beats so a
    /// server that quits disappears within about three seconds.
    func live(now: Date = Date()) -> [DiscoveredServer] {
        servers = servers.filter { now.timeIntervalSince($0.value.lastSeen) <= expiry }
        return servers.values.sorted {
            $0.lastSeen == $1.lastSeen ? $0.address < $1.address : $0.lastSeen > $1.lastSeen
        }
    }

    func forget(_ address: String) { servers[address] = nil }
    func clear() { servers.removeAll() }
    func knownAddresses() -> Set<String> { Set(servers.keys) }
}

/// Receives beacons on the beacon port and reports them with their source host.
///
/// One `NWConnectionGroup` joined to the multicast group, which also sees
/// unicast probes and broadcast datagrams on the same port, so one listener
/// covers all three discovery paths.
final class RemoteBeaconReceiver: @unchecked Sendable {
    private let group: NWMulticastGroup?
    private var connection: NWConnectionGroup?
    private let onBeacon: @Sendable (RemoteBeacon, String) -> Void
    private let lock = NSLock()
    private var running = false

    init(port: UInt16 = RemoteLimits.beaconPort,
         multicastGroup: String = RemoteLimits.beaconGroup,
         onBeacon: @escaping @Sendable (RemoteBeacon, String) -> Void) {
        let endpoint = RemoteAddress.Endpoint.multicast(group: multicastGroup, port: port)
        group = try? NWMulticastGroup(for: [endpoint])
        self.onBeacon = onBeacon
    }

    func start() {
        lock.withLock {
            guard !running, let group else { return }
            running = true
            let connection = NWConnectionGroup(with: group, using: .udp)
            self.connection = connection
            connection.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                if case .failed = state { self.lock.withLock { self.running = false } }
                if case .cancelled = state { self.lock.withLock { self.running = false } }
            }
            connection.setReceiveHandler { [weak self] message, data, _ in
                guard let self, let data, let beacon = RemoteBeacon.decode([UInt8](data)) else { return }
                guard let source = Self.sourceHost(of: message) else { return }
                self.onBeacon(beacon, source)
            }
            connection.start(queue: .global(qos: .utility))
        }
    }

    func stop() {
        let pending = lock.withLock { () -> NWConnectionGroup? in
            running = false
            defer { connection = nil }
            return connection
        }
        pending?.cancel()
    }

    /// The sender's address, which is what the client should dial.
    ///
    /// Taken from the datagram's own metadata rather than the payload's name: a
    /// `.local` name can resolve differently on two machines, and a beacon is
    /// unauthenticated, so its self-description is not worth trusting.
    private static func sourceHost(of message: NWConnectionGroup.Message) -> String? {
        guard case .hostPort(let host, _) = message.remoteEndpoint else { return nil }
        return host.debugDescription
    }
}