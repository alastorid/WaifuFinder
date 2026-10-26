import Foundation
import Network
import Darwin

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
        // Bits 3...7 are reserved for reconstruction algorithms.
        static let safireReady = BeaconFlags(rawValue: 1 << 8)
        static let truforReady = BeaconFlags(rawValue: 1 << 9)
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
    var offersSafire: Bool
    var offersTruFor: Bool
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
            keyID: beacon.keyID,
            offersSafire: beacon.flags.contains(.safireReady),
            offersTruFor: beacon.flags.contains(.truforReady), offersTagging: beacon.flags.contains(.taggerReady),
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
/// A UDP socket receives subnet broadcasts on the beacon port and joins the
/// multicast group on every active IPv4 interface.
final class RemoteBeaconReceiver: @unchecked Sendable {
    private let port: UInt16
    private let multicastGroup: String
    private let onBeacon: @Sendable (RemoteBeacon, String) -> Void
    private let lock = NSLock()
    private var running = false
    private var descriptor: Int32 = -1
    private var readSource: DispatchSourceRead?
    private let queue = DispatchQueue(label: "waifu.remote.beacon.receive", qos: .utility)

    init(port: UInt16 = RemoteLimits.beaconPort,
         multicastGroup: String = RemoteLimits.beaconGroup,
         onBeacon: @escaping @Sendable (RemoteBeacon, String) -> Void) {
        self.port = port
        self.multicastGroup = multicastGroup
        self.onBeacon = onBeacon
    }

    func start() {
        let socketFD = lock.withLock { () -> Int32 in
            guard !running else { return -1 }
            let fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
            guard fd >= 0 else { return -1 }

            var reuse: Int32 = 1
            guard setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse,
                             socklen_t(MemoryLayout<Int32>.size)) == 0 else {
                Darwin.close(fd)
                return -1
            }
            let flags = fcntl(fd, F_GETFL, 0)
            guard flags >= 0, fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 else {
                Darwin.close(fd)
                return -1
            }

            var local = sockaddr_in()
            local.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            local.sin_family = sa_family_t(AF_INET)
            local.sin_port = in_port_t(port).bigEndian
            local.sin_addr = in_addr(s_addr: in_addr_t(0))
            let bindStatus = withUnsafePointer(to: &local) { address in
                address.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindStatus == 0 else {
                Darwin.close(fd)
                return -1
            }

            joinMulticast(on: fd)
            descriptor = fd
            running = true
            return fd
        }

        guard socketFD >= 0 else { return }
        let source = DispatchSource.makeReadSource(fileDescriptor: socketFD, queue: queue)
        source.setEventHandler { [weak self] in self?.receiveAvailable(from: socketFD) }
        source.setCancelHandler { Darwin.close(socketFD) }
        lock.withLock { readSource = source }
        source.resume()
    }

    func stop() {
        let pending = lock.withLock { () -> DispatchSourceRead? in
            running = false
            descriptor = -1
            defer { readSource = nil }
            return readSource
        }
        pending?.cancel()
    }

    private func joinMulticast(on fd: Int32) {
        var groupAddress = in_addr()
        guard multicastGroup.withCString({ inet_pton(AF_INET, $0, &groupAddress) }) == 1 else { return }

        // Join on every active IPv4 interface. This covers Wi-Fi and Ethernet
        // when both are up, and leaves broadcast reception on the same socket.
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let first = interfaces else { return }
        defer { freeifaddrs(first) }
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        var joinedAny = false
        while let pointer = cursor {
            let entry = pointer.pointee
            cursor = entry.ifa_next
            guard Int32(entry.ifa_flags) & IFF_UP != 0,
                  Int32(entry.ifa_flags) & IFF_LOOPBACK == 0,
                  let address = entry.ifa_addr, address.pointee.sa_family == UInt8(AF_INET) else { continue }
            var membership = ip_mreq()
            membership.imr_multiaddr = groupAddress
            membership.imr_interface = address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee.sin_addr }
            if setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &membership,
                          socklen_t(MemoryLayout<ip_mreq>.size)) == 0 {
                joinedAny = true
            }
        }
        if !joinedAny {
            var membership = ip_mreq()
            membership.imr_multiaddr = groupAddress
            membership.imr_interface = in_addr(s_addr: in_addr_t(0))
            _ = setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &membership,
                           socklen_t(MemoryLayout<ip_mreq>.size))
        }
    }

    private func receiveAvailable(from fd: Int32) {
        while lock.withLock({ running && descriptor == fd }) {
            var bytes = [UInt8](repeating: 0, count: 512)
            var sender = sockaddr_in()
            var senderLength = socklen_t(MemoryLayout<sockaddr_in>.size)
            let count = bytes.withUnsafeMutableBytes { buffer in
                withUnsafeMutablePointer(to: &sender) { address in
                    address.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                        recvfrom(fd, buffer.baseAddress, buffer.count, 0, socketAddress, &senderLength)
                    }
                }
            }
            guard count >= 0 else {
                if errno == EAGAIN || errno == EWOULDBLOCK { return }
                return
            }
            let data = Array(bytes.prefix(count))
            guard let beacon = RemoteBeacon.decode(data), let source = Self.numericHost(sender) else { continue }
            onBeacon(beacon, source)
        }
    }

    private static func numericHost(_ address: sockaddr_in) -> String? {
        var address = address
        var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getnameinfo($0, socklen_t(MemoryLayout<sockaddr_in>.size), &host,
                            socklen_t(host.count), nil, 0, NI_NUMERICHOST)
            }
        }
        guard result == 0 else { return nil }
        return String(decoding: host.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}
