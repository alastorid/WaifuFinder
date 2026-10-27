import Foundation
import Network
import Darwin
import os

/// Broadcasts a beacon every second while the service is running.
///
/// Multicast is primary because it does not depend on the subnet mask; a
/// directed IPv4 broadcast also goes out on each active broadcast interface for
/// networks that filter multicast.
final class RemoteBeaconSender: @unchecked Sendable {
    private let port: UInt16
    private let multicastGroup: String
    private let queue = DispatchQueue(label: "waifu.remote.beacon", qos: .utility)
    private let lock = NSLock()
    private var connection: RemoteBeaconConnection?
    private var timer: DispatchSourceTimer?

    init(port: UInt16 = RemoteLimits.beaconPort, multicastGroup: String = RemoteLimits.beaconGroup) {
        self.port = port
        self.multicastGroup = multicastGroup
    }

    func start(beacon: @escaping @Sendable () async -> RemoteBeacon?) {
        queue.sync {
            let pending = lock.withLock { () -> RemoteBeaconConnection? in
                guard connection == nil else { return nil }
                let connection = RemoteBeaconConnection(multicastGroup: multicastGroup, port: port)
                self.connection = connection
                let timer = DispatchSource.makeTimerSource(queue: queue)
                timer.schedule(deadline: .now(), repeating: 1.0, leeway: .milliseconds(100))
                timer.setEventHandler {
                    // One beat per second. The closure is async because building
                    // the beacon reads the model library, which is off the main
                    // actor; a blocking read here would stall the send queue.
                    Task {
                        guard let current = await beacon() else { return }
                        connection.send(current)
                    }
                }
                timer.resume()
                self.timer = timer
                return connection
            }
            _ = pending
        }
    }

    func stop() {
        queue.sync {
            let pending = lock.withLock { () -> (RemoteBeaconConnection, DispatchSourceTimer)? in
                defer { connection = nil; timer = nil }
                guard let connection, let timer else { return nil }
                return (connection, timer)
            }
            pending?.1.cancel()
            pending?.0.cancel()
        }
    }
}

/// The sending half: one `NWConnectionGroup` for multicast and a BSD UDP socket
/// for subnet-directed broadcast.
private final class RemoteBeaconConnection: @unchecked Sendable {
    private let port: UInt16
    private let multicast: NWConnectionGroup?
    private let broadcast = RemoteBeaconBroadcaster()
    private let queue = DispatchQueue(label: "waifu.remote.beacon.send", qos: .utility)

    init(multicastGroup: String, port: UInt16) {
        self.port = port
        let descriptor = try? NWMulticastGroup(for: [RemoteAddress.Endpoint.multicast(group: multicastGroup, port: port)])
        multicast = descriptor.flatMap { NWConnectionGroup(with: $0, using: .udp) }
        multicast?.start(queue: queue)
    }

    func send(_ beacon: RemoteBeacon) {
        let payload = Data(beacon.encoded())
        // The multicast group and the broadcast socket each take the payload
        // whole: a beacon is one datagram or it is not received at all.
        multicast?.send(content: payload, completion: { _ in })
        // Best effort by design: a network that refuses broadcast still gets
        // multicast, and a failure here is not worth surfacing to anyone.
        broadcast.send(payload, port: port)
    }

    func cancel() {
        multicast?.cancel()
        broadcast.cancel()
    }
}

/// UDP broadcast is not supported by Network.framework. Send the beacon to each
/// active IPv4 interface's broadcast address with SO_BROADCAST enabled.
/// The recipient reads the packet source address and combines it with tcpPort.
private final class RemoteBeaconBroadcaster: @unchecked Sendable {
    private let lock = NSLock()
    private let logger = Logger(subsystem: "com.alastorid.waifufinder", category: "RemoteBeacon")
    private var descriptor: Int32 = -1
    private var lastSendErrors: [String: Int32] = [:]

    init() {
        descriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
        guard descriptor >= 0 else {
            let error = errno
            logger.error("Could not create LAN beacon socket (errno: \(error, privacy: .public): \(String(cString: strerror(error)), privacy: .public))")
            return
        }
        var enabled: Int32 = 1
        guard setsockopt(descriptor, SOL_SOCKET, SO_BROADCAST, &enabled,
                         socklen_t(MemoryLayout<Int32>.size)) == 0 else {
            let error = errno
            Darwin.close(descriptor)
            descriptor = -1
            logger.error("Could not enable LAN beacon broadcast (errno: \(error, privacy: .public): \(String(cString: strerror(error)), privacy: .public))")
            return
        }
    }

    func send(_ payload: Data, port: UInt16) {
        lock.withLock {
            guard descriptor >= 0 else { return }
            var interfaces: UnsafeMutablePointer<ifaddrs>?
            guard getifaddrs(&interfaces) == 0, let first = interfaces else { return }
            defer { freeifaddrs(first) }

            var cursor: UnsafeMutablePointer<ifaddrs>? = first
            while let pointer = cursor {
                let entry = pointer.pointee
                cursor = entry.ifa_next
                guard Int32(entry.ifa_flags) & IFF_UP != 0,
                      Int32(entry.ifa_flags) & IFF_RUNNING != 0,
                      Int32(entry.ifa_flags) & IFF_LOOPBACK == 0,
                      Int32(entry.ifa_flags) & IFF_BROADCAST != 0,
                      let local = entry.ifa_addr, local.pointee.sa_family == UInt8(AF_INET),
                      let broadcast = entry.ifa_dstaddr,
                      broadcast.pointee.sa_family == UInt8(AF_INET) else { continue }

                var destination = broadcast.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee }
                destination.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
                destination.sin_family = sa_family_t(AF_INET)
                destination.sin_port = in_port_t(port).bigEndian
                let interfaceName = String(cString: entry.ifa_name)
                var interfaceIndex = Int32(if_nametoindex(entry.ifa_name))
                var sendError: Int32 = 0

                // A directed broadcast address usually selects the right route,
                // but VPN and multi-homed route tables can send it elsewhere.
                // Pin each datagram to the interface that supplied its address.
                var boundInterface: Int32 = 0
                _ = setsockopt(descriptor, IPPROTO_IP, IP_BOUND_IF, &boundInterface,
                               socklen_t(MemoryLayout<Int32>.size))
                let bindingSucceeded = interfaceIndex != 0 && withUnsafePointer(to: &interfaceIndex) {
                    setsockopt(descriptor, IPPROTO_IP, IP_BOUND_IF, $0,
                               socklen_t(MemoryLayout<Int32>.size)) == 0
                }

                let sent = withUnsafePointer(to: &destination) { address in
                    address.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                        payload.withUnsafeBytes { bytes in
                            sendto(descriptor, bytes.baseAddress, payload.count, 0,
                                   socketAddress, socklen_t(MemoryLayout<sockaddr_in>.size))
                        }
                    }
                }
                if sent != payload.count { sendError = sent < 0 ? errno : EIO }

                // Do not leave one interface bound for the next iteration.
                boundInterface = 0
                _ = setsockopt(descriptor, IPPROTO_IP, IP_BOUND_IF, &boundInterface,
                               socklen_t(MemoryLayout<Int32>.size))

                if sendError != 0 {
                    if lastSendErrors[interfaceName] != sendError {
                        logger.error("LAN beacon send failed on \(interfaceName, privacy: .public) (interface bound: \(bindingSucceeded, privacy: .public), errno: \(sendError, privacy: .public): \(String(cString: strerror(sendError)), privacy: .public))")
                    }
                    lastSendErrors[interfaceName] = sendError
                } else if lastSendErrors.removeValue(forKey: interfaceName) != nil {
                    logger.info("LAN beacon sending recovered on \(interfaceName, privacy: .public)")
                }
            }
        }
    }

    func cancel() {
        lock.withLock {
            guard descriptor >= 0 else { return }
            Darwin.close(descriptor)
            descriptor = -1
        }
    }

    deinit { cancel() }
}
