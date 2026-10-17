import Foundation
import Network

/// Broadcasts a beacon every second while the service is running.
///
/// Multicast is primary because it does not depend on the subnet mask; the
/// limited broadcast goes out alongside it for networks that filter multicast.
/// A client only needs one of the two, so one redundant datagram per second is a
/// fair price for working on both kinds of network.
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

/// The sending half: one `NWConnectionGroup` for multicast and one datagram
/// connection for the limited broadcast.
private final class RemoteBeaconConnection: @unchecked Sendable {
    private let multicast: NWConnectionGroup?
    private let broadcast: NWConnection?
    private let queue = DispatchQueue(label: "waifu.remote.beacon.send", qos: .utility)

    init(multicastGroup: String, port: UInt16) {
        let descriptor = try? NWMulticastGroup(for: [RemoteAddress.Endpoint.multicast(group: multicastGroup, port: port)])
        multicast = descriptor.flatMap { NWConnectionGroup(with: $0, using: .udp) }
        broadcast = NWConnection(to: RemoteAddress.Endpoint.broadcast(port: port), using: .udp)
        multicast?.start(queue: queue)
        broadcast?.start(queue: queue)
    }

    func send(_ beacon: RemoteBeacon) {
        let payload = Data(beacon.encoded())
        // The multicast group and the broadcast socket each take the payload
        // whole: a beacon is one datagram or it is not received at all.
        multicast?.send(content: payload, completion: { _ in })
        // Best effort by design: a network that refuses broadcast still gets
        // multicast, and a failure here is not worth surfacing to anyone.
        broadcast?.send(content: payload, contentContext: .defaultMessage, isComplete: true,
                        completion: .contentProcessed { _ in })
    }

    func cancel() {
        multicast?.cancel()
        broadcast?.cancel()
    }
}