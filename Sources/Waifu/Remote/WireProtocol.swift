import Foundation

// MARK: - Opcodes

/// One opcode per protocol message. Connection-level messages use
/// `streamID == 0`; every other request picks a `streamID >= 1` that all of its
/// frames and its response share.
///
/// Raw values are pinned: a frame is rejected before it is understood, so a
/// mismatch here fails loudly at the socket rather than silently misreading a
/// payload.
enum RemoteOpcode: UInt16, Sendable {
    case hello = 0x0001
    case challenge = 0x0002
    case auth = 0x0003
    case authResult = 0x0004
    case capabilities = 0x0005
    case ping = 0x0006
    case pong = 0x0007
    case close = 0x0008
    case error = 0x0009

    case tagBatch = 0x0101
    case tagResult = 0x0102
    case vocabularyRequest = 0x0110
    case vocabularyResponse = 0x0111

    case uncoverBegin = 0x0120
    case uncoverInput = 0x0121
    case uncoverInputEnd = 0x0122
    case uncoverProgress = 0x0123
    case uncoverResultBegin = 0x0124
    case uncoverResult = 0x0125
    case uncoverResultEnd = 0x0126
    case truforRequest = 0x0150
    case truforResult = 0x0151
    case truforData = 0x0152
    case truforEnd = 0x0153
    case safireRequest = 0x0140
    case safireResult = 0x0141
    case cancelJob = 0x0130

    /// True for messages that mean something on their own, with no request.
    var isConnectionLevel: Bool { rawValue < 0x0100 || self == .cancelJob }
}

// MARK: - Flags

struct RemoteFlags: OptionSet, Sendable {
    let rawValue: UInt32
    init(rawValue: UInt32) { self.rawValue = rawValue }
    /// Last frame of this message. A message that fits in one frame is always final.
    static let isFinal = RemoteFlags(rawValue: 1 << 0)
    /// Payload is zlib-compressed. Only ever set on control or vocabulary payloads.
    static let zlib = RemoteFlags(rawValue: 1 << 1)
    /// The message ended in an error rather than a result.
    static let isError = RemoteFlags(rawValue: 1 << 2)
}

// MARK: - Limits

enum RemoteLimits {
    static let protocolVersion: UInt16 = 2
    static let headerBytes = 24
    /// A frame is bounded so a hostile or buggy peer cannot make the reader
    /// allocate without limit. Anything larger is split across frames.
    static let maximumFramePayload = 1 << 20
    /// A whole message is bounded too: a 300-frame Uncover job cannot exceed this.
    static let maximumMessagePayload = 64 << 20
    static let defaultPort: UInt16 = 47410
    static let beaconPort: UInt16 = 42888
    static let beaconGroup = "239.255.42.99"
    /// Tagging keeps eight batches in flight, bounded separately from buffered
    /// bytes. Leave two request slots for vocabulary and other control work.
    static let taggingBatchWindow = 8
    static let maximumOutstandingRequests = taggingBatchWindow + 2
    static let maximumOutstandingJobs = 2
    static let handshakeTimeout: TimeInterval = 10
    static let idleTimeout: TimeInterval = 60
    static let pingInterval: TimeInterval = 15
    static let clientPingInterval: TimeInterval = 5
    static let clientPongTimeout: TimeInterval = 20
}

// MARK: - Errors

enum RemoteError: Error, LocalizedError, Sendable {
    case framing(String)
    case badKey
    case busy
    case modelUnavailable(String)
    case invalidRequest(String)
    case tooLarge
    case cancelled
    case internalFailure(String)

    var errorDescription: String? {
        switch self {
        case .framing(let detail): "Malformed remote message: \(detail)"
        case .badKey: "The server rejected the access key."
        case .busy: "The server is busy with another job."
        case .modelUnavailable(let model): model
        case .invalidRequest(let detail): detail
        case .tooLarge: "That request is larger than the remote protocol allows."
        case .cancelled: "Cancelled"
        case .internalFailure(let detail): detail
        }
    }

    /// The code sent in an `error` or `authResult` frame.
    var code: String {
        switch self {
        case .framing: "framing"
        case .badKey: "badKey"
        case .busy: "busy"
        case .modelUnavailable: "modelUnavailable"
        case .invalidRequest: "invalidRequest"
        case .tooLarge: "tooLarge"
        case .cancelled: "cancelled"
        case .internalFailure: "internal"
        }
    }

    init(code: String, message: String) {
        switch code {
        case "badKey": self = .badKey
        case "busy": self = .busy
        case "modelUnavailable": self = .modelUnavailable(message)
        case "invalidRequest": self = .invalidRequest(message)
        case "tooLarge": self = .tooLarge
        case "cancelled": self = .cancelled
        default: self = .internalFailure(message)
        }
    }
}

// MARK: - Header

/// The 24 bytes that precede every frame.
///
/// Little-endian, which is correct because both supported architectures are
/// little-endian — the argument `WDTaggerWorker.decodeProbabilities` already
/// makes for the tagger's own pipe protocol.
struct RemoteHeader: Equatable, Sendable {
    static let magic: [UInt8] = [0x57, 0x46, 0x52, 0x31] // "WFR1"

    var version: UInt16 = RemoteLimits.protocolVersion
    var opcode: UInt16
    var flags: RemoteFlags
    var streamID: UInt32
    var byteCount: UInt32

    init(opcode: RemoteOpcode, streamID: UInt32 = 0,
         flags: RemoteFlags = [.isFinal], byteCount: Int = 0) {
        self.opcode = opcode.rawValue
        self.streamID = streamID
        self.flags = flags
        self.byteCount = UInt32(clamping: byteCount)
    }

    /// Rebuilds a header from wire values, for the decode path where the opcode
    /// is a raw `UInt16` that may not name a known message.
    init(version: UInt16, opcode: UInt16, flags: RemoteFlags, streamID: UInt32, byteCount: UInt32) {
        self.version = version
        self.opcode = opcode
        self.flags = flags
        self.streamID = streamID
        self.byteCount = byteCount
    }

    var opcodeValue: RemoteOpcode? { RemoteOpcode(rawValue: opcode) }
    var payloadByteCount: Int { Int(byteCount) }

    func encoded() -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: RemoteLimits.headerBytes)
        bytes.replaceSubrange(0..<4, with: Self.magic)
        RemoteHeader.put(version, into: &bytes, at: 4)
        RemoteHeader.put(opcode, into: &bytes, at: 6)
        RemoteHeader.put(flags.rawValue, into: &bytes, at: 8)
        RemoteHeader.put(streamID, into: &bytes, at: 12)
        RemoteHeader.put(byteCount, into: &bytes, at: 16)
        // bytes 20..<24 stay zero: reserved, and a peer that sets them is
        // speaking a dialect this build does not know.
        return bytes
    }

    /// Decodes a header, or throws before anything is allocated.
    static func decode(_ bytes: [UInt8]) throws -> RemoteHeader {
        guard bytes.count == RemoteLimits.headerBytes else {
            throw RemoteError.framing("short header")
        }
        guard bytes[0..<4].elementsEqual(magic) else { throw RemoteError.framing("bad magic") }
        let version = get(UInt16.self, from: bytes, at: 4)
        guard version == RemoteLimits.protocolVersion else {
            throw RemoteError.framing("protocol version \(version) is not supported")
        }
        let opcode = get(UInt16.self, from: bytes, at: 6)
        let flags = RemoteFlags(rawValue: get(UInt32.self, from: bytes, at: 8))
        guard flags.subtracting([.isFinal, .zlib, .isError]).isEmpty else { throw RemoteError.framing("unknown flags") }
        let streamID = get(UInt32.self, from: bytes, at: 12)
        let byteCount = get(UInt32.self, from: bytes, at: 16)
        guard get(UInt32.self, from: bytes, at: 20) == 0 else {
            throw RemoteError.framing("reserved header bytes are not zero")
        }
        guard Int(byteCount) <= RemoteLimits.maximumFramePayload else {
            // Refused on the declared length alone, so a peer cannot make the
            // reader reserve a gigabyte by writing four bytes.
            throw RemoteError.framing("frame payload of \(byteCount) bytes exceeds the limit")
        }
        return RemoteHeader(version: version, opcode: opcode, flags: flags,
                            streamID: streamID, byteCount: byteCount)
    }

    private static func put<T: FixedWidthInteger>(_ value: T, into bytes: inout [UInt8], at offset: Int) {
        withUnsafeBytes(of: value.littleEndian) { bytes.replaceSubrange(offset..<(offset + MemoryLayout<T>.size), with: $0) }
    }
    private static func get<T: FixedWidthInteger>(_ type: T.Type, from bytes: [UInt8], at offset: Int) -> T {
        var value: T = 0
        withUnsafeMutableBytes(of: &value) { source in
            for index in 0..<MemoryLayout<T>.size { source[index] = bytes[offset + index] }
        }
        return T(littleEndian: value)
    }
}

// MARK: - Frame

/// One frame on the wire. `byteCount == 0` is legal: `ping`, `uncoverInputEnd`
/// and `uncoverResultEnd` are markers.
struct RemoteFrame: Sendable {
    var header: RemoteHeader
    var payload: [UInt8]

    var opcode: RemoteOpcode? { header.opcodeValue }
    var streamID: UInt32 { header.streamID }

    init(opcode: RemoteOpcode, streamID: UInt32 = 0,
         flags: RemoteFlags = [.isFinal], payload: [UInt8] = []) {
        header = RemoteHeader(opcode: opcode, streamID: streamID, flags: flags, byteCount: payload.count)
        self.payload = payload
    }

    /// Rebuilds a frame from a header that has already been decoded off the
    /// socket, where the opcode and stream id must be taken from the wire rather
    /// than re-derived.
    init(header: RemoteHeader, payload: [UInt8]) {
        self.header = header
        self.payload = payload
    }

    var data: Data { Data(payload) }

    func encoded() -> [UInt8] { header.encoded() + payload }
}

// MARK: - Splitting and reassembly

enum RemoteMessage {
    /// Splits a payload into frames that each respect `maximumFramePayload`.
    ///
    /// An empty payload is still one frame: `isFinal` alone identifies a
    /// zero-length message, so an empty run is never produced.
    static func frames(opcode: RemoteOpcode, streamID: UInt32, payload: [UInt8],
                       extraFlags: RemoteFlags = []) -> [RemoteFrame] {
        let limit = RemoteLimits.maximumFramePayload
        guard payload.count > limit else {
            return [RemoteFrame(opcode: opcode, streamID: streamID,
                                flags: extraFlags.union(.isFinal), payload: payload)]
        }
        var frames: [RemoteFrame] = []
        var offset = 0
        while offset < payload.count {
            let end = min(payload.count, offset + limit)
            let isLast = end == payload.count
            frames.append(RemoteFrame(opcode: opcode, streamID: streamID,
                                      flags: isLast ? extraFlags.union(.isFinal) : extraFlags,
                                      payload: Array(payload[offset..<end])))
            offset = end
        }
        return frames
    }
}

/// Reassembles the frames of one message and enforces the message ceiling.
///
/// A frame with a mismatched opcode or stream ID is a protocol violation, not a
/// message to be tolerated: the caller tears the connection down rather than
/// guessing which request the bytes belonged to.
/// Reassembles the frames of one message and enforces the message ceiling.
///
/// A `struct`, not an actor: the owning `RemoteLink` already serialises every
/// call, so a second isolation boundary here would only add a suspension point
/// in the middle of parsing a frame.
struct RemoteMessageAssembler {
    private var opcode: RemoteOpcode?
    private var streamID: UInt32 = 0
    private var chunks: [UInt8] = []
    private var received = 0
    private var flags: RemoteFlags = []

    /// Appends one frame. Returns the payload when that frame completed a
    /// message, and `nil` while more frames are expected.
    mutating func append(_ frame: RemoteFrame) throws -> (RemoteOpcode, UInt32, [UInt8])? {
        guard let op = frame.opcode else { throw RemoteError.framing("unknown opcode") }
        if let current = opcode {
            guard current == op else { throw RemoteError.framing("interleaved message opcode") }
            guard flags == frame.header.flags.subtracting(.isFinal) else { throw RemoteError.framing("fragment flags changed") }
            guard self.streamID == frame.streamID else { throw RemoteError.framing("interleaved message stream") }
        } else {
            guard op.isConnectionLevel ? frame.streamID == 0 : frame.streamID >= 1 else {
                throw RemoteError.framing("bad stream id for opcode \(op.rawValue)")
            }
            flags = frame.header.flags.subtracting(.isFinal)
            opcode = op
            streamID = frame.streamID
            chunks = []
            received = 0
        }
        received += frame.payload.count
        guard received <= RemoteLimits.maximumMessagePayload else {
            reset()
            throw RemoteError.framing("message exceeds \(RemoteLimits.maximumMessagePayload) bytes")
        }
        chunks.append(contentsOf: frame.payload)
        guard frame.header.flags.contains(.isFinal) else { return nil }
        let finished = (op, streamID, chunks)
        reset()
        return finished
    }

    private mutating func reset() {
        opcode = nil
        streamID = 0
        chunks = []
        received = 0
    }
}

// MARK: - Binary payload reading and writing

/// Little-endian scalar and byte-run writer. Every append is length-checked by
/// the reader, so a malformed payload is refused rather than trusted.
struct WireWriter {
    private(set) var bytes: [UInt8] = []

    init(capacity: Int = 0) { if capacity > 0 { bytes.reserveCapacity(capacity) } }
    init(_ bytes: [UInt8]) { self.bytes = bytes }

    mutating func append(_ value: [UInt8]) { bytes.append(contentsOf: value) }

    mutating func u8(_ value: UInt8) { bytes.append(value) }
    mutating func u16(_ value: UInt16) { appendFixed(value.littleEndian) }
    mutating func u32(_ value: UInt32) { appendFixed(value.littleEndian) }
    mutating func u64(_ value: UInt64) { appendFixed(value.littleEndian) }
    mutating func f32(_ value: Float) { appendFixed(value.bitPattern.littleEndian) }
    mutating func f64(_ value: Double) { appendFixed(value.bitPattern.littleEndian) }

    /// Length-prefixed UTF-8. Never used for binary: a byte run goes through
    /// `raw`, which is not valid UTF-8 in general and must not be laundered
    /// through a string on its way to the socket.
    mutating func string(_ value: String) {
        let utf8 = Array(value.utf8)
        u16(UInt16(clamping: utf8.count))
        bytes.append(contentsOf: utf8)
    }

    mutating func raw(_ value: [UInt8]) {
        u32(UInt32(clamping: value.count))
        bytes.append(contentsOf: value)
    }

    mutating func data(_ value: Data) { raw([UInt8](value)) }

    /// Raw Float32 without a length prefix, for the fixed-size probability rows.
    mutating func floats(_ values: [Float]) {
        for value in values { appendFixed(value.bitPattern.littleEndian) }
    }

    private mutating func appendFixed<T: FixedWidthInteger>(_ value: T) {
        withUnsafeBytes(of: value) { bytes.append(contentsOf: $0) }
    }
}

/// Bounds-checked reader over one message payload.
///
/// Every accessor throws rather than trapping, and counts are validated against
/// the bytes actually remaining before a `reserveCapacity` or an array literal is
/// built from them.
struct WireReader {
    private let bytes: [UInt8]
    private var offset = 0

    init(_ bytes: [UInt8]) { self.bytes = bytes }
    init(_ data: Data) { self.bytes = [UInt8](data) }

    var remaining: Int { bytes.count - offset }
    var isAtEnd: Bool { offset >= bytes.count }

    /// Everything not yet consumed, as one byte run.
    mutating func remainingBytes() throws -> [UInt8] {
        Array(try take(remaining))
    }

    private mutating func take(_ count: Int) throws -> ArraySlice<UInt8> {
        guard count >= 0, remaining >= count else {
            throw RemoteError.framing("payload ended \(remaining) bytes short of \(count)")
        }
        defer { offset += count }
        return bytes[offset..<(offset + count)]
    }

    mutating func u8() throws -> UInt8 { try UInt8(throwIfEmpty: take(1)) }
    mutating func u16() throws -> UInt16 { try fixed(UInt16.self) }
    mutating func u32() throws -> UInt32 { try fixed(UInt32.self) }
    mutating func u64() throws -> UInt64 { try fixed(UInt64.self) }
    mutating func f32() throws -> Float { Float(bitPattern: try fixed(UInt32.self)) }
    mutating func f64() throws -> Double { Double(bitPattern: try fixed(UInt64.self)) }

    mutating func string() throws -> String {
        let count = Int(try u16())
        let slice = try take(count)
        guard let value = String(bytes: slice, encoding: .utf8) else {
            throw RemoteError.framing("string is not valid UTF-8")
        }
        return value
    }

    /// A length-prefixed byte run, returned as bytes. This is how every blob
    /// travels: the same bytes, unmodified.
    mutating func raw() throws -> [UInt8] {
        let count = Int(try u32())
        guard count <= RemoteLimits.maximumFramePayload else {
            throw RemoteError.framing("byte run of \(count) exceeds the frame limit")
        }
        return Array(try take(count))
    }

    /// A byte run whose declared length must match the rest of the payload.
    mutating func rawRemainder() throws -> [UInt8] {
        let count = Int(try u32())
        guard count == remaining else {
            throw RemoteError.framing("byte run of \(count) does not match \(remaining) remaining")
        }
        return Array(try take(count))
    }

    mutating func floats(_ count: Int) throws -> [Float] {
        let bytes = try raw()
        guard bytes.count == count * 4 else {
            throw RemoteError.framing("expected \(count * 4) bytes of probabilities, found \(bytes.count)")
        }
        return bytes.withUnsafeBytes { source in
            (0..<count).map { index in
                Float(bitPattern: source.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self).littleEndian)
            }
        }
    }

    private mutating func fixed<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        let slice = try take(size)
        var value: T = 0
        withUnsafeMutableBytes(of: &value) { destination in
            for index in 0..<size { destination[index] = slice[slice.startIndex + index] }
        }
        return T(littleEndian: value)
    }
}

private extension FixedWidthInteger {
    init(throwIfEmpty slice: ArraySlice<UInt8>) throws {
        guard let first = slice.first else { throw RemoteError.framing("unexpected end of payload") }
        self.init(first)
    }
}

// MARK: - JSON control payloads

/// The shape of every `error`, `close`, `authResult` and `capabilities` frame.
///
/// One type for all of them keeps the client's decode path to a single switch
/// and means a server-side refactor cannot silently change one message's shape
/// while another's stays stable.
enum RemoteControl: Codable, Sendable {
    case close(reason: String)
    case error(code: String, message: String)
    case authResult(ok: Bool, sessionID: String?, serverProof: Data?, capabilities: RemoteCapabilities?)
    case capabilities(RemoteCapabilities)
    case hello(protocolVersion: Int, appVersion: String, clientName: String, clientNonce: Data)
    case challenge(serverName: String, appVersion: String, serverNonce: Data, keyID: Data)
    case ping(counter: UInt64)
    case pong(counter: UInt64)
    case cancelJob(streamID: UInt32)
    case progress(message: String)
    case uncoverBegin(manifest: UncoverManifest, algorithm: String, allowUnsafeMemory: Bool)
    case uncoverResultBegin(model: String, seconds: Double, outputScale: Int)

    private enum Kind: String, Codable {
        case close, error, authResult, capabilities, hello, challenge, ping, pong
        case cancelJob, progress, uncoverBegin, uncoverResultBegin
    }

    private enum CodingKeys: String, CodingKey {
        case kind, reason, code, message, ok, sessionID, serverProof
        case protocolVersion, appVersion, clientName, clientNonce
        case serverName, serverNonce, keyID, counter, streamID
        case manifest, algorithm, allowUnsafeMemory, model, seconds, outputScale
        case capabilities
    }

    private struct Box: Codable {
        var kind: Kind
        var reason: String?
        var code: String?
        var message: String?
        var ok: Bool?
        var sessionID: String?
        var serverProof: Data?
        var protocolVersion: Int?
        var appVersion: String?
        var clientName: String?
        var clientNonce: Data?
        var serverName: String?
        var serverNonce: Data?
        var keyID: Data?
        var counter: UInt64?
        var streamID: UInt32?
        var manifest: UncoverManifest?
        var algorithm: String?
        var allowUnsafeMemory: Bool?
        var model: String?
        var seconds: Double?
        var outputScale: Int?
        var capabilities: RemoteCapabilities?
    }

    func encode(to encoder: Encoder) throws {
        var box = Box(kind: .close)
        switch self {
        case .close(let reason):
            box.kind = .close; box.reason = reason
        case .error(let code, let message):
            box.kind = .error; box.code = code; box.message = message
        case .authResult(let ok, let sessionID, let proof, let capabilities):
            box.kind = .authResult; box.ok = ok; box.sessionID = sessionID
            box.serverProof = proof; box.capabilities = capabilities
        case .capabilities(let capabilities):
            box.kind = .capabilities; box.capabilities = capabilities
        case .hello(let version, let appVersion, let clientName, let nonce):
            box.kind = .hello; box.protocolVersion = version; box.appVersion = appVersion
            box.clientName = clientName; box.clientNonce = nonce
        case .challenge(let serverName, let appVersion, let nonce, let keyID):
            box.kind = .challenge; box.serverName = serverName; box.appVersion = appVersion
            box.serverNonce = nonce; box.keyID = keyID
        case .ping(let counter):
            box.kind = .ping; box.counter = counter
        case .pong(let counter):
            box.kind = .pong; box.counter = counter
        case .cancelJob(let streamID):
            box.kind = .cancelJob; box.streamID = streamID
        case .progress(let message):
            box.kind = .progress; box.message = message
        case .uncoverBegin(let manifest, let algorithm, let allowUnsafeMemory):
            box.kind = .uncoverBegin; box.manifest = manifest; box.algorithm = algorithm
            box.allowUnsafeMemory = allowUnsafeMemory
        case .uncoverResultBegin(let model, let seconds, let outputScale):
            box.kind = .uncoverResultBegin; box.model = model; box.seconds = seconds
            box.outputScale = outputScale
        }
        try box.encode(to: encoder)
    }

    init(from decoder: Decoder) throws {
        let box = try Box(from: decoder)
        func required<T: Decodable>(_ key: KeyPath<Box, T?>, _ name: String) throws -> T {
            guard let value = box[keyPath: key] else {
                throw RemoteError.framing("control message is missing \(name)")
            }
            return value
        }
        switch box.kind {
        case .close:
            self = .close(reason: try required(\.reason, "reason"))
        case .error:
            self = .error(code: try required(\.code, "code"), message: box.message ?? "")
        case .authResult:
            self = .authResult(ok: box.ok ?? false, sessionID: box.sessionID,
                               serverProof: box.serverProof, capabilities: box.capabilities)
        case .capabilities:
            self = .capabilities(try required(\.capabilities, "capabilities"))
        case .hello:
            self = .hello(protocolVersion: try required(\.protocolVersion, "protocolVersion"),
                          appVersion: box.appVersion ?? "", clientName: box.clientName ?? "",
                          clientNonce: try required(\.clientNonce, "clientNonce"))
        case .challenge:
            self = .challenge(serverName: box.serverName ?? "", appVersion: box.appVersion ?? "",
                              serverNonce: try required(\.serverNonce, "serverNonce"),
                              keyID: box.keyID ?? Data())
        case .ping:
            self = .ping(counter: box.counter ?? 0)
        case .pong:
            self = .pong(counter: box.counter ?? 0)
        case .cancelJob:
            self = .cancelJob(streamID: box.streamID ?? 0)
        case .progress:
            self = .progress(message: box.message ?? "")
        case .uncoverBegin:
            self = .uncoverBegin(manifest: try required(\.manifest, "manifest"),
                                 algorithm: try required(\.algorithm, "algorithm"),
                                 allowUnsafeMemory: box.allowUnsafeMemory ?? false)
        case .uncoverResultBegin:
            self = .uncoverResultBegin(model: box.model ?? "", seconds: box.seconds ?? 0,
                                       outputScale: box.outputScale ?? 1)
        }
    }

    /// Encodes to the bytes of one frame payload.
    static func encode(_ value: RemoteControl) throws -> [UInt8] {
        [UInt8](try JSONEncoder().encode(value))
    }

    static func decode(_ bytes: [UInt8]) throws -> RemoteControl {
        try JSONDecoder().decode(RemoteControl.self, from: Data(bytes))
    }
}
