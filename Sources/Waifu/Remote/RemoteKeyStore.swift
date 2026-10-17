import CryptoKit
import Foundation

/// Storage for the server's pre-shared key, behind a protocol so the tests can
/// exercise the whole flow without touching the user's key store.
protocol RemoteKeyStore: Sendable {
    /// The stored key, or `nil` when none has been generated yet.
    func loadKey() -> [UInt8]?
    /// Stores a new key, replacing any previous one.
    func saveKey(_ key: [UInt8]) throws
    func deleteKey() throws
}

/// A `0600` file store for remote access keys.
final class RemoteFileKeyStore: RemoteKeyStore, @unchecked Sendable {
    private let url: URL
    private let lock = NSLock()

    init(url: URL) { self.url = url }

    func loadKey() -> [UInt8]? {
        lock.withLock {
            guard let data = try? Data(contentsOf: url), data.count == RemoteCrypto.keyByteCount else {
                return nil
            }
            return [UInt8](data)
        }
    }

    func deleteKey() throws {
        try lock.withLock {
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        }
    }

    func saveKey(_ key: [UInt8]) throws {
        guard key.count == RemoteCrypto.keyByteCount else { throw RemoteError.badKey }
        try lock.withLock {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try Data(key).write(to: url, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }
}

/// The store the app uses for both server and client access keys.
struct RemoteDefaultKeyStore: RemoteKeyStore {
    private let fileStore: RemoteFileKeyStore

    init(clientAddress: String? = nil) {
        let suffix = clientAddress.map { address in
            SHA256.hash(data: Data(address.utf8)).map { String(format: "%02x", $0) }.joined()
        }
        let url = suffix.map { Self.storageDirectory.appendingPathComponent("client-\($0).key") }
            ?? Self.serverKeyURL
        fileStore = RemoteFileKeyStore(url: url)
    }

    static var storageDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/Remote", isDirectory: true)
    }

    static var serverKeyURL: URL { storageDirectory.appendingPathComponent("server.key") }

    func loadKey() -> [UInt8]? { fileStore.loadKey() }
    func deleteKey() throws { try fileStore.deleteKey() }
    func saveKey(_ key: [UInt8]) throws { try fileStore.saveKey(key) }
}
