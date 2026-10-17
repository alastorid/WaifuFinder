import CryptoKit
import Foundation
import Security

/// Storage for the server's pre-shared key, behind a protocol so the tests can
/// exercise the whole flow without a Keychain prompt.
protocol RemoteKeyStore: Sendable {
    /// The stored key, or `nil` when none has been generated yet.
    func loadKey() -> [UInt8]?
    /// Stores a new key, replacing any previous one.
    func saveKey(_ key: [UInt8]) throws
    func deleteKey() throws
}

/// Keychain-backed store.
///
/// A secret belongs in the system store rather than a file next to the settings
/// the app already writes, and `Security` is an Apple framework, so this adds no
/// dependency of any licence. Nothing else in WaifuFinder keeps a secret today,
/// so this is the only Keychain use in the app.
struct RemoteKeychainStore: RemoteKeyStore {
    private let service: String
    private let account: String

    init(service: String = "com.alastorid.waifufinder", account: String = "remote-server-key") {
        self.service = service
        self.account = account
    }

    private var baseQuery: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }

    func loadKey() -> [UInt8]? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data, data.count == RemoteCrypto.keyByteCount else { return nil }
        return [UInt8](data)
    }

    func deleteKey() throws {
        let status = SecItemDelete(baseQuery as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw RemoteError.internalFailure("Could not remove the stored access key.")
        }
    }

    func saveKey(_ key: [UInt8]) throws {
        guard key.count == RemoteCrypto.keyByteCount else { throw RemoteError.badKey }
        let data = Data(key)
        // Delete first: `SecItemUpdate` fails for an account that does not exist
        // yet, and "generate on first start" has to work on a fresh install.
        SecItemDelete(baseQuery as CFDictionary)
        var attributes = baseQuery
        attributes[kSecValueData as String] = data
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        let status = SecItemAdd(attributes as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw RemoteError.internalFailure("Could not store the access key (Keychain status \(status)).")
        }
    }
}

/// `0600` file store.
///
/// The fallback for a machine where the Keychain is unavailable, and the
/// implementation the tests use, since a test binary has no business writing to
/// the user's login keychain.
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

/// The store the app uses: Keychain, falling back to a `0600` file only if the
/// Keychain refuses.
struct RemoteDefaultKeyStore: RemoteKeyStore {
    private let keychain: RemoteKeychainStore
    private let fallback: RemoteFileKeyStore

    init(clientAddress: String? = nil) {
        let suffix = clientAddress.map { address in
            SHA256.hash(data: Data(address.utf8)).map { String(format: "%02x", $0) }.joined()
        }
        keychain = RemoteKeychainStore(account: suffix.map { "remote-client-" + $0 } ?? "remote-server-key")
        let url = suffix.map { Self.fallbackURL.deletingLastPathComponent().appendingPathComponent("client-\($0).key") }
            ?? Self.fallbackURL
        fallback = RemoteFileKeyStore(url: url)
    }

    static var fallbackURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/Remote/server.key")
    }

    func loadKey() -> [UInt8]? { keychain.loadKey() ?? fallback.loadKey() }

    func deleteKey() throws {
        try keychain.deleteKey()
        try fallback.deleteKey()
    }

    func saveKey(_ key: [UInt8]) throws {
        do {
            try keychain.saveKey(key)
            try fallback.deleteKey()
        } catch {
            // The file store is a real fallback rather than a second code path
            // to reason about: it is the same interface, so the rest of the app
            // never learns which one answered.
            try fallback.saveKey(key)
        }
    }
}