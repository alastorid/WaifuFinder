import Combine
import Foundation

/// Independent authenticated connections; the primary remains the Uncover connection.
@MainActor
final class RemoteInferencePool: ObservableObject {
    struct Entry: Identifiable {
        let id: String
        let client: RemoteClient
    }

    static let shared = RemoteInferencePool()
    @Published private(set) var entries: [Entry]
    @Published private(set) var remoteVocabulary: TagVocabulary?
    private let discoveryEnabled: Bool
    private let defaults: UserDefaults
    private var observations: [AnyCancellable] = []

    init(defaults: UserDefaults = .standard, primary: RemoteClient = .shared, discoveryEnabled: Bool = true) {
        self.defaults = defaults
        self.discoveryEnabled = discoveryEnabled
        entries = [Entry(id: "primary", client: primary)]
        for id in defaults.stringArray(forKey: "remoteInferenceServerIDs") ?? [] {
            entries.append(Entry(id: id, client: RemoteClient(defaults: defaults, discoveryEnabled: discoveryEnabled, configurationID: id)))
        }
        // An unfinished setup form must not become a configured card after relaunch.
        if let pending = defaults.string(forKey: "remoteInferencePendingAddition") {
            remove(pending)
        }
        observeClients()
        for entry in entries { entry.client.restoreEnabled() }
        refresh()
    }

    func add() {
        let id = UUID().uuidString
        entries.append(Entry(id: id, client: RemoteClient(defaults: defaults, discoveryEnabled: discoveryEnabled, configurationID: id)))
        save()
        observeClients()
    }

    /// Reuse the primary slot after removal so Uncover keeps its shared client.
    func beginAdding() -> Entry {
        let entry: Entry
        if let empty = entries.first(where: { $0.client.manualAddress.isEmpty }) {
            entry = empty
        } else {
            add()
            entry = entries[entries.count - 1]
        }
        defaults.set(entry.id, forKey: "remoteInferencePendingAddition")
        return entry
    }

    func finishAdding(_ id: String) {
        if defaults.string(forKey: "remoteInferencePendingAddition") == id {
            defaults.removeObject(forKey: "remoteInferencePendingAddition")
        }
    }

    func remove(_ id: String) {
        guard let entry = entries.first(where: { $0.id == id }) else { return }
        entry.client.clearConfiguration()
        finishAdding(id)
        if id != "primary" { entries.removeAll { $0.id == id } }
        defaults.removeObject(forKey: "remoteClientAddress.\(id)")
        defaults.removeObject(forKey: "remoteClientEnabled.\(id)")
        save()
        observeClients()
        refresh()
    }

    private func save() {
        defaults.set(entries.filter { $0.id != "primary" }.map(\.id), forKey: "remoteInferenceServerIDs")
    }

    private func observeClients() {
        observations = entries.map { entry in
            entry.client.objectWillChange.sink { [weak self] _ in
                Task { @MainActor [weak self] in self?.refresh() }
            }
        }
    }

    private func refresh() {
        let vocabulary = taggingClients.first?.remoteVocabulary
        if remoteVocabulary?.cacheFingerprint != vocabulary?.cacheFingerprint { remoteVocabulary = vocabulary }
        objectWillChange.send()
    }

    /// One lane per endpoint, even if it was entered in more than one settings row.
    var taggingClients: [RemoteClient] {
        var seen = Set<String>()
        return entries.compactMap { entry in
            let client = entry.client
            guard client.offersRemoteTagging, client.remoteVocabulary != nil,
                  seen.insert(RemoteAddress.parse(client.manualAddress)?.display ?? client.manualAddress).inserted else { return nil }
            return client
        }
    }

    func taggingClients(matching fingerprint: UInt64?) -> [RemoteClient] {
        taggingClients.filter { $0.remoteVocabulary?.cacheFingerprint == fingerprint }
    }

    /// Forensic inference uses one lane per connected endpoint, like tagging.
    private func forensicClients(where offers: (RemoteClient) -> Bool) -> [RemoteClient] {
        var seen = Set<String>()
        return entries.map(\.client).filter { client in
            offers(client) && seen.insert(RemoteAddress.parse(client.manualAddress)?.display ?? client.manualAddress).inserted
        }
    }

    var safireClients: [RemoteClient] { forensicClients { $0.offersSafire } }
    var truforClients: [RemoteClient] { forensicClients { $0.offersTruFor } }

    var offersRemoteTagging: Bool { !taggingClients.isEmpty }
    var serverNames: String { taggingClients.compactMap { $0.capabilities?.serverName }.joined(separator: ", ") }

    func loadVocabularyIfNeeded() async -> TagVocabulary? {
        for entry in entries where entry.client.offersRemoteTagging {
            await entry.client.loadVocabularyIfNeeded()
        }
        refresh()
        return remoteVocabulary
    }
}
