import Combine
import Foundation

enum PhotosOrderFilter: String, CaseIterable, Hashable, Identifiable, Sendable {
    case all
    case ordered
    case unordered

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: "All"
        case .ordered: "Ordered"
        case .unordered: "Unordered"
        }
    }

    var next: Self {
        switch self {
        case .all: .ordered
        case .ordered: .unordered
        case .unordered: .all
        }
    }

    func includes(isOrdered: Bool) -> Bool {
        switch self {
        case .all: true
        case .ordered: isOrdered
        case .unordered: !isOrdered
        }
    }
}

/// Local-only labels used to remember which Photos items the user has synced
/// to Photos successfully. The markers themselves never modify the library.
@MainActor
final class PhotosOrderState: ObservableObject {
    static let shared = PhotosOrderState()

    @Published private var orderedByAlbum: [String: Set<String>]

    // Earlier versions marked items without writing to Photos. Those markers
    // must not be presented as successfully synced library changes.
    private let defaultsKey = "photosSyncedAssetIDsByAlbum.v2"

    private init() {
        if let data = UserDefaults.standard.data(forKey: defaultsKey),
           let stored = try? JSONDecoder().decode([String: [String]].self, from: data) {
            orderedByAlbum = stored.mapValues(Set.init)
        } else {
            orderedByAlbum = [:]
        }
    }

    func isOrdered(albumIdentifier: String, assetIdentifier: String) -> Bool {
        orderedByAlbum[albumIdentifier]?.contains(assetIdentifier) == true
    }

    func markOrdered(albumIdentifier: String, assetIdentifiers: [String]) {
        guard !assetIdentifiers.isEmpty else { return }
        var updated = orderedByAlbum[albumIdentifier, default: []]
        let previousCount = updated.count
        updated.formUnion(assetIdentifiers)
        guard updated.count != previousCount else { return }
        orderedByAlbum[albumIdentifier] = updated
        persist()
    }

    func reset(albumIdentifier: String) {
        var updated = orderedByAlbum
        guard updated.removeValue(forKey: albumIdentifier) != nil else { return }
        orderedByAlbum = updated
        persist()
    }

    private func persist() {
        let stored = orderedByAlbum.mapValues { $0.sorted() }
        guard let data = try? JSONEncoder().encode(stored) else { return }
        UserDefaults.standard.set(data, forKey: defaultsKey)
    }
}
