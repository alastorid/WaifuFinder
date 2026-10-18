import CoreGraphics
import SwiftUI

/// Publishes a cell only after its pixels are decoded. Identity and request
/// tickets prevent a late result from painting an image from an earlier album.
@MainActor
final class GridThumbnailPresentation: ObservableObject {
    struct ReadyImage {
        let identity: String
        let image: CGImage
    }

    @Published private(set) var ready: ReadyImage?
    private let loader: GridThumbnailLoader
    private let owner = UUID()
    private let releaseDelay: Duration
    private var pinnedIdentity: String?
    private var activeIdentity: String?
    private var revision: UInt64 = 0
    private var releaseTask: Task<Void, Never>?

    init(identity: String, loader: GridThumbnailLoader = .shared,
         releaseDelay: Duration = .milliseconds(250)) {
        self.loader = loader
        self.releaseDelay = releaseDelay
        if let pixels = loader.cachedImage(identity: identity) { publish(pixels, identity: identity) }
    }

    deinit {
        releaseTask?.cancel()
        if let pinnedIdentity { loader.releaseVisibleImage(identity: pinnedIdentity, owner: owner) }
    }

    func image(for identity: String) -> CGImage? {
        guard ready?.identity == identity else { return nil }
        return ready?.image
    }

    func prepare(identity: String, load: () async -> CGImage?) async {
        guard !Task.isCancelled else { return }
        releaseTask?.cancel()
        releaseTask = nil
        revision &+= 1
        let ticket = revision
        activeIdentity = identity
        if ready?.identity == identity { return }
        clear()
        if let pixels = loader.cachedImage(identity: identity) {
            publish(pixels, identity: identity)
            return
        }
        guard let pixels = await load(), !Task.isCancelled,
              revision == ticket, activeIdentity == identity else { return }
        publish(pixels, identity: identity)
    }

    func disappear(identity: String) {
        guard activeIdentity == identity || (activeIdentity == nil && ready?.identity == identity) else { return }
        activeIdentity = nil
        revision &+= 1
        let ticket = revision
        releaseTask?.cancel()
        releaseTask = Task { [weak self, releaseDelay] in
            do { try await Task.sleep(for: releaseDelay) }
            catch { return }
            guard let self, self.activeIdentity == nil, self.revision == ticket else { return }
            self.clear()
            self.releaseTask = nil
        }
    }

    private func publish(_ pixels: CGImage, identity: String) {
        if pinnedIdentity != identity {
            if let pinnedIdentity { loader.releaseVisibleImage(identity: pinnedIdentity, owner: owner) }
            loader.retainVisibleImage(pixels, identity: identity, owner: owner)
            pinnedIdentity = identity
        }
        let transaction = Transaction(animation: nil)
        // Publication never animates layout. The native image layer starts its
        // explicit reveal when these pixels actually enter the viewport.
        withTransaction(transaction) {
            ready = ReadyImage(identity: identity, image: pixels)
        }
    }

    private func clear() {
        if let pinnedIdentity { loader.releaseVisibleImage(identity: pinnedIdentity, owner: owner) }
        pinnedIdentity = nil
        if ready != nil { ready = nil }
    }
}
