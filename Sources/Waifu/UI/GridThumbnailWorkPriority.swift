import Combine
import CoreGraphics

/// Foreground demand suspends speculative reads through their decode step.
/// Prefetchers resume as soon as all visible requests have finished.
@MainActor
final class GridThumbnailWorkPriority {
    static let shared = GridThumbnailWorkPriority()
    @Published private(set) var visibleWorkCount = 0

    func withVisibleWork(_ load: () async -> CGImage?) async -> CGImage? {
        guard !Task.isCancelled else { return nil }
        visibleWorkCount += 1
        defer { visibleWorkCount -= 1 }
        return await load()
    }
}
