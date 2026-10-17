import Combine
import Foundation

enum TaggingPanel: String, Sendable, Equatable {
    case rating
    case tags
}

struct TaggingProgress: Sendable, Equatable {
    var isRunning = false
    var origin: TaggingPanel?
    var processed = 0
    var total = 0
    var elapsed: TimeInterval = 0

    var fraction: Double {
        guard total > 0 else { return 0 }
        return min(1, Double(processed) / Double(total))
    }

    var framesPerSecond: Double {
        guard elapsed > 0 else { return 0 }
        return Double(processed) / elapsed
    }
}

/// Only the small progress card observes batch updates. Publishing these on the
/// session would invalidate the sidebar and every visible frame cell as well.
@MainActor
final class TaggingStatus: ObservableObject {
    @Published private(set) var progress = TaggingProgress()

    func update(_ value: TaggingProgress) {
        guard value.isRunning != progress.isRunning || value.origin != progress.origin
                || value.total != progress.total || value.elapsed < progress.elapsed
                || value.elapsed - progress.elapsed >= 0.2 else { return }
        progress = value
    }
}
