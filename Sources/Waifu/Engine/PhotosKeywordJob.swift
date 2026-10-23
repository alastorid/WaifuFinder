import Combine
import Foundation

struct PhotosKeywordJobProgress: Equatable, Sendable {
    enum Phase: Sendable { case preparing, writing, stopping }
    var phase: Phase = .preparing
    var completed = 0
    var total: Int

    var title: String {
        switch phase {
        case .preparing: return "Preparing tags"
        case .writing: return "Setting tags"
        case .stopping: return "Stopping…"
        }
    }
}

/// Frequent counts update only the title-bar job view, rather than the entire photo grid.
@MainActor
final class PhotosKeywordJobStatus: ObservableObject {
    @Published var progress: PhotosKeywordJobProgress?
}
