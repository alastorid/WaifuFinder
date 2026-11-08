import Combine
import Foundation

/// Owns cloud work by viewport rather than by the lifetime of lazy grid cells.
@MainActor
final class GridPhotoQualityLoader: ObservableObject {
    private struct Request {
        let identity: String
        let frame: AnalyzedFrame
    }

    private struct Active {
        let id: UUID
        let identity: String
        let ticket: PhotosCloudQueue.PriorityTicket
        let task: Task<Void, Never>
    }

    private var generation: UUID?
    private var source: URL?
    private var visible: [Request] = []
    private var neighbors: [Request] = []
    private var attempted: Set<String> = []
    private var active: Active?
    private var focused: FrameID?
    private var selected: Set<FrameID> = []

    func update(snapshot: GridStageSnapshot, viewport: CGRect, horizontalOffsets: [String: CGFloat],
                focused: FrameID?, selected: Set<FrameID>, preloading: Bool) {
        guard let url = snapshot.sourceURL, PhotosMedia.isPhotos(url) else { stop(); return }
        if generation != snapshot.key.generation || source != url {
            stop()
            generation = snapshot.key.generation
            source = url
        }

        self.focused = focused
        self.selected = selected
        let range = snapshot.visibleRowRange(in: viewport)
        let items = range.flatMap {
            snapshot.items(inRow: $0, width: viewport.width, horizontalOffsets: horizontalOffsets)
        }
        let prefix = snapshot.key.generation.uuidString + ":"
        func request(_ item: GridStageItem) -> Request {
            Request(identity: prefix + item.frame.thumbnailKey, frame: item.frame)
        }

        let previousVisible = Set(visible.map(\.identity))
        visible = items.filter { $0.id == focused }.map(request)
            + items.filter { $0.id != focused && selected.contains($0.id) }.map(request)
            + items.filter { $0.id != focused && !selected.contains($0.id) }.map(request)
        neighbors = []
        if preloading, !range.isEmpty {
            for distance in 1...7 {
                for row in [range.lowerBound - distance, range.upperBound + distance - 1]
                    where snapshot.rows.indices.contains(row) {
                    neighbors.append(contentsOf: snapshot.items(inRow: row, width: viewport.width,
                                                                horizontalOffsets: horizontalOffsets).map(request))
                }
            }
        }

        let visibleIDs = Set(visible.map(\.identity))
        let neighborIDs = Set(neighbors.map(\.identity))
        attempted.formIntersection(visibleIDs.union(neighborIDs))
        for identity in visibleIDs.subtracting(previousVisible)
            where !GridThumbnailLoader.shared.hasOriginal(identity: identity) {
            attempted.remove(identity)
        }
        if let active, let request = visible.first(where: { $0.identity == active.identity }) {
            active.ticket.update(priority(for: request))
        }
        if let active, !visibleIDs.contains(active.identity),
           previousVisible.contains(active.identity) || !neighborIDs.contains(active.identity)
                || visible.contains(where: { !attempted.contains($0.identity) }) {
            active.task.cancel()
            self.active = nil
        }
        startNext()
    }

    func stop() {
        active?.task.cancel()
        active = nil
        generation = nil
        source = nil
        visible.removeAll()
        neighbors.removeAll()
        attempted.removeAll()
    }

    private func priority(for request: Request) -> PhotosCloudQueue.Priority {
        request.frame.id == focused || selected.contains(request.frame.id) ? .selectedThumbnail : .thumbnail
    }

    private func startNext() {
        guard active == nil, let source else { return }
        let nextVisible = visible.first { !attempted.contains($0.identity) }
        guard let request = nextVisible ?? neighbors.first(where: { !attempted.contains($0.identity) }) else { return }
        let ticket = PhotosCloudQueue.PriorityTicket(nextVisible == nil ? .preload : priority(for: request))
        let id = UUID()
        let task = Task { [weak self] in
            if !GridThumbnailLoader.shared.hasOriginal(identity: request.identity),
               let url = PhotosMedia.shared.photoItemURL(for: source, seconds: request.frame.timeSeconds),
               let pixels = try? await PhotosMedia.shared.originalThumbnail(url, ticket: ticket),
               !Task.isCancelled {
                GridThumbnailLoader.shared.upgrade(pixels, identity: request.identity, isOriginal: true)
            }
            guard let self, self.active?.id == id else { return }
            self.active = nil
            self.attempted.insert(request.identity)
            self.startNext()
        }
        active = Active(id: id, identity: request.identity, ticket: ticket, task: task)
    }
}
