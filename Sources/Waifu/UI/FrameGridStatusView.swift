import SwiftUI

enum FrameGridStatus: Equatable {
    case content, loading, filtered, empty

    static func resolve(snapshotHasRows: Bool, snapshotGenerationMatches: Bool,
                        snapshotRevisionMatches: Bool, analysisRunning: Bool,
                        displayedCount: Int, analyzedCount: Int, filtersActive: Bool) -> Self {
        // Keep the last committed content while its replacement is prepared.
        if snapshotGenerationMatches && snapshotHasRows { return .content }
        if analysisRunning || !snapshotGenerationMatches || !snapshotRevisionMatches || displayedCount > 0 {
            return .loading
        }
        return analyzedCount > 0 && filtersActive ? .filtered : .empty
    }
}

/// Observe status independently of the geometry-keyed grid. Progress can change
/// without a new layout, and an old empty snapshot cannot explain new results.
struct FrameGridStatusView: View {
    @ObservedObject var session: WaifuSession
    @ObservedObject var stage: GridThumbnailStage

    private var status: FrameGridStatus {
        FrameGridStatus.resolve(
            snapshotHasRows: stage.current?.rows.isEmpty == false,
            snapshotGenerationMatches: stage.current?.key.generation == session.thumbnailGeneration,
            snapshotRevisionMatches: stage.current?.key.revision == session.gridRevision,
            analysisRunning: session.progress.isRunning,
            displayedCount: session.displayOrderFrames.count,
            analyzedCount: session.frames.count,
            filtersActive: !session.frameLimits.isEmpty || session.faceFilter != .all
                || session.tagFilter.hasActiveTags)
    }

    var body: some View {
        switch status {
        case .content:
            EmptyView()
        case .loading:
            VStack(spacing: 10) {
                ProgressView()
                Text(session.progress.isRunning
                     ? (session.photosAlbumURL != nil ? "Loading photos…" : "Loading frames…")
                     : "Preparing thumbnails…")
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .textBackgroundColor))
            .allowsHitTesting(false)
        case .filtered:
            VStack(spacing: 8) {
                Image(systemName: "line.3.horizontal.decrease.circle")
                    .font(.system(size: 32))
                    .foregroundStyle(.tertiary)
                Text(session.photosAlbumURL != nil ? "No photos match the active filters" : "No frames match the active filters")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                Text(filterHint)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            }
        case .empty:
            VStack(spacing: 8) {
                Text(emptyTitle)
                    .font(.headline)
                    .foregroundStyle(.secondary)
                Text(session.progress.message)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var emptyTitle: String {
        if session.progress.phase == .failed {
            return session.photosAlbumURL != nil ? "Couldn’t load this album" : "Couldn’t load this video"
        }
        return session.photosAlbumURL != nil ? "No photos to display" : "No frames to display"
    }

    private var filterHint: String {
        if session.tagFilter.hasActiveTags {
            return "Review wanted and unwanted tags in the Tags panel."
        }
        if session.faceFilter != .all { return "Review the face filter in the controls panel." }
        return "Widen or clear a range in the Sort and Filter panel."
    }
}
