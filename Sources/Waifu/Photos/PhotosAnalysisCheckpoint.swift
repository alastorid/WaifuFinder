import Foundation

/// One album revision and scoring configuration. Completed work survives a
/// download-policy change, even before the next grid snapshot is published.
actor PhotosAnalysisCheckpoint {
    let sourceURL: URL
    let settings: WaifuSettings
    private var completed: [FrameID: AnalyzedFrame] = [:]

    init(sourceURL: URL, settings: WaifuSettings) {
        self.sourceURL = sourceURL
        self.settings = settings
    }

    func retain(_ frame: AnalyzedFrame) { completed[frame.id] = frame }

    func frames() -> [AnalyzedFrame] {
        completed.values.sorted { ($0.timeSeconds, $0.id.raw) < ($1.timeSeconds, $1.id.raw) }
    }

    /// The merger emits an ordered prefix. Keep its newly grouped results and
    /// append the retained photos it has not reached yet, so retry publications
    /// can advance the grid/count without making the existing tail disappear.
    nonisolated static func includingRetainedTail(_ retained: [AnalyzedFrame],
                                                after prefix: [AnalyzedFrame]) -> [AnalyzedFrame] {
        guard let last = prefix.last else { return retained }
        let boundary = (last.timeSeconds, last.id.raw)
        return prefix + retained.drop { ($0.timeSeconds, $0.id.raw) <= boundary }
    }

    /// Replaying retained scores through grouping preserves chronological
    /// neighbourhoods when a newly downloaded photo fills an earlier gap.
    nonisolated static func merging(_ retained: [AnalyzedFrame],
                                   with incoming: AsyncThrowingStream<AnalyzedFrame, Error>) -> AsyncThrowingStream<AnalyzedFrame, Error> {
        let merger = PhotosAnalysisMerger(retained: retained, incoming: incoming)
        return AsyncThrowingStream(unfolding: { try await merger.next() })
    }
}

private actor PhotosAnalysisMerger {
    private let retained: [AnalyzedFrame]
    private var index = 0
    private var iterator: AsyncThrowingStream<AnalyzedFrame, Error>.Iterator
    private var nextIncoming: AnalyzedFrame?
    private var exhausted = false

    init(retained: [AnalyzedFrame], incoming: AsyncThrowingStream<AnalyzedFrame, Error>) {
        self.retained = retained
        iterator = incoming.makeAsyncIterator()
    }

    func next() async throws -> AnalyzedFrame? {
        try Task.checkCancellation()
        if nextIncoming == nil, !exhausted {
            var active = iterator
            nextIncoming = try await active.next(isolation: self)
            iterator = active
            exhausted = nextIncoming == nil
        }
        try Task.checkCancellation()
        if index < retained.count {
            let frame = retained[index]
            if let incoming = nextIncoming, incoming.id == frame.id {
                index += 1
                nextIncoming = nil
                return incoming
            }
            if nextIncoming == nil || (frame.timeSeconds, frame.id.raw) < (nextIncoming!.timeSeconds, nextIncoming!.id.raw) {
                index += 1
                return frame
            }
        }
        defer { nextIncoming = nil }
        return nextIncoming
    }
}
