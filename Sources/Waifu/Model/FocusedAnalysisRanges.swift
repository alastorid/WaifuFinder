import Foundation

/// The union of the selected thumbnails' bounded analysis windows, in time order.
struct FocusedAnalysisRanges {
    static func make(times: [Double], duration: Double) -> [ClosedRange<Double>] {
        guard duration.isFinite, duration >= 0 else { return [] }
        let ranges = times.filter { $0.isFinite && $0 >= 0 && $0 <= duration }
            .map { max(0, $0 - 3)...min(duration, $0 + 3) }
            .sorted { $0.lowerBound < $1.lowerBound }
        var folded: [ClosedRange<Double>] = []
        for range in ranges {
            if let last = folded.last, range.lowerBound <= last.upperBound {
                folded[folded.count - 1] = last.lowerBound...max(last.upperBound, range.upperBound)
            } else {
                folded.append(range)
            }
        }
        return folded
    }

    static func targets(clicked: AnalyzedFrame, selected: Set<FrameID>, frames: [AnalyzedFrame]) -> [AnalyzedFrame] {
        selected.count > 1 && selected.contains(clicked.id)
            ? frames.filter { selected.contains($0.id) } : [clicked]
    }
}
