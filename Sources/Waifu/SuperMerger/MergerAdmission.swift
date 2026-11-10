import CoreGraphics
import Foundation

/// Admission preserves every measured pixel. It never trims regions to meet a quota.
enum MergerAdmission {
    static let maximumRegions = 7

    struct Assessment: Sendable {
        let analysis: ArenaAnalysis
        let excluded: [ArenaAnalyzedSource]
        let reason: String?
        var admitted: Bool { reason == nil }
    }

    static func assess(_ input: ArenaAnalysis) throws -> Assessment {
        try Task.checkCancellation()
        let accepted = input.sources.filter { $0.accepted }
        let excluded = input.sources.filter { !$0.accepted }
        guard accepted.count >= 2 else {
            return Assessment(analysis: input, excluded: excluded,
                reason: "Fewer than two sources have reliable correspondence to this photograph.")
        }
        guard !input.regions.contains(where: { $0.maskDeferred == true }) else {
            return Assessment(analysis: input, excluded: excluded,
                reason: "Difference masks exceed the processing budget. Automatic reconstruction is unavailable.")
        }
        let rawRuns: [ArenaMaskRun] = input.regions.flatMap { $0.changeMask?.runs ?? [] }
        let runs = rawRuns.sorted {
            $0.y == $1.y ? $0.x < $1.x : $0.y < $1.y
        }
        let mask = ArenaPixelMask(runs: runs)
        // A global mismatch can collapse into one connected component. Region count
        // alone must never turn such a mismatch into an eligible sparse edit.
        let fraction = Double(mask.pixelCount) / (Double(input.width) * Double(input.height))
        guard fraction <= 0.35 else {
            return Assessment(analysis: input, excluded: excluded,
                reason: "Differences cover more than 35% of the reference image. Color, compression or alignment requires further analysis before merging.")
        }
        let components = mask.components
        guard components.count <= 4096 else {
            return Assessment(analysis: input, excluded: excluded,
                reason: "Differences are too fragmented for bounded automatic analysis.")
        }
        var groups: [ArenaPixelMask] = []
        let spacing = max(3.0, min(16.0, Double(min(input.width, input.height)) * 0.01))
        for component in components {
            try Task.checkCancellation()
            var merged = component
            var index = 0
            while index < groups.count {
                if groups[index].bounds.insetBy(dx: -spacing, dy: -spacing).intersects(merged.bounds) {
                    merged.runs.append(contentsOf: groups.remove(at: index).runs)
                    index = 0
                } else { index += 1 }
            }
            groups.append(merged)
        }
        groups.sort {
            $0.bounds.minY == $1.bounds.minY ? $0.bounds.minX < $1.bounds.minX : $0.bounds.minY < $1.bounds.minY
        }
        var result = input
        result.sources = accepted
        result.regions = groups.enumerated().map { index, group in
            let bounds = group.bounds
            let tiles = input.regions.filter { $0.rect.cgRect.intersects(bounds) }
            return ArenaRegion(id: "merger-region-\(index + 1)", rect: ArenaPixelRect(bounds),
                measurements: measurements(tiles, sources: accepted), unresolved: true,
                evidence: ["Grouped across analysis tile boundaries; all measured changed pixels retained."],
                changeMask: ArenaPixelMask(runs: group.runs.sorted {
                    $0.y == $1.y ? $0.x < $1.x : $0.y < $1.y
                }))
        }
        // Preserve whole-canvas coverage constraints for backbone selection.
        result.regions.append(ArenaRegion(id: "canvas-coverage",
            rect: ArenaPixelRect(CGRect(x: 0, y: 0, width: input.width, height: input.height)),
            measurements: measurements(input.regions, sources: accepted), unresolved: false,
            evidence: [], changeMask: ArenaPixelMask(runs: [])))
        let reason = groups.count > maximumRegions
            ? "\(groups.count) distinct difference regions remain. Super Merger supports at most seven; all regions are retained for inspection."
            : nil
        return Assessment(analysis: result, excluded: excluded, reason: reason)
    }

    private static func measurements(_ regions: [ArenaRegion], sources: [ArenaAnalyzedSource]) -> [ArenaRegionMeasurement] {
        sources.map { source in
            let values = regions.flatMap(\.measurements).filter { $0.sourceID == source.id }
            return ArenaRegionMeasurement(sourceID: source.id,
                difference: values.map(\.difference).max() ?? 0,
                sharpness: values.map(\.sharpness).min() ?? 0,
                clipping: values.map(\.clipping).max() ?? 0,
                validFraction: values.map(\.validFraction).min() ?? 0,
                textCandidateFraction: values.map(\.textCandidateFraction).max() ?? 0,
                evidence: ["Conservative bounds across the contributing analysis tiles."])
        }
    }
}
