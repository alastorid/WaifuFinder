import Foundation
import CoreGraphics
import Vision

struct CrystalOrbMatch: Identifiable, Sendable, Equatable {
    let frame: AnalyzedFrame
    let distance: Double
    var id: FrameID { frame.id }

    static func ranked(_ values: [Self]) -> [Self] {
        values.filter { $0.distance.isFinite && $0.distance >= 0 }.sorted {
            $0.distance == $1.distance ? $0.id.raw < $1.id.raw : $0.distance < $1.distance
        }
    }
}

/// A serial, session-scoped revision-2 index. It reuses the application's persisted
/// Vision payloads; missing/corrupt entries use analysis renditions, never originals.
/// No SwiftUI or presentation state crosses this boundary.
actor CrystalOrbSearchService {
    private var source: UUID?
    private var features: [FrameID: FeaturePrintObservation] = [:]
    private var queryFeature: (UUID, FeaturePrintObservation)?
    private(set) var queryComputations = 0
    var cachedCount: Int { features.count }
    private(set) var computations = 0

    func search(query: CGImage, queryID: UUID, frames: [AnalyzedFrame], sourceURL: URL, generation: UUID,
                progress: @Sendable (Int, Int) async -> Void) async throws -> [CrystalOrbMatch] {
        try Task.checkCancellation()
        if source != generation {
            source = generation
            features.removeAll()
            computations = 0
        }
        let queryObservation: FeaturePrintObservation
        if let cached = queryFeature, cached.0 == queryID { queryObservation = cached.1 }
        else {
            do { queryObservation = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: query) }
            catch { try Task.checkCancellation(); throw error }
            try Task.checkCancellation()
            queryFeature = (queryID, queryObservation)
            queryComputations += 1
        }
        var matches: [CrystalOrbMatch] = []
        for (index, frame) in frames.enumerated() {
            try Task.checkCancellation()
            guard source == generation else { throw CancellationError() }
            var feature = features[frame.id]
            if feature == nil { feature = frame.featurePrint.flatMap(FeaturePrintCodec.decode) }
            if feature == nil {
                do {
                    let image = try await PhotosMedia.shared.analysisImage(sourceURL, seconds: frame.timeSeconds)
                    try Task.checkCancellation()
                    feature = try await GenerateImageFeaturePrintRequest(.revision2).perform(on: image)
                    computations += 1
                } catch is CancellationError { throw CancellationError() }
                catch { /* Explicitly counted as unavailable in the result summary. */ }
            }
            try Task.checkCancellation()
            guard source == generation else { throw CancellationError() }
            if let feature {
                features[frame.id] = feature
                if let distance = try? queryObservation.distance(to: feature), distance.isFinite {
                    matches.append(CrystalOrbMatch(frame: frame, distance: distance))
                }
            }
            if index % 32 == 0 || index == frames.count - 1 { await progress(index + 1, frames.count) }
        }
        return CrystalOrbMatch.ranked(matches)
    }
}
