import Foundation

/// Translate workspace frame IDs to the same video/ordinal cache keys used when
/// opening each original alone. Batch order and membership are never cache keys.
struct VideoAnalysisCacheKeys: Sendable {
    let sources: [Int: AnalysisResultCacheIdentity]
    let isCollection: Bool

    static func make(videoURL: URL, analysisWidth: Int, minimumFaceAreaFraction: Float,
                     analysisMode: AnalysisMode) -> VideoAnalysisCacheKeys? {
        let collection = VideoCollection.get(videoURL)
        let urls = collection?.members.map { $0.video.url } ?? [videoURL]
        var sources: [Int: AnalysisResultCacheIdentity] = [:]
        for (index, url) in urls.enumerated() {
            sources[index] = AnalysisResultCacheIdentity(videoURL: url, analysisWidth: analysisWidth,
                minimumFaceAreaFraction: minimumFaceAreaFraction, analysisMode: analysisMode)
        }
        return sources.isEmpty ? nil : VideoAnalysisCacheKeys(sources: sources, isCollection: collection != nil)
    }

    func identity(for sampleIndex: Int) -> AnalysisResultCacheIdentity? {
        sources[isCollection ? VideoCollection.sourceIndex(sampleIndex) : 0]
    }
    func localIndex(_ sampleIndex: Int) -> Int {
        isCollection ? VideoCollection.localIndex(sampleIndex) : sampleIndex
    }
    func workspaceIndex(source: Int, local: Int) -> Int? {
        isCollection ? try? VideoCollection.frameIndex(source: source, local: local) : local
    }
    func records(from frames: [AnalyzedFrame]) -> [AnalysisResultCacheIdentity: [AnalysisResultMemoryRecord]] {
        var grouped: [AnalysisResultCacheIdentity: [Int: Data]] = [:]
        for frame in frames {
            guard let identity = identity(for: frame.id.sampleIndex),
                  let data = AnalysisResultCacheCodec.encode(frame) else { continue }
            grouped[identity, default: [:]][localIndex(frame.id.sampleIndex)] = data
        }
        return grouped.mapValues { records in
            records.map { AnalysisResultMemoryRecord(sampleIndex: $0.key, compressedResult: $0.value) }
                .sorted { $0.sampleIndex < $1.sampleIndex }
        }
    }
}

extension AnalysisResultMemory {
    func cachedResults(for keys: VideoAnalysisCacheKeys) async -> [Int: Data] {
        var combined: [Int: Data] = [:]
        for (source, identity) in keys.sources {
            for (local, data) in await cachedResults(for: identity) {
                if let index = keys.workspaceIndex(source: source, local: local) { combined[index] = data }
            }
        }
        return combined
    }
}
