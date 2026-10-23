import Foundation

/// Builds replacement keywords from the full stored scores, never the capped top-tag list.
enum PhotosKeywords {
    struct Update: Sendable {
        let assetIdentifier: String
        let keywords: [String]
    }

    static func keywords(for frame: AnalyzedFrame, vocabulary: TagVocabulary,
                         threshold: Float = 0.35) throws -> [String] {
        try keywords(for: frame, vocabulary: vocabulary, fingerprint: vocabulary.cacheFingerprint,
                     threshold: threshold)
    }

    /// Validate the vocabulary once per job, rather than rehashing every label for every photo.
    static func prepare(_ frames: [AnalyzedFrame], vocabulary: TagVocabulary, threshold: Float,
                        identifier: (AnalyzedFrame) throws -> String) throws -> [Update] {
        let fingerprint = vocabulary.cacheFingerprint
        return try frames.map { frame in
            try Task.checkCancellation()
            return Update(assetIdentifier: try identifier(frame),
                          keywords: try keywords(for: frame, vocabulary: vocabulary,
                                                 fingerprint: fingerprint, threshold: threshold))
        }
    }

    private static func keywords(for frame: AnalyzedFrame, vocabulary: TagVocabulary,
                                 fingerprint: UInt64, threshold: Float) throws -> [String] {
        guard threshold.isFinite, (0.35...1).contains(threshold),
              frame.tagVocabularyFingerprint == nil || frame.tagVocabularyFingerprint == fingerprint,
              let rating = frame.tagRating,
              let data = frame.tagProbabilities,
              let scores = TagProbabilityCodec.sparseValues(in: data) else {
            throw PhotosMedia.failure("Tag this photo with the current tagger before writing keywords.")
        }
        let tags = scores.filter {
            vocabulary.labels.indices.contains($0.index)
                && vocabulary.labels[$0.index].category != 9 && $0.value > threshold
        }.sorted { $0.value == $1.value ? $0.index < $1.index : $0.value > $1.value }
        var seen: Set<String> = [rating.rawValue]
        return [rating.rawValue] + tags.compactMap {
            let name = vocabulary.labels[$0.index].name
            return seen.insert(name).inserted ? name : nil
        }
    }

}
