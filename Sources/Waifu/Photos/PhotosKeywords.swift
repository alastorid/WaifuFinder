import Foundation

/// Builds replacement keywords from the full stored scores, never the capped top-tag list.
enum PhotosKeywords {
    struct Update: Sendable {
        let assetIdentifier: String
        let keywords: [String]
    }

    static func keywords(for frame: AnalyzedFrame, vocabulary: TagVocabulary,
                         threshold: Float = 0.35) throws -> [String] {
        guard threshold.isFinite, (0.35...1).contains(threshold),
              frame.tagVocabularyFingerprint == nil || frame.tagVocabularyFingerprint == vocabulary.cacheFingerprint,
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

    static func replacementScript(for updates: [Update]) -> String {
        let assignments = updates.map { update in
            let keywords = update.keywords.map(literal).joined(separator: ", ")
            return "set keywords of media item id \(literal(update.assetIdentifier)) to {\(keywords)}"
        }.joined(separator: "\n")
        return "tell application id \"com.apple.Photos\"\n\(assignments)\nend tell"
    }

    private static func literal(_ text: String) -> String {
        "\"" + text.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\t", with: "\\t") + "\""
    }
}
