import CoreGraphics
import Foundation

/// Rating head order is pinned to the exported model, rather than alphabetic.
enum TagRating: String, Codable, CaseIterable, Sendable {
    case general, sensitive, questionable, explicit
    var index: Int { Self.allCases.firstIndex(of: self)! }
}

protocol TaggerBackend: Sendable {
    /// Accepts 1...8 images. The implementation pads a tail to its fixed batch,
    /// then returns one result per input, preserving input order.
    func tag(batch: [CGImage]) async throws -> [TagResult]

    /// Preserve cached encoded frames until the selected lane prepares them.
    /// Remote lanes can upload a thumbnail without decoding and rasterizing it.
    func tag(sources: [TaggingImageSource]) async throws -> [TagResult]

    /// The raw `Float32[10_861]` rows, before any threshold is applied.
    ///
    /// Retained for numerical validation. Production tagging carries compact
    /// UInt8/LZ4 rows, including all four rating probabilities.
    func probabilityRows(batch: [CGImage]) async throws -> [[Float]]
}

extension TaggerBackend {
    func tag(sources: [TaggingImageSource]) async throws -> [TagResult] {
        let images = try await TaggingImageLoader.load(sources)
        return try await tag(batch: images)
    }

    func probabilityRows(batch: [CGImage]) async throws -> [[Float]] {
        try await tag(batch: batch).map(\.probabilities)
    }
}

struct TagResult: Sendable {
    let probabilities: [Float]
    let rating: TagRating
    let topTags: [String]
    let generalScore: Float
    var encodedProbabilities: Data? = nil
}

/// Shared native-worker and network result with four UInt8 rating scores.
struct QuantizedTagRow: Sendable, Equatable {
    let scores: Data

    func result(vocabulary: TagVocabulary, threshold: Float) throws -> TagResult {
        guard let probabilities = TagProbabilityCodec.decode(scores) else { throw TaggerError.invalidProbabilities }
        let derived = try vocabulary.result(probabilities: probabilities, threshold: threshold)
        return TagResult(probabilities: probabilities, rating: derived.rating, topTags: derived.topTags,
                         generalScore: derived.generalScore, encodedProbabilities: scores)
    }
}

enum TaggerError: LocalizedError {
    case invalidVocabulary(String)
    case invalidModel(String)
    case invalidProbabilities
    case invalidBatch
    case preprocessing

    var errorDescription: String? {
        switch self {
        case .invalidVocabulary(let reason): return "Invalid WD Tagger labels: \(reason)"
        case .invalidModel(let reason): return "Invalid WD Tagger model: \(reason)"
        case .invalidProbabilities: return "WD Tagger returned invalid probabilities."
        case .invalidBatch: return "WD Tagger accepts at most eight frames per batch."
        case .preprocessing: return "Could not prepare the frame for WD Tagger."
        }
    }
}
