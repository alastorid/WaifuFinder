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
}

struct TagResult: Sendable {
    let probabilities: [Float]
    let rating: TagRating
    let topTags: [String]
    let generalScore: Float
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
