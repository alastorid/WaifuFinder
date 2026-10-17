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

    /// The raw `Float32[10_861]` rows, before any threshold is applied.
    ///
    /// A remote server ships these rather than `TagResult`, because the threshold
    /// is a *client* preference: applying it on the serving machine would bake one
    /// machine's setting into another machine's results.
    ///
    /// The default recovers the rows from `tag`, whose results already carry
    /// them, so a backend that only implements `tag` still works over the wire —
    /// it just cannot avoid its own threshold having been applied first.
    func probabilityRows(batch: [CGImage]) async throws -> [[Float]]
}

extension TaggerBackend {
    func probabilityRows(batch: [CGImage]) async throws -> [[Float]] {
        try await tag(batch: batch).map(\.probabilities)
    }
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
