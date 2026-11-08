import Foundation

/// Decisions retain category and author independently; unknown is never a loss.
enum ArenaOutcome: String, Codable, CaseIterable, Sendable { case a = "A", b = "B", tie, unknown }
enum ArenaCategory: String, Codable, CaseIterable, Sendable {
    case authenticity = "Authenticity evidence", watermark = "Clean pixels", detail = "Detail", quality = "Technical quality", consistency = "Source consistency"
}
struct ArenaSource: Codable, Identifiable, Hashable {
    let id: String
    let url: URL
    let seconds: Double
}
struct ArenaVote: Codable, Identifiable {
    let id: UUID
    let date: Date
    let candidateA: String
    let candidateB: String
    let category: ArenaCategory
    let outcome: ArenaOutcome
    let author: String
    let region: String
    let evidence: String
}
struct ArenaDocument: Codable {
    var schemaVersion = 1
    let sources: [ArenaSource]
    var votes: [ArenaVote] = []

    /// Deterministic sparse schedule: adjacent sources plus the closing edge.
    /// This stays linear in group size and retains non-transitive comparisons.
    var pairs: [(ArenaSource, ArenaSource)] {
        guard sources.count >= 2 else { return [] }
        var result = zip(sources, sources.dropFirst()).map { ($0, $1) }
        if sources.count > 2 { result.append((sources[sources.count - 1], sources[0])) }
        return result
    }
    func validate() throws {
        guard schemaVersion == 1, sources.count >= 2,
              Set(sources.map(\.id)).count == sources.count,
              sources.allSatisfy({ $0.seconds.isFinite && $0.seconds >= 0 }) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let ids = Set(sources.map(\.id))
        guard votes.allSatisfy({ ids.contains($0.candidateA) && ids.contains($0.candidateB)
            && $0.candidateA != $0.candidateB && !$0.author.isEmpty }) else {
            throw CocoaError(.fileReadCorruptFile)
        }
    }
}
