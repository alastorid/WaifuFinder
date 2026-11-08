import CoreGraphics
import CryptoKit
import Foundation

/// Evidence is keyed by the pixels actually submitted, their reference geometry,
/// and the exact runtime/prompt contract. No image or activation scratch is stored.
struct ArenaRegionalComparison: Codable, Sendable {
    let regionID: String
    let candidateA: String
    let candidateB: String
    let referenceRect: ArenaPixelRect
    let runtimeRevision: String
    let cacheKey: String
    let result: ArenaJudgeResult
}

struct ArenaSourceRank: Codable, Identifiable, Sendable {
    var id: String { sourceID }
    let sourceID: String
    let score: Double
    let compared: Int
    let tied: Bool
    let reasons: [String]
}

actor ArenaRegionalJudge {
    static let shared = ArenaRegionalJudge()
    static let contractRevision = "native-roi-448-v1"
    private let cacheDirectory: URL
    init(cacheDirectory: URL? = nil) {
        self.cacheDirectory = cacheDirectory ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Waifu/PhotoArena/judgments", isDirectory: true)
    }
    nonisolated static func key(images: [Data], regionID: String, rect: ArenaPixelRect) -> String {
        var digest = SHA256()
        digest.update(data: Data((ArenaJudgeConfiguration.revision + "|" + contractRevision + "|" + regionID).utf8))
        // Length-prefixed fields cannot collide through ambiguous concatenation.
        let geometry = [rect.x, rect.y, rect.width, rect.height].map { String($0.bitPattern, radix: 16) }.joined(separator: "|")
        for field in [Data(geometry.utf8)] + images {
            var count = UInt64(field.count).bigEndian
            withUnsafeBytes(of: &count) { digest.update(data: Data($0)) }
            digest.update(data: field)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
    /// Active sparse schedule: disputed regions first, compare each other source
    /// to one anchor. A strict whole-job cap prevents quadratic VLM work.
    nonisolated static func schedule(_ analysis: ArenaAnalysis, maximumComparisons: Int = 8) -> [(String, String, String)] {
        let accepted = Set(analysis.sources.filter(\.accepted).map(\.id))
        let regions = analysis.regions.sorted {
            let a = $0.measurements.map(\.difference).max() ?? 0
            let b = $1.measurements.map(\.difference).max() ?? 0
            return a == b ? $0.id < $1.id : a > b
        }
        var pairs: [(String, String, String)] = []
        for region in regions {
            let candidates = region.measurements.filter { accepted.contains($0.sourceID) && $0.validFraction >= 0.995 }
                .sorted { $0.sourceID < $1.sourceID }
            guard candidates.count >= 2,
                  candidates.contains(where: { $0.difference > 0.001 }) else { continue }
            let anchor = candidates.first(where: { $0.sourceID == analysis.referenceID }) ?? candidates[0]
            for candidate in candidates where candidate.sourceID != anchor.sourceID {
                if pairs.count >= max(0, min(64, maximumComparisons)) { return pairs }
                pairs.append((region.id, anchor.sourceID, candidate.sourceID))
            }
        }
        return pairs
    }
    func judge(analysis: ArenaAnalysis, maximumComparisons: Int = 8,
               infer: @Sendable ([Data]) async throws -> ArenaJudgeResult = { try await ArenaJudge.local($0) }) async throws -> [ArenaRegionalComparison] {
        var records: [ArenaRegionalComparison] = []
        for (regionID, aID, bID) in Self.schedule(analysis, maximumComparisons: maximumComparisons) {
            try Task.checkCancellation()
            guard let region = analysis.regions.first(where: { $0.id == regionID }),
                  let a = analysis.sources.first(where: { $0.id == aID }),
                  let b = analysis.sources.first(where: { $0.id == bID }) else { continue }
            let rect = Self.inspectionRect(region.rect.cgRect)
            let inputs = try [a, b].map { source in
                try ArenaJudge.png(ArenaForensics.crop(source, rect: rect))
            }
            let key = Self.key(images: inputs, regionID: regionID, rect: ArenaPixelRect(rect))
            let url = cacheDirectory.appendingPathComponent(key + ".json")
            let record: ArenaRegionalComparison
            if let data = try? Data(contentsOf: url),
               let saved = try? JSONDecoder().decode(ArenaRegionalComparison.self, from: data),
               saved.cacheKey == key, saved.candidateA == aID, saved.candidateB == bID,
               saved.runtimeRevision == ArenaJudgeConfiguration.revision,
               (try? saved.result.validate()) != nil {
                record = saved
            } else {
                let result = try await infer(inputs)
                try result.validate()
                try Task.checkCancellation()
                record = ArenaRegionalComparison(regionID: regionID, candidateA: aID, candidateB: bID,
                    referenceRect: ArenaPixelRect(rect), runtimeRevision: ArenaJudgeConfiguration.revision,
                    cacheKey: key, result: result)
                try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
                try JSONEncoder().encode(record).write(to: url, options: .atomic)
            }
            records.append(record)
        }
        return records
    }
    nonisolated static func inspectionRect(_ rect: CGRect) -> CGRect {
        let width = min(448, rect.width), height = min(448, rect.height)
        return CGRect(x: floor(rect.midX - width / 2), y: floor(rect.midY - height / 2), width: width, height: height)
    }
    nonisolated static func votes(_ records: [ArenaRegionalComparison]) -> [ArenaVote] {
        records.flatMap { record in
            ArenaCategory.allCases.map { category in
                ArenaVote(id: UUID(), date: Date(), candidateA: record.candidateA, candidateB: record.candidateB,
                    category: category, outcome: record.result.judgment.outcome(category),
                    author: "AI:" + record.runtimeRevision, region: record.regionID,
                    evidence: record.result.judgment.evidence.joined(separator: " • ") + " • Uncalibrated crop judgment; pixel provenance is not established.")
            }
        }
    }
}

/// Bradley–Terry-style regularized logistic fitting. Unknown outcomes contribute
/// no edge; repeated decisions are not treated as independent extra confidence.
enum ArenaRanking {
    /// Human authenticity preferences can guide pixel selection, while their
    /// unproven status remains in the selection manifest. AI votes alone never
    /// become automatic source overrides, even when the model says "certain".
    static func selectionOverrides(analysis: ArenaAnalysis, votes: [ArenaVote]) -> [String: String] {
        let accepted = Set(analysis.sources.filter(\.accepted).map(\.id))
        var latest: [String: ArenaVote] = [:]
        for vote in votes where vote.author == "human" && vote.category == .authenticity {
            guard accepted.contains(vote.candidateA), accepted.contains(vote.candidateB),
                  vote.candidateA != vote.candidateB,
                  let region = analysis.regions.first(where: { $0.id == vote.region }),
                  region.measurements.contains(where: { $0.sourceID == vote.candidateA && $0.validFraction >= 0.995 }),
                  region.measurements.contains(where: { $0.sourceID == vote.candidateB && $0.validFraction >= 0.995 }) else { continue }
            if latest[vote.region].map({ $0.date <= vote.date }) ?? true { latest[vote.region] = vote }
        }
        return latest.reduce(into: [:]) { result, entry in
            let vote = entry.value
            if vote.outcome == .a { result[entry.key] = vote.candidateA }
            if vote.outcome == .b { result[entry.key] = vote.candidateB }
        }
    }
    static func pairwise(sourceIDs: [String], votes: [ArenaVote], category: ArenaCategory,
                         regionID: String? = nil) -> [ArenaSourceRank] {
        let ids = Array(Set(sourceIDs)).sorted()
        let indices = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        var latest: [String: ArenaVote] = [:]
        for vote in votes where vote.category == category && (regionID == nil || vote.region == regionID) {
            guard vote.candidateA != vote.candidateB, indices[vote.candidateA] != nil,
                  indices[vote.candidateB] != nil else { continue }
            let pair = [vote.candidateA, vote.candidateB].sorted().joined(separator: "\u{0}")
            let key = pair + "|" + vote.region + "|" + vote.author
            if latest[key].map({ $0.date <= vote.date }) ?? true { latest[key] = vote }
        }
        let edges = latest.values.filter { $0.outcome != .unknown }.sorted { $0.id.uuidString < $1.id.uuidString }
        var connected: Set<String> = ids.first.map { [$0] } ?? []
        var changed = true
        while changed {
            changed = false
            for edge in edges where connected.contains(edge.candidateA) || connected.contains(edge.candidateB) {
                if connected.insert(edge.candidateA).inserted { changed = true }
                if connected.insert(edge.candidateB).inserted { changed = true }
            }
        }
        let disconnected = connected.count < ids.count
        var values = Array(repeating: 0.0, count: ids.count)
        var counts = Array(repeating: 0, count: ids.count)
        for edge in edges { counts[indices[edge.candidateA]!] += 1; counts[indices[edge.candidateB]!] += 1 }
        for _ in 0..<200 {
            var gradient = values.map { -0.25 * $0 }
            for edge in edges {
                let a = indices[edge.candidateA]!, b = indices[edge.candidateB]!
                let target = edge.outcome == .a ? 1.0 : edge.outcome == .b ? 0.0 : 0.5
                let probability = 1 / (1 + exp(values[b] - values[a]))
                gradient[a] += target - probability; gradient[b] -= target - probability
            }
            let step = 0.3 / Double(max(1, edges.count))
            for index in values.indices { values[index] += step * gradient[index] }
        }
        return ids.enumerated().map { index, id in
            ArenaSourceRank(sourceID: id, score: disconnected ? 0 : values[index], compared: counts[index],
                tied: disconnected || values.enumerated().contains { $0.offset != index && abs($0.element - values[index]) < 0.001 },
                reasons: disconnected ? ["Disconnected comparison graph; a global preference is unestablished. Stable display order only."] :
                    counts[index] == 0 ? ["Insufficient pairwise evidence; stable display order only."] :
                    ["Relative pairwise preference, not an authenticity probability."])
        }.sorted {
            let a = ($0.score * 1000).rounded(), b = ($1.score * 1000).rounded()
            return a == b ? $0.sourceID < $1.sourceID : a > b
        }
    }
    /// Deterministic initial quality order. Separate from authenticity judgments.
    static func technical(_ analysis: ArenaAnalysis) -> [ArenaSourceRank] {
        let ranks = analysis.sources.map { source -> ArenaSourceRank in
            var total = 0.0, area = 0.0
            for region in analysis.regions {
                guard let measurement = region.measurements.first(where: { $0.sourceID == source.id }),
                      measurement.validFraction >= 0.995 else { continue }
                let weight = region.rect.width * region.rect.height
                let sharpness = min(1, max(0, measurement.sharpness))
                total += weight * (sharpness - 0.3 * measurement.clipping - 0.2 * measurement.difference)
                area += weight
            }
            return ArenaSourceRank(sourceID: source.id, score: source.accepted && area > 0 ? total / area : -1,
                compared: 0, tied: false, reasons: source.accepted ?
                    ["Measured texture, clipping and cross-source discrepancy; originality unknown."] : ["Alignment rejected."])
        }
        return ranks.map { rank in
            ArenaSourceRank(sourceID: rank.sourceID, score: rank.score, compared: rank.compared,
                tied: ranks.contains { $0.sourceID != rank.sourceID && abs($0.score - rank.score) < 0.001 }, reasons: rank.reasons)
        }.sorted {
            let a = ($0.score * 1000).rounded(), b = ($1.score * 1000).rounded()
            return a == b ? $0.sourceID < $1.sourceID : a > b
        }
    }
}
