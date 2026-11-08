import CoreGraphics
import Foundation

func registerArenaRegionalJudgeTests() {
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "regional cache identity binds native pixels geometry and order", knownBug: nil) {
        let rect = ArenaPixelRect(CGRect(x: 10, y: 20, width: 448, height: 448))
        let a = Data([1, 2]), b = Data([3, 4])
        let first = ArenaRegionalJudge.key(images: [a, b], regionID: "r", rect: rect)
        checkEqual(first, ArenaRegionalJudge.key(images: [a, b], regionID: "r", rect: rect), "repeatable cache identity")
        check(first != ArenaRegionalJudge.key(images: [b, a], regionID: "r", rect: rect), "candidate order preserved")
        check(first != ArenaRegionalJudge.key(images: [a, Data([3, 5])], regionID: "r", rect: rect), "source content invalidates cache")
        check(first != ArenaRegionalJudge.key(images: [a, b], regionID: "r", rect: ArenaPixelRect(CGRect(x: 11, y: 20, width: 448, height: 448))), "reference geometry invalidates cache")
        let crop = ArenaRegionalJudge.inspectionRect(CGRect(x: 0, y: 0, width: 1000, height: 500))
        checkEqual(crop.width, 448, "native crop width never globally downsamples")
        checkEqual(crop.height, 448, "native crop height")
    })
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "active region schedule bounds calls and rejects invalid crops", knownBug: nil) {
        let sources = (0..<10).map { index in
            ArenaAnalyzedSource(id: "s\(index)", url: URL(fileURLWithPath: "/fixture"), width: 8000, height: 5000,
                transform: ArenaTransform(), alignmentConfidence: 1, accepted: index != 9, issues: [])
        }
        func region(_ id: String, difference: Double) -> ArenaRegion {
            ArenaRegion(id: id, rect: ArenaPixelRect(CGRect(x: 0, y: 0, width: 448, height: 448)),
                measurements: sources.map { source in
                    ArenaRegionMeasurement(sourceID: source.id, difference: difference, sharpness: 0.1,
                        clipping: 0, validFraction: source.id == "s8" ? 0.5 : 1,
                        textCandidateFraction: 0, evidence: [])
                }, unresolved: true, evidence: [])
        }
        let analysis = ArenaAnalysis(referenceID: "s0", width: 8000, height: 5000, sources: sources,
            regions: [region("identical", difference: 0), region("low", difference: 0.01), region("high", difference: 0.8)],
            warnings: [], elapsedSeconds: 0)
        let schedule = ArenaRegionalJudge.schedule(analysis, maximumComparisons: 4)
        checkEqual(schedule.count, 4, "strict job inference budget")
        check(schedule.allSatisfy { $0.0 == "high" }, "disputed region inspected first")
        check(schedule.allSatisfy { $0.1 == "s0" && $0.2 != "s8" && $0.2 != "s9" }, "only accepted complete overlapping crop pairs")
        checkEqual(ArenaRegionalJudge.schedule(analysis, maximumComparisons: 0).count, 0, "zero budget")
        checkEqual(ArenaRegionalJudge.schedule(analysis, maximumComparisons: 100).count, 14, "linear star excludes exact regions")
        func preference(_ author: String, _ outcome: ArenaOutcome, _ date: Double) -> ArenaVote {
            ArenaVote(id: UUID(), date: Date(timeIntervalSince1970: date), candidateA: "s0", candidateB: "s1",
                category: .authenticity, outcome: outcome, author: author, region: "high", evidence: "assessment")
        }
        check(ArenaRanking.selectionOverrides(analysis: analysis, votes: [preference("AI:model", .b, 0)]).isEmpty,
            "uncalibrated AI never automatically changes source pixels")
        let human = preference("human", .a, 1)
        checkEqual(ArenaRanking.selectionOverrides(analysis: analysis, votes: [human, preference("AI:model", .b, 2)])["high"], "s0", "human preference takes precedence")
        check(ArenaRanking.selectionOverrides(analysis: analysis, votes: [human, preference("human", .unknown, 3)]).isEmpty,
            "later human abstention clears vote-derived override")
    })
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "pairwise ranking respects unknown ties reverse labels and superseded votes", knownBug: nil) {
        func vote(_ outcome: ArenaOutcome, a: String = "A", b: String = "B", seconds: Double = 0) -> ArenaVote {
            ArenaVote(id: UUID(), date: Date(timeIntervalSince1970: seconds), candidateA: a, candidateB: b,
                category: .quality, outcome: outcome, author: "human", region: "r", evidence: "test")
        }
        let unknown = ArenaRanking.pairwise(sourceIDs: ["B", "A"], votes: [vote(.unknown)], category: .quality)
        checkEqual(unknown.map(\.sourceID), ["A", "B"], "stable insufficient-evidence order")
        check(unknown.allSatisfy { $0.tied && $0.compared == 0 }, "abstention is not a loss")
        let tied = ArenaRanking.pairwise(sourceIDs: ["A", "B"], votes: [vote(.tie)], category: .quality)
        check(tied.allSatisfy(\.tied), "explicit tie remains tie")
        let reversed = ArenaRanking.pairwise(sourceIDs: ["A", "B"], votes: [vote(.b, a: "B", b: "A")], category: .quality)
        checkEqual(reversed.first?.sourceID, "A", "reverse labels rank correct source")
        let updated = ArenaRanking.pairwise(sourceIDs: ["A", "B"], votes: [vote(.a), vote(.b, seconds: 10)], category: .quality)
        checkEqual(updated.first?.sourceID, "B", "latest same-author decision replaces previous vote for fitting")
        checkEqual(updated.first?.compared, 1, "repeated author does not add false confidence")
        let disconnected = ArenaRanking.pairwise(sourceIDs: ["A", "B", "C"], votes: [vote(.a)], category: .quality)
        check(disconnected.allSatisfy { $0.tied && $0.score == 0 }, "uncompared source prevents false global preference")
    })
}
