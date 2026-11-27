import CoreGraphics
import Foundation

/// Versioned N-source review. Scores are comparative model assessments, never probabilities.
enum ArenaDefect: String, Codable, CaseIterable, Sendable {
    case watermark, text, overlay, retouch, aiAlteration, upscale, sharpening, denoising, compression, color, geometry, natural, unknown
}
enum ArenaReviewConfidence: String, Codable, Sendable { case low, medium, high }
struct ArenaCandidateFinding: Codable, Sendable {
    let source: Int
    let originality: Int
    let cleanliness: Int
    let detail: Int
    let alteration: Int
    let enhancement: Int
    let degradation: Int
    let consistency: Int
    let usable: Bool
    let confidence: ArenaReviewConfidence
    let defects: [ArenaDefect]
    let evidence: String
}
struct ArenaGroupAssessment: Codable, Sendable {
    let sources: [ArenaCandidateFinding]
    var selected_source: Int? = nil
    var unresolved: Bool? = nil
    var reason: String? = nil
    func validate(count: Int) throws {
        if let unresolved {
            guard let reason, !reason.isEmpty, reason.utf8.count <= 2048,
                  unresolved == (selected_source == nil) else {
                throw ModelInstaller.fail("Inconsistent Super Merger donor decision")
            }
            if let selected_source {
                guard let selected = sources.first(where: { $0.source == selected_source }),
                      selected.usable, selected.confidence != .low else {
                    throw ModelInstaller.fail("Selected donor lacks usable, confident evidence")
                }
            }
        } else if selected_source != nil || reason != nil {
            throw ModelInstaller.fail("Incomplete Super Merger donor decision")
        }
        guard (2...16).contains(count), sources.count == count,
              Set(sources.map(\.source)) == Set(0..<count),
              sources.allSatisfy({ finding in
                  [finding.originality, finding.cleanliness, finding.detail, finding.alteration,
                   finding.enhancement, finding.degradation, finding.consistency].allSatisfy { (0...100).contains($0) }
                  && finding.defects.count <= ArenaDefect.allCases.count
                  && Set(finding.defects).count == finding.defects.count
                  && !finding.evidence.isEmpty && finding.evidence.utf8.count <= 2048
              }) else { throw ModelInstaller.fail("Invalid multi-source PhotoArena assessment") }
    }
}
struct ArenaReviewJob: Codable, Identifiable, Sendable {
    let id: String
    let rect: ArenaPixelRect
    let overview: Bool
    var detailRect: ArenaPixelRect? = nil
}
struct ArenaGroupPayload: Codable, Sendable {
    static let revision = "super-merger-review-v1"
    let revision: String
    let sourceIDs: [String]
    let images: [Data]
    let context: String
    func validate() throws {
        guard revision == Self.revision, (2...16).contains(sourceIDs.count), Set(sourceIDs).count == sourceIDs.count,
              sourceIDs.allSatisfy({ !$0.isEmpty && $0.utf8.count < 1024 }), (1...2).contains(images.count),
              images.allSatisfy({ !$0.isEmpty && $0.count <= 16*1024*1024 }), context.utf8.count <= 24_000 else {
            throw ModelInstaller.fail("Invalid PhotoArena group evidence")
        }
    }
}
struct ArenaReviewRecord: Codable, Identifiable, Sendable {
    var id: String { job.id }
    let job: ArenaReviewJob
    let sourceIDs: [String]
    let cacheKey: String
    let result: ArenaJudgeResult
    func finding(_ sourceID: String) -> ArenaCandidateFinding? {
        guard let index = sourceIDs.firstIndex(of: sourceID) else { return nil }
        return result.review?.sources.first { $0.source == index }
    }
}
struct ArenaCandidateAssessment: Codable, Identifiable, Sendable {
    var id: String { sourceID }
    let sourceID: String
    let originality: Double?
    let cleanliness: Double?
    let detail: Double?
    let alteration: Double?
    let enhancement: Double?
    let degradation: Double?
    let consistency: Double?
    let coverage: Double
    let confidence: ArenaReviewConfidence
    let defects: [ArenaDefect]
    let reasons: [String]
}
enum ArenaWorkflowPhase: String, Codable, Sendable {
    case empty, ingesting, aligning, reviewing, planning, verifying, ready, partial, cancelled, failed
}
enum ArenaResultKind: String, Codable, Sendable { case original, composite, unresolved }
struct ArenaRecoveryPlan: Codable, Sendable {
    let kind: ArenaResultKind
    let bestSourceID: String?
    let backboneID: String
    let rankings: [ArenaCandidateAssessment]
    let selections: [ArenaSourceSelection]
    let pendingRegionIDs: [String]
    let unresolvedRegionIDs: [String]
    let reasons: [String]
    var verification: [ArenaPatchVerification]? = nil
}

/// Uses the shared source/region model directly. No tournament or pairwise win aggregation.
enum ArenaRecoveryPlanner {
    static func needsReview(_ region: ArenaRegion) -> Bool {
        if region.maskDeferred == true { return true }
        if let mask = region.changeMask { return mask.pixelCount > 0 }
        return region.measurements.contains { $0.difference > 0.001 }
    }
    static func jobs(_ analysis: ArenaAnalysis) -> [ArenaReviewJob] {
        guard analysis.manualReviewReason == nil else { return [] }
        return [ArenaReviewJob(id: "overview", rect: ArenaPixelRect(CGRect(x: 0, y: 0, width: analysis.width, height: analysis.height)), overview: true)] +
        analysis.regions.filter(needsReview).sorted {
            let a = $0.measurements.map(\.difference).max() ?? 0, b = $1.measurements.map(\.difference).max() ?? 0
            return a == b ? $0.id < $1.id : a > b
        }.map { region in
            let canvas = CGRect(x: 0, y: 0, width: analysis.width, height: analysis.height)
            let focus = region.changeMask?.bounds.insetBy(dx:-12,dy:-12).intersection(canvas).integral
            let detail = focus.flatMap { !$0.isNull && !$0.isEmpty ? ArenaPixelRect($0) : nil }
            return ArenaReviewJob(id:region.id,rect:region.rect,overview:false,detailRect:detail)
        }
    }
    static func plan(_ analysis: ArenaAnalysis, reviews: [ArenaReviewRecord], overrides: [String: String] = [:], preferredSource: String? = nil) -> ArenaRecoveryPlan {
        let latest = analysis.manualReviewReason == nil
            ? Dictionary(reviews.map { ($0.id, $0) }, uniquingKeysWith: { _, new in new }) : [:]
        let regions = analysis.regions.filter(needsReview)
        let allJobs = jobs(analysis)
        let pending = allJobs.filter { latest[$0.id] == nil }.map(\.id)
        let equivalent = Dictionary(grouping: analysis.sources) { $0.fileSHA256 ?? $0.id }
        let groups = Dictionary(uniqueKeysWithValues: analysis.sources.map { ($0.id,equivalent[$0.fileSHA256 ?? $0.id]!.map(\.id)) })
        func finding(_ record: ArenaReviewRecord?, _ id: String) -> ArenaCandidateFinding? {
            // Identical bytes cannot have differing photographic evidence. Use the more cautious
            // observed assessment and expose the untouched model findings separately for audit.
            (groups[id] ?? [id]).compactMap { record?.finding($0) }.min {
                if $0.confidence != $1.confidence { return $0.confidence == .low || ($0.confidence == .medium && $1.confidence == .high) }
                if $0.cleanliness != $1.cleanliness { return $0.cleanliness < $1.cleanliness }
                return $0.originality < $1.originality
            }
        }
        let area = Double(analysis.width)*Double(analysis.height)
        let changedArea = max(1, regions.reduce(0) { $0 + $1.rect.width*$1.rect.height })
        let ranks = analysis.sources.map { source -> ArenaCandidateAssessment in
            var sums = [Double](repeating: 0, count: 7), weight = 0.0, reviewed = 0.0
            var defects: Set<ArenaDefect> = [], reasons: [String] = [], low = false
            for job in allJobs {
                guard let finding = finding(latest[job.id],source.id) else { continue }
                let w = job.overview ? 0.25 : 0.75*job.rect.width*job.rect.height/changedArea
                let values = [finding.originality, finding.cleanliness, finding.detail, finding.alteration, finding.enhancement, finding.degradation, finding.consistency]
                for i in values.indices { sums[i] += Double(values[i])*w }
                weight += w
                if !job.overview { reviewed += job.rect.width*job.rect.height }
                defects.formUnion(finding.defects); low = low || finding.confidence == .low
                if reasons.count < 8 { reasons.append("\(job.id): \(finding.evidence)") }
            }
            let coverage = regions.isEmpty ? (latest["overview"] == nil ? 0.0 : 1.0) : min(1, reviewed/changedArea)
            let values = sums.map { weight > 0 ? $0/weight : nil }
            if !source.accepted { reasons.append("Alignment rejected; source is assessed independently and cannot supply aligned patches.") }
            if (groups[source.id]?.count ?? 0) > 1 { reasons.append("Byte-identical candidates share the more cautious assessment; repeated copies add no independent support.") }
            return ArenaCandidateAssessment(sourceID: source.id, originality: values[0], cleanliness: values[1], detail: values[2],
                alteration: values[3], enhancement: values[4], degradation: values[5], consistency: values[6],
                coverage: coverage, confidence: low || !pending.isEmpty || weight == 0 ? .low : .medium,
                defects: defects.sorted { $0.rawValue < $1.rawValue }, reasons: reasons)
        }
        let sourcesByID = Dictionary(uniqueKeysWithValues: analysis.sources.map { ($0.id, $0) })
        let originalityMargin = 3.0
        func strongest(_ candidates: [ArenaCandidateAssessment]) -> ArenaCandidateAssessment? {
            candidates.max {
                let a = $0.originality ?? -1, b = $1.originality ?? -1
                return a == b ? $0.sourceID > $1.sourceID : a < b
            }
        }
        func prefersJPEG(_ jpeg: ArenaCandidateAssessment, over webP: ArenaCandidateAssessment) -> Bool {
            sourcesByID[jpeg.sourceID]?.isJPEG == true && sourcesByID[webP.sourceID]?.isWebP == true
                && (webP.originality ?? -1) - (jpeg.originality ?? -1) < originalityMargin
        }
        func preferredWholeSource(_ candidates: [ArenaCandidateAssessment]) -> ArenaCandidateAssessment? {
            guard let leader = strongest(candidates) else { return nil }
            return strongest(candidates.filter { prefersJPEG($0, over: leader) }) ?? leader
        }
        func uniqueBest(_ candidates: [ArenaCandidateAssessment]) -> String? {
            guard let top = preferredWholeSource(candidates), let score = top.originality,
                  !candidates.contains(where: {
                      $0.sourceID != top.sourceID && score - ($0.originality ?? -1) < originalityMargin
                          && !prefersJPEG(top, over: $0)
                  }) else { return nil }
            return top.sourceID
        }
        func selectedSource(_ record: ArenaReviewRecord?) -> String? {
            guard let record, record.result.review?.unresolved == false,
                  let ordinal = record.result.review?.selected_source,
                  record.sourceIDs.indices.contains(ordinal) else { return nil }
            return record.sourceIDs[ordinal]
        }
        let overview = latest["overview"]
        let overviewSelection = selectedSource(overview)
        let overviewRank = ranks.first { $0.sourceID == overviewSelection }
        let best = overview?.result.review?.unresolved != nil ? overviewSelection : uniqueBest(ranks)
        let reference = sourcesByID[analysis.referenceID]!
        // A backbone must cover the complete reference canvas. Cropped candidates remain eligible as patches.
        let wholeSources = ranks.filter { rank in
            sourcesByID[rank.sourceID]?.accepted == true && analysis.regions.allSatisfy { region in
                region.measurements.contains { $0.sourceID == rank.sourceID && $0.validFraction >= 0.9999 }
            }
        }
        let requested = wholeSources.first { $0.sourceID == preferredSource && (sourcesByID[$0.sourceID]?.width ?? 0) > 0 }
        func clean(_ f: ArenaCandidateFinding) -> Bool {
            f.usable && f.confidence != .low && f.cleanliness >= 90 && f.alteration <= 10 && f.enhancement <= 10
                && f.defects.allSatisfy { [.natural, .color, .geometry, .compression].contains($0) }
        }
        let eligible = wholeSources.filter { rank in
            pending.isEmpty && (rank.originality ?? 0) >= 60 && (rank.detail ?? 0) >= 45 && (rank.degradation ?? 100) <= 40
                && allJobs.allSatisfy {
                    let record = latest[$0.id]
                    let decisionSupportsSource: Bool
                    if $0.overview {
                        decisionSupportsSource = record?.result.review?.unresolved == nil || overviewSelection == rank.sourceID
                            || overviewRank.map { prefersJPEG(rank, over: $0) } == true
                    } else {
                        decisionSupportsSource = record?.result.review?.unresolved != true
                    }
                    return decisionSupportsSource && finding(record, rank.sourceID).map(clean) == true
                }
        }
        let single = uniqueBest(eligible)
        let direct = single != nil && overrides.isEmpty && preferredSource == nil
        // A region override keeps the clean whole-source winner as the base.
        let backboneID = requested?.sourceID ?? single ?? preferredWholeSource(wholeSources)?.sourceID
        let backbone = backboneID.flatMap { sourcesByID[$0] } ?? reference
        let base = ArenaSourceSelection(regionID: "backbone", sourceID: backbone.id,
            rect: ArenaPixelRect(CGRect(x: 0, y: 0, width: analysis.width, height: analysis.height)), unresolved: true,
            reasons: ["Whole-source backbone; model assessment does not establish provenance."], manual: false)
        var selections = [base], unresolved: [String] = []
        for region in regions {
            if direct { unresolved.append(region.id); continue }
            if let manual = overrides[region.id], manual != backbone.id,
               let mask = region.changeMask, mask.pixelCount > 0,
               analysis.sources.contains(where: { $0.id == manual && $0.accepted }),
               region.measurements.contains(where: { $0.sourceID == manual && $0.validFraction >= 0.9999 }) {
                selections.append(ArenaSourceSelection(regionID:region.id,sourceID:manual,rect:region.rect,unresolved:true,
                    reasons:["Explicit region source override; model evidence may remain pending."],manual:true,mask:mask))
                unresolved.append(region.id); continue
            }
            guard let review = latest[region.id], let baseFinding = finding(review,backbone.id) else { unresolved.append(region.id); continue }
            let manual = overrides[region.id]
            let candidate = analysis.sources.filter { source in
                source.accepted && source.id != backbone.id && region.measurements.contains { $0.sourceID == source.id && $0.validFraction >= 0.9999 }
            }.filter { source in
                if let manual { return source.id == manual }
                if review.result.review?.unresolved != nil && selectedSource(review) != source.id { return false }
                guard let finding = finding(review,source.id) else { return false }
                return !clean(baseFinding) && clean(finding)
                    && finding.cleanliness >= baseFinding.cleanliness + 10
            }.sorted {
                let a = review.finding($0.id)?.originality ?? 0, b = review.finding($1.id)?.originality ?? 0
                return a == b ? $0.id < $1.id : a > b
            }.first
            if let candidate, let mask = region.changeMask, mask.pixelCount > 0 {
                selections.append(ArenaSourceSelection(regionID: region.id, sourceID: candidate.id, rect: region.rect,
                    unresolved: true, reasons: [manual == nil ? "Model-supported clean replacement; registration and output pixels must be verified." : "Explicit region source override.", review.finding(candidate.id)?.evidence ?? ""],
                    manual: manual != nil, mask: mask))
            }
            // Model judgments are uncertain; all changed regions retain that uncertainty in the manifest.
            unresolved.append(region.id)
        }
        let kind: ArenaResultKind = direct ? .original : selections.count > 1 ? .composite : .unresolved
        let reason = analysis.manualReviewReason ?? (kind == .original ? "One candidate satisfies the reviewed evidence; preserve its original file bytes." :
            kind == .composite ? "No single clean candidate was established; recover only supported source patches." :
            "Evidence is insufficient for an automatic clean recovery; retain the best available source with unresolved findings.")
        return ArenaRecoveryPlan(kind: kind, bestSourceID: direct ? single : best, backboneID: backbone.id, rankings: ranks,
            selections: selections, pendingRegionIDs: pending, unresolvedRegionIDs: unresolved,
            reasons: [reason,
                "Scores compare supplied candidates and are not calibrated probabilities.",
                "\(pending.count) review tasks pending; \(Int(area)) reference pixels measured."])
    }
}
