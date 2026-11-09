import CoreGraphics
import Foundation

struct ArenaPatchVerification: Codable, Sendable {
    let regionID: String
    let sourceID: String
    let accepted: Bool
    let contextSamples: Int
    let contextMeanAbsoluteError: Double
    let reason: String
}

/// Rejects automatic patches whose surrounding real source pixels disagree.
/// Hard masks preserve source samples; no hidden blending or synthesized repair.
enum ArenaMergeVerifier {
    static func verify(_ analysis: ArenaAnalysis, plan: ArenaRecoveryPlan) throws -> ArenaRecoveryPlan {
        guard plan.kind == .composite, let base = analysis.sources.first(where: { $0.id == plan.backboneID }) else { return plan }
        var checks: [ArenaPatchVerification] = []
        var kept = plan.selections.filter { $0.regionID == "backbone" }
        let canvas = CGRect(x: 0,y: 0,width: analysis.width,height: analysis.height)
        for selection in plan.selections where selection.regionID != "backbone" {
            try Task.checkCancellation()
            guard let source = analysis.sources.first(where: { $0.id == selection.sourceID && $0.accepted }),
                  let mask = selection.mask else { continue }
            let bounds = mask.bounds.insetBy(dx: -3,dy: -3).intersection(canvas).integral
            let a = ArenaOutputInspector.pixels(try ArenaForensics.crop(base,rect:bounds))
            let b = ArenaOutputInspector.pixels(try ArenaForensics.crop(source,rect:bounds))
            let w = Int(bounds.width), h = Int(bounds.height)
            var selected = [Bool](repeating:false,count:w*h), invalidCoverage = false
            for nearby in plan.selections where nearby.sourceID == source.id && nearby.rect.cgRect.intersects(bounds) {
                guard let nearbyMask = nearby.mask else { continue }
                for run in nearbyMask.runs where run.y >= Int(bounds.minY) && run.y < Int(bounds.maxY) {
                    let start = max(run.x,Int(bounds.minX)), end = min(run.x+run.length,Int(bounds.maxX))
                    guard start < end else { continue }
                    for x in start..<end {
                        let i = (run.y-Int(bounds.minY))*w+x-Int(bounds.minX)
                        selected[i] = true
                        if nearby.regionID == selection.regionID && b[i*4+3] <= 250 { invalidCoverage = true }
                    }
                }
            }
            var total = 0.0, samples = 0
            for y in 0..<h {
                for x in 0..<w where !selected[y*w+x] {
                    // Evaluate a one-pixel exterior ring, including holes in the mask.
                    let adjacent = (x > 0 && selected[y*w+x-1]) || (x+1 < w && selected[y*w+x+1])
                        || (y > 0 && selected[(y-1)*w+x]) || (y+1 < h && selected[(y+1)*w+x])
                    let i = (y*w+x)*4
                    guard adjacent, a[i+3] > 250, b[i+3] > 250 else { continue }
                    total += (0..<3).reduce(0.0) { $0 + abs(Double(a[i+$1])-Double(b[i+$1])) }; samples += 1
                }
            }
            let error = total/Double(max(1,samples)*3)
            let accepted = !invalidCoverage && (selection.manual || (samples >= 8 && error <= 8))
            let reason = invalidCoverage ? "Patch rejected because donor pixels do not cover the full selected mask." :
                selection.manual ? "Explicit manual patch retained; measured context mismatch remains disclosed." :
                accepted ? "Exterior source pixels agree within 8/255 mean absolute channel error; no blending was required." :
                "Automatic patch rejected: insufficient surrounding correspondence or a visible context mismatch."
            checks.append(ArenaPatchVerification(regionID:selection.regionID,sourceID:source.id,accepted:accepted,
                contextSamples:samples,contextMeanAbsoluteError:error,reason:reason))
            if accepted { kept.append(selection) }
        }
        return ArenaRecoveryPlan(kind:kept.count > 1 ? .composite : .unresolved,bestSourceID:plan.bestSourceID,backboneID:plan.backboneID,
            rankings:plan.rankings,selections:kept,pendingRegionIDs:plan.pendingRegionIDs,unresolvedRegionIDs:plan.unresolvedRegionIDs,
            reasons:plan.reasons + checks.filter { !$0.accepted }.map(\.reason),verification:checks)
    }
}
