import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

private func recoveryFinding(_ source: Int, originality: Int = 90, clean: Bool = true, detail: Int = 70,
                             confidence: ArenaReviewConfidence = .medium) -> ArenaCandidateFinding {
    ArenaCandidateFinding(source: source, originality: originality, cleanliness: clean ? 98 : 20,
        detail: detail, alteration: clean ? 0 : 50, enhancement: 0, degradation: 0, consistency: 90,
        usable: clean, confidence: confidence, defects: clean ? [] : [.watermark], evidence: clean ? "Test assessment: no observed overlay." : "Test assessment: visible synthetic overlay.")
}
private func recoveryRecord(_ analysis: ArenaAnalysis, job: ArenaReviewJob, findings: [ArenaCandidateFinding]) -> ArenaReviewRecord {
    ArenaReviewRecord(job: job, sourceIDs: analysis.sources.map(\.id), cacheKey: "fixture",
        result: ArenaJudgeResult(judgment: ArenaJudgment(watermark: .unknown, authenticity: .unknown, detail: .unknown,
            quality: .unknown, consistency: .unknown, recommended_source: .unknown, uncertain: true, evidence: ["Injected contract fixture; not measured model accuracy"]),
            tokens: 0, prefillSeconds: 0, decodeSeconds: 0, physicalFootprintBytes: 0, compressedBytes: 0,
            review: ArenaGroupAssessment(sources: findings)))
}
private func recoveryFixture(_ root: URL, count: Int = 3) throws -> ArenaAnalysis {
    let width = 32, height = 24
    let sources = try (0..<count).map { index -> ArenaAnalyzedSource in
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width*4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(red: Double(index+1)/4, green: 0.25, blue: 0.35, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let url = root.appendingPathComponent("source-\(index).png")
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        return ArenaAnalyzedSource(id: "s\(index)", url: url, width: width, height: height,
            transform: ArenaTransform(), alignmentConfidence: 1, accepted: true, issues: [])
    }
    let region = ArenaRegion(id: "patch", rect: ArenaPixelRect(CGRect(x: 0,y: 0,width: width,height: height)),
        measurements: sources.map { ArenaRegionMeasurement(sourceID: $0.id, difference: 0.1, sharpness: 0.1, clipping: 0,
            validFraction: 1, textCandidateFraction: 0, evidence: []) }, unresolved: true, evidence: [],
        changeMask: ArenaPixelMask(runs: [ArenaMaskRun(y: 3,x: 5,length: 4), ArenaMaskRun(y: 4,x: 5,length: 2), ArenaMaskRun(y: 18,x: 25,length: 1)]))
    return ArenaAnalysis(referenceID: sources.last!.id, width: width, height: height, sources: sources, regions: [region], warnings: [], elapsedSeconds: 0)
}
func registerArenaRecoveryTests() {
    Registry.shared.add(suite: "PhotoArena recovery", TestCase(name: "N-source contract rejects missing duplicated and invalid assessments", knownBug: nil) {
        try ArenaGroupAssessment(sources: (0..<16).map { recoveryFinding($0) }).validate(count: 16)
        for values in [[recoveryFinding(0),recoveryFinding(0)], [recoveryFinding(0)], [recoveryFinding(0,originality: 101),recoveryFinding(1)]] {
            do { try ArenaGroupAssessment(sources: values).validate(count: 2); check(false,"invalid source assessments rejected") } catch { }
        }
        let mask = ArenaPixelMask(runs: [ArenaMaskRun(y: 0,x: 0,length: 3),ArenaMaskRun(y: 1,x: 1,length: 1),ArenaMaskRun(y: 7,x: 4,length: 2)])
        checkEqual(mask.components.count, 2, "precise connected components")
        checkEqual(mask.pixelCount, 6, "all pixels retained")
        check(!mask.contains(x: 0,y: 1), "mask is not a bounding rectangle")
    })
    Registry.shared.add(suite: "PhotoArena recovery", TestCase(name: "one best source from three preserves original bytes and metadata container", knownBug: nil) {
        let root = try Temporary.directory("arena-best-source")
        let analysis = try recoveryFixture(root)
        let jobs = ArenaRecoveryPlanner.jobs(analysis)
        checkEqual(jobs.count,2,"overview plus one multi-source region, independent of pair count")
        let records = jobs.map { recoveryRecord(analysis, job: $0, findings: [recoveryFinding(0,originality: 30,clean: false,detail: 100),recoveryFinding(1,originality: 95),recoveryFinding(2,originality: 50,clean: false)]) }
        let plan = ArenaRecoveryPlanner.plan(analysis, reviews: records)
        checkEqual(plan.bestSourceID,"s1","originality outranks apparent sharpness")
        checkEqual(plan.rankings.map(\.sourceID), analysis.sources.map(\.id), "the rest retain input order, not score order")
        checkEqual(plan.kind,.original,"single clean candidate avoids composite")
        check(plan.pendingRegionIDs.isEmpty,"all meaningful regions reviewed")
        let directory = root.appendingPathComponent("export")
        let manifest = try ArenaReconstruction.export(analysis,selections: plan.selections,to: directory,plan: plan)
        checkEqual(try Data(contentsOf: directory.appendingPathComponent(manifest.imageFile)),try Data(contentsOf: analysis.sources[1].url),"byte-for-byte original container preservation")
        check(manifest.selections.isEmpty,"reference-space masks not misreported as native-file operations")
        checkEqual(manifest.pixelOperations?.count,3,"operation and coordinate-system provenance retained")
    })
    Registry.shared.add(suite: "PhotoArena recovery", TestCase(name: "ties shared defects low confidence and deferred regions do not force an original", knownBug: nil) {
        let root = try Temporary.directory("arena-abstain")
        let analysis = try recoveryFixture(root)
        for clean in [false,true] {
            let records = ArenaRecoveryPlanner.jobs(analysis).map { recoveryRecord(analysis,job:$0,findings:(0..<3).map { recoveryFinding($0,clean:clean) }) }
            let plan = ArenaRecoveryPlanner.plan(analysis,reviews:records)
            checkEqual(plan.bestSourceID,nil,"equivalent evidence is a tie")
            checkEqual(plan.kind,.unresolved,"shared damage or equivalence does not cause invented recovery")
            checkEqual(plan.selections.count,1,"no unsupported patch")
        }
        let overview = recoveryRecord(analysis,job:ArenaRecoveryPlanner.jobs(analysis)[0],findings:[recoveryFinding(0,originality:99),recoveryFinding(1,originality:60),recoveryFinding(2,originality:30)])
        let partial = ArenaRecoveryPlanner.plan(analysis,reviews:[overview])
        checkEqual(partial.pendingRegionIDs,["patch"],"unreviewed region explicitly pending")
        checkEqual(partial.kind,.unresolved,"overview alone cannot establish clean source")
        check(partial.rankings.allSatisfy { $0.confidence == .low },"partial coverage lowers confidence")
        let low = ArenaRecoveryPlanner.jobs(analysis).map { recoveryRecord(analysis,job:$0,findings:[recoveryFinding(0,originality:99,confidence:.low),recoveryFinding(1,originality:50),recoveryFinding(2,originality:30)]) }
        checkEqual(ArenaRecoveryPlanner.plan(analysis,reviews:low).kind,.unresolved,"low confidence prevents automatic original claim")
    })
    Registry.shared.add(suite: "PhotoArena recovery", TestCase(name: "identical inputs cannot acquire different scores from model position", knownBug: nil) {
        let root = try Temporary.directory("arena-equivalent")
        let fixture = try recoveryFixture(root,count:2)
        try FileManager.default.removeItem(at:fixture.sources[1].url)
        try FileManager.default.copyItem(at:fixture.sources[0].url,to:fixture.sources[1].url)
        let analysis = try ArenaForensics.analyzeFiles(fixture.sources.map { ArenaSource(id:$0.id,url:$0.url,seconds:0) })
        checkEqual(analysis.sources.filter(\.accepted).count,2,"exact bytes establish registration even without texture")
        checkEqual(ArenaRecoveryPlanner.jobs(analysis).count,1,"no difference regions invented for duplicate files")
        let record = recoveryRecord(analysis,job:ArenaRecoveryPlanner.jobs(analysis)[0],findings:[recoveryFinding(0,originality:99),recoveryFinding(1,originality:40,clean:false)])
        let plan = ArenaRecoveryPlanner.plan(analysis,reviews:[record])
        checkEqual(plan.rankings[0].originality,plan.rankings[1].originality,"identical candidates share cautious assessment")
        checkEqual(plan.bestSourceID,nil,"no arbitrary winner for identical candidates")
        checkEqual(plan.selections.count,1,"identical inputs never cause compositing")
    })
    Registry.shared.add(suite: "PhotoArena recovery", TestCase(name: "masked recovery preserves other pixels source map and output-versus-every-source truth", knownBug: nil) {
        let root = try Temporary.directory("arena-mask-recovery")
        let analysis = try recoveryFixture(root)
        let overview = recoveryRecord(analysis,job:ArenaRecoveryPlanner.jobs(analysis)[0],findings:[recoveryFinding(0,originality:50,clean:false),recoveryFinding(1,originality:60,clean:false),recoveryFinding(2,originality:99,clean:false)])
        let regional = recoveryRecord(analysis,job:ArenaRecoveryPlanner.jobs(analysis)[1],findings:[recoveryFinding(0,originality:95),recoveryFinding(1,originality:40,clean:false),recoveryFinding(2,originality:80,clean:false)])
        let plan = ArenaRecoveryPlanner.plan(analysis,reviews:[overview,regional],preferredSource:"s2")
        checkEqual(plan.kind,.composite,"supported local recovery")
        checkEqual(plan.selections.map(\.sourceID),["s2","s0"],"backbone followed by clean patch despite source-list order")
        let directory = root.appendingPathComponent("export")
        let manifest = try ArenaReconstruction.export(analysis,selections:plan.selections,to:directory,plan:plan)
        let container = CGImageSourceCreateWithURL(directory.appendingPathComponent(manifest.imageFile) as CFURL,nil)!
        let exported = CGImageSourceCreateImageAtIndex(container,0,nil)!
        let rect = CGRect(x:0,y:0,width:analysis.width,height:analysis.height)
        let replay = try ArenaOutputInspector.tile(analysis,selections:plan.selections,rect:rect)
        checkEqual(ArenaOutputInspector.pixels(exported),ArenaOutputInspector.pixels(replay),"inspector and exported output pixels exactly match")
        let source = try ArenaForensics.crop(analysis.sources[2],rect:rect)
        let a = ArenaOutputInspector.pixels(exported), b = ArenaOutputInspector.pixels(source)
        var changed = 0
        for p in 0..<analysis.width*analysis.height where a[p*4] != b[p*4] { changed += 1 }
        checkEqual(changed,7,"only exact seven mask pixels replaced; surroundings untouched")
        for source in analysis.sources {
            let diff = try ArenaOutputInspector.compare(analysis,selections:plan.selections,sourceID:source.id)
            checkEqual(diff.0.uncoveredPixels,0,"complete coverage")
            checkEqual(diff.0.changedPixels,source.id == "s2" ? 7 : source.id == "s0" ? analysis.width*analysis.height-7 : analysis.width*analysis.height,"independently recomputed source-specific mask")
            let red = ArenaOutputInspector.pixels(diff.1)
            checkEqual(stride(from:3,to:red.count,by:4).filter { red[$0] > 0 }.count,diff.0.changedPixels,"red mask matches actual output differences")
        }
        let mapContainer = CGImageSourceCreateWithURL(directory.appendingPathComponent(manifest.sourceMapFile) as CFURL,nil)!
        let map = CGImageSourceCreateImageAtIndex(mapContainer,0,nil)!
        let mapBytes = Array(map.dataProvider!.data! as Data)
        var patchPixels = 0
        for y in 0..<map.height { for x in 0..<map.width where mapBytes[y*map.bytesPerRow+x] == 1 { patchPixels += 1 } }
        checkEqual(patchPixels,7,"source index map exactly follows nonrectangular patch")
        check(plan.selections.allSatisfy(\.unresolved),"model assessment never proves provenance")
        let rejected = try ArenaMergeVerifier.verify(analysis,plan:plan)
        checkEqual(rejected.kind,.unresolved,"inconsistent surrounding colors reject an automatic seam")
        checkEqual(rejected.verification?.first?.accepted,false,"measured rejection retained")
    })
    Registry.shared.add(suite: "PhotoArena recovery", TestCase(name: "prepared evidence keys bind all candidates ordering geometry and masks", knownBug: nil) {
        let root = try Temporary.directory("arena-group-evidence")
        let analysis = try recoveryFixture(root)
        let job = ArenaRecoveryPlanner.jobs(analysis)[1]
        let payload = try ArenaGroupEvidence.prepare(analysis,job:job)
        try payload.validate()
        checkEqual(payload.sourceIDs,["s0","s1","s2"],"every candidate included in one shared evidence package")
        let key = try ArenaGroupEvidence.key(payload,job:job)
        let reversed = ArenaGroupPayload(revision:payload.revision,sourceIDs:payload.sourceIDs.reversed(),images:payload.images,context:payload.context)
        check(key != (try ArenaGroupEvidence.key(reversed,job:job)),"candidate identity mapping participates in cache")
        checkEqual(payload.images.count,2,"context and detail montages, not source pair")
        let worker = root.appendingPathComponent("worker.sh")
        let result = recoveryRecord(analysis,job:job,findings:(0..<3).map { recoveryFinding($0) }).result
        let encoded = String(decoding:try JSONEncoder().encode(result),as:UTF8.self)
        let script = "#!/bin/sh\nprintf '%s\\n' '{\"ready\":true,\"groupReview\":\"\(ArenaGroupPayload.revision)\"}'\nwhile IFS= read -r request; do\nprintf '%s\\n' \"$request\" >> \"${1%/*}/requests.jsonl\"\nprintf '%s\\n' '"+encoded+"'\ndone\n"
        try script.write(to:worker,atomically:true,encoding:.utf8)
        try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:worker.path)
        let jobs = ArenaRecoveryPlanner.jobs(analysis)
        let records = try await ArenaJudge.review(analysis:analysis,jobs:jobs,previous:[],directory:root,executable:worker,onResult:{ _ in })
        checkEqual(records.count,2,"one process serves overview and region")
        let lines = try String(contentsOf:root.appendingPathComponent("requests.jsonl"),encoding:.utf8).split(separator:"\n")
        checkEqual(lines.count,2,"one N-source request per evidence package")
        for line in lines {
            let json = try JSONSerialization.jsonObject(with:Data(line.utf8)) as! [String:Any]
            checkEqual(json["reviewCount"] as? Int,3,"native worker receives N-source contract")
        }
        _ = try await ArenaJudge.review(analysis:analysis,jobs:jobs,previous:records,directory:root,executable:root.appendingPathComponent("must-not-run"),onResult:{ _ in })
        checkEqual(try String(contentsOf:root.appendingPathComponent("requests.jsonl"),encoding:.utf8).split(separator:"\n").count,2,"validated scalar cache avoids starting a worker")
    })
    Registry.shared.add(suite: "PhotoArena recovery", TestCase(name: "real native N-source forensic review and timing (opt-in)", knownBug: nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_ARENA_GROUP_BENCHMARK"] == "1" else { throw SkipTest(reason:"Opt in for actual native multi-source inference") }
        let root = try Temporary.directory("arena-native-group")
        let repository = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
        let original = repository.appendingPathComponent("Tests/Fixtures/RealESRGAN/nasa-collins-896x1134.jpg")
        let input = CGImageSourceCreateWithURL(original as CFURL,nil)!
        let image = CGImageSourceCreateImageAtIndex(input,0,nil)!
        var sources: [ArenaSource] = []
        for i in 0..<3 {
            let context = CGContext(data:nil,width:image.width,height:image.height,bitsPerComponent:8,bytesPerRow:image.width*4,
                space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image,in:CGRect(x:0,y:0,width:image.width,height:image.height))
            if i > 0 {
                context.setFillColor(red:1,green:0,blue:0,alpha:0.9)
                context.fill(CGRect(x:200,y:180,width:100,height:70))
            }
            let url = root.appendingPathComponent("source-\(i).png")
            let destination = CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil)!
            CGImageDestinationAddImage(destination,context.makeImage()!,nil)
            guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
            sources.append(ArenaSource(id:"s\(i)",url:url,seconds:0))
        }
        let start = Date(), sampler = RealESRGANMemorySampler()
        let analysis = try await ArenaForensics.analyze(sources,tileSize:256)
        let jobs = ArenaRecoveryPlanner.jobs(analysis)
        let records = try await ArenaJudge.review(analysis:analysis,jobs:jobs,previous:[],directory:repository.appendingPathComponent("models/photoarena"),
            executable:repository.appendingPathComponent(".native-runtime/waifu-photoarena"),onResult:{ _ in })
        checkEqual(records.count,jobs.count,"real model returned all requested packages")
        check(records.allSatisfy { $0.result.review?.sources.count == 3 },"real model reviewed all candidates")
        let mergeStart = Date()
        let plan = try ArenaMergeVerifier.verify(analysis,plan:ArenaRecoveryPlanner.plan(analysis,reviews:records))
        let mergeSeconds = Date().timeIntervalSince(mergeStart)
        let exportStart = Date()
        let directory = root.appendingPathComponent("result")
        let manifest = try ArenaReconstruction.export(analysis,selections:plan.selections,to:directory,plan:plan)
        let exportSeconds = Date().timeIntervalSince(exportStart)
        let verificationStart = Date()
        let differences = try analysis.sources.filter(\.accepted).map {
            try ArenaOutputInspector.compare(analysis,selections:plan.selections,sourceID:$0.id,maxPixelSize:256).0
        }
        let verificationSeconds = Date().timeIntervalSince(verificationStart)
        checkEqual(differences.count,3,"selected output inspected against every registered source")
        checkEqual(plan.rankings.first(where: { $0.sourceID == "s1" })?.originality,
            plan.rankings.first(where: { $0.sourceID == "s2" })?.originality,"duplicate candidates evaluated consistently")
        struct Report: Encodable {
            let scope: String; let elapsedSeconds: Double; let preprocessingSeconds: Double; let clientPeakPhysicalFootprintBytes: UInt64
            let mergeSeconds: Double; let exportSeconds: Double; let verificationSeconds: Double
            let outputFile: String; let differences: [ArenaOutputDifference]
            let knownCleanSourceID: String; let outputMatchesKnownCleanFile: Bool; let modelRecommendedKnownCleanSource: Bool
            let records: [ArenaReviewRecord]; let plan: ArenaRecoveryPlan
        }
        let report = Report(scope:"Actual 3-source optimized Swift + native MiniCPM end-to-end workflow on NASA photograph with conventional red overlays. Client footprint is sampled; native physical lifetime peak, Metal allocation, compressed bytes and raw swap-in ledger are worker measurements. No AI-edit accuracy calibration or whole GUI app measurement.",
            elapsedSeconds:Date().timeIntervalSince(start),preprocessingSeconds:analysis.elapsedSeconds,clientPeakPhysicalFootprintBytes:sampler.peakBytes,
            mergeSeconds:mergeSeconds,exportSeconds:exportSeconds,verificationSeconds:verificationSeconds,outputFile:manifest.imageFile,differences:differences,
            knownCleanSourceID:"s0",outputMatchesKnownCleanFile:try Data(contentsOf:directory.appendingPathComponent(manifest.imageFile)) == Data(contentsOf:sources[0].url),
            modelRecommendedKnownCleanSource:plan.bestSourceID == "s0",records:records,plan:plan)
        if let path = ProcessInfo.processInfo.environment["WAIFU_ARENA_GROUP_REPORT"] {
            let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
            try encoder.encode(report).write(to:URL(fileURLWithPath:path),options:.atomic)
        }
    })
}
