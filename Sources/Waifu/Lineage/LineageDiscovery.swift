import Foundation
import CoreGraphics
import CoreImage

/// Reuses CrystalOrb's persisted Vision index for retrieval, then verifies actual
/// pixel correspondence. Vision distance alone never creates a family or an edge.
actor LineageDiscovery {
    private let search = CrystalOrbSearchService()
    private var previews: [(String,CGImage)] = []
    private var verified: [String:LineageRelation] = [:]
    private var rejected = Set<String>()

    static func indexFrame(_ photo: LineagePhoto, existing: [AnalyzedFrame]) -> AnalyzedFrame {
        existing.first { abs($0.timeSeconds-photo.offset) < 0.000001 } ?? AnalyzedFrame(
            sampleIndex:PhotosMedia.frameIndex(seconds:photo.offset), timeSeconds:photo.offset,
            artScore:0, faceScore:nil, faceCount:0, faceAreaFraction:nil, featurePrint:nil,
            thumbnailKey:photo.contentIdentity, thumbnailAspectRatio:Double(photo.width)/Double(max(1,photo.height)))
    }
    private func preview(_ photo: LineagePhoto) async throws -> CGImage {
        if let hit = previews.first(where: { $0.0 == photo.contentIdentity }) { return hit.1 }
        let image = try await PhotosMedia.shared.analysisImage(photo.url,seconds:0)
        try Task.checkCancellation()
        previews.append((photo.contentIdentity,image))
        if previews.count > 12 { previews.removeFirst() }
        return image
    }
    func discover(photos: [LineagePhoto], frames: [AnalyzedFrame], source: URL, generation: UUID,
                  update: @Sendable ([LineageRelation],Int,Int,String?) async -> Void) async throws -> [LineageRelation] {
        let entries = photos.map { Self.indexFrame($0,existing:frames) }
        let byFrame = Dictionary(uniqueKeysWithValues:zip(entries,photos).map { ($0.0.id,$0.1) })
        var links: [LineageRelation] = [], pairs = Set<String>()
        var unavailable = 0
        let identities = Set(photos.map(\.contentIdentity))
        previews.removeAll { !identities.contains($0.0) }
        // Bound retained pair cache across albums; keys include both content revisions.
        if verified.count + rejected.count > 20_000 { verified.removeAll(); rejected.removeAll() }
        for (position, photo) in photos.enumerated() {
            try Task.checkCancellation()
            do {
                let image = try await preview(photo)
                let matches = try await search.search(query:image, queryID:UUID(), frames:entries,
                    sourceURL:source,generation:generation,resultLimit:9,progress:{ _,_ in })
                let hash = LineageCorrespondence.hash(image)
                for match in matches where match.distance <= 0.22 {
                    guard let other = byFrame[match.id], other.id != photo.id else { continue }
                    let key = [photo.contentIdentity,other.contentIdentity].sorted().joined(separator:"|")
                    guard pairs.insert(key).inserted, !rejected.contains(key) else { continue }
                    if let cached = verified[key] { links.append(cached); continue }
                    let candidate = try await preview(other)
                    // Hashing is an inexpensive veto before spatial registration.
                    guard (hash ^ LineageCorrespondence.hash(candidate)).nonzeroBitCount <= 16 || match.distance <= 0.10 else { rejected.insert(key); continue }
                    if let result = try await LineageCorrespondence.verify(image,candidate,photo:photo,other:other,generation:generation) {
                        verified[key] = result; links.append(result)
                    } else { rejected.insert(key) }
                }
            } catch is CancellationError { throw CancellationError() }
            catch { unavailable += 1 }
            await update(links,position+1,photos.count,unavailable == 0 ? nil : "\(unavailable) photos could not be read. Enable iCloud downloads or retry after they are available locally.")
        }
        return links
    }
}

enum LineageCorrespondence {
    static func hash(_ image: CGImage) -> UInt64 {
        let ci = CIImage(cgImage:image).transformed(by:CGAffineTransform(scaleX:9/CGFloat(image.width),y:8/CGFloat(image.height)))
        let bytes = CrystalOrbAnalysis.bytes(ci,rect:CGRect(x:0,y:0,width:9,height:8))
        var result: UInt64 = 0
        for y in 0..<8 { for x in 0..<8 {
            let a = (y*9+x)*4, b = a+4
            let la = Int(bytes[a])*3 + Int(bytes[a+1])*6 + Int(bytes[a+2])
            let lb = Int(bytes[b])*3 + Int(bytes[b+1])*6 + Int(bytes[b+2])
            if la > lb { result |= UInt64(1) << (y*8+x) }
        } }
        return result
    }
    static func verify(_ a: CGImage,_ b: CGImage,photo:LineagePhoto,other:LineagePhoto,generation:UUID) async throws -> LineageRelation? {
        let geometry = CrystalOrbAlignment(target:CGSize(width:a.width,height:a.height),candidate:CGSize(width:b.width,height:b.height))
        let translation = try await CrystalOrbRegistrar.shared.align(target:a,candidate:b,geometry:geometry,
            query:UUID(),frame:FrameID(sampleIndex:PhotosMedia.frameIndex(seconds:other.offset)),generation:generation)
        let registration:CrystalOrbRegistrationResult
        if let translation,translation.correlation>=0.985 { registration=translation }
        else if let crop=try await LineageCropRegistrar.shared.align(target:a,candidate:b,geometry:geometry,identity:photo.contentIdentity+"|"+other.contentIdentity) { registration=crop }
        else { return nil }
        let alignment = CrystalOrbAlignment(target:geometry.targetSize,candidate:geometry.candidateSize,registration:registration.registration)
        let difference = try await CrystalOrbPairRenderer.shared.render(target:a,candidate:b,alignment:alignment,settings:.init())
        let measures = try await CrystalOrbAnalysis.shared.measure(difference,roi:nil)
        let targetArea=alignment.target.width*alignment.target.height
        let candidateArea=alignment.candidate.width*alignment.candidate.height*registration.registration.scale*registration.registration.scale
        let overlap=CGFloat(measures.full.overlap)
        guard overlap/min(targetArea,candidateArea)>=0.90,overlap/max(targetArea,candidateArea)>=0.35,
              measures.full.mean <= 0.035 else { return nil }
        return .init(a:photo.id,b:other.id,correlation:registration.correlation,meanAbsoluteDifference:measures.full.mean,
                     overlap:measures.full.coverage/100,registration:.init(x:registration.registration.x,y:registration.registration.y,scale:registration.registration.scale),dimensionChange:photo.width != other.width || photo.height != other.height)
    }
}

actor LineageStore {
    static let shared = LineageStore()
    private let root: URL
    init(root: URL? = nil) {
        self.root = root ?? FileManager.default.urls(for:.applicationSupportDirectory,in:.userDomainMask)[0]
            .appendingPathComponent("Waifu/ImageLineage",isDirectory:true)
    }
    func read(album: String) -> LineageInvestigation? {
        let url = root.appendingPathComponent(LineageIdentity.hash(album)+".json")
        guard let values = try? url.resourceValues(forKeys:[.fileSizeKey]), (values.fileSize ?? Int.max) <= 8*1024*1024,
              let data = try? Data(contentsOf:url) else { return nil }
        return try? JSONDecoder().decode(LineageInvestigation.self,from:data)
    }
    func save(_ value: LineageInvestigation,album:String) throws {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        // Maps have their own content/model keyed cache; avoid duplicating huge
        // rasters into an album's navigation document.
        var document = value; document.evidence = [:]
        try JSONEncoder().encode(document).write(to:root.appendingPathComponent(LineageIdentity.hash(album)+".json"),options:.atomic)
    }
}

actor SafireEvidenceCache {
    static let shared=SafireEvidenceCache()
    private let root=FileManager.default.urls(for:.cachesDirectory,in:.userDomainMask)[0].appendingPathComponent("Waifu/ImageLineage",isDirectory:true)
    static func key(photo:LineagePhoto,fingerprint:String) -> String {
        LineageIdentity.hash([photo.contentIdentity,fingerprint,SafireSpatialResult.preprocessingVersion,SafireSpatialResult.runtimeRevision].joined(separator:"\n"))
    }
    func read(photo:LineagePhoto,fingerprint:String) -> SafireSpatialResult? {
        let key=Self.key(photo:photo,fingerprint:fingerprint)
        let current=root.appendingPathComponent(key+".sfm")
        let url=FileManager.default.fileExists(atPath:current.path) ? current : root.appendingPathComponent(key+".json")
        guard let values=try? url.resourceValues(forKeys:[.fileSizeKey]),(values.fileSize ?? Int.max)<=48*1024*1024,
              let data=try? Data(contentsOf:url),let result=try? SafireSpatialResult.decodeSpatialData(data),
              result.assetID==photo.id,result.contentIdentity==photo.contentIdentity,result.modelFingerprint==fingerprint,
              (try? result.validate()) != nil else { return nil }
        return result
    }
    func save(_ result:SafireSpatialResult) throws {
        try result.validate();try Task.checkCancellation()
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let data=try result.encodedSpatialData()
        // A bounded disk cache. Investigations retain their review state separately.
        let files=(try? FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:[.fileSizeKey,.contentModificationDateKey])) ?? []
        var total=data.count
        let ordered=files.sorted { ((try? $0.resourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate) ?? .distantPast) < ((try? $1.resourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate) ?? .distantPast) }
        for file in files { total += (try? file.resourceValues(forKeys:[.fileSizeKey]).fileSize) ?? 0 }
        for file in ordered where total>512*1024*1024 {
            let size=(try? file.resourceValues(forKeys:[.fileSizeKey]).fileSize) ?? 0
            try? FileManager.default.removeItem(at:file);total-=size
        }
        try data.write(to:root.appendingPathComponent(result.cacheKey+".sfm"),options:.atomic)
    }
}
