import Foundation
import CoreGraphics
import CryptoKit

/// All spatial evidence uses oriented image pixels, with an origin at the top left.
/// This is also the SAFIRE wire contract. CrystalOrb uses bottom-left coordinates.
struct LineageRegion: Codable, Equatable, Sendable, Identifiable {
    let id: Int
    let bounds: CGRect
    let pixelCount: Int
    /// Pixel-edge segments, including holes; no smoothing that could move evidence.
    let contour: [LineageSegment]
}
struct LineageSegment: Codable, Equatable, Sendable { let a: CGPoint; let b: CGPoint }

struct SafireSpatialResult: Codable, Sendable {
    static let runtimeRevision = "safire-native-v2"
    static let preprocessingVersion = "safire-opencv-linear-rgb-square-v1"
    let assetID: String
    let contentIdentity: String
    let modelFingerprint: String
    let runtimeRevision: String
    let preprocessingVersion: String
    let width: Int
    let height: Int
    let probabilities: [Float]
    /// Cluster labels are source partitions, never semantic labels or chronology.
    let partitions: [UInt16]?
    func validate() throws {
        guard !assetID.isEmpty, contentIdentity.count == 64, contentIdentity.allSatisfy(\.isHexDigit),
              modelFingerprint.count == 64, modelFingerprint.allSatisfy(\.isHexDigit),
              runtimeRevision == Self.runtimeRevision, preprocessingVersion == Self.preprocessingVersion,
              width > 0, height > 0, width <= 1024, height <= 1024,
              probabilities.count == width * height,
              probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              (partitions == nil || (partitions?.count == width * height && partitions!.allSatisfy({ $0 < 8 }))) else {
            throw PhotosMedia.failure("Invalid or incompatible SAFIRE spatial evidence.")
        }
    }
    var cacheKey: String {
        LineageIdentity.hash([contentIdentity, modelFingerprint, preprocessingVersion, runtimeRevision].joined(separator: "\n"))
    }
    func regions(threshold: Float = 0.5, minimumPixels: Int = 4) throws -> [LineageRegion] {
        try validate()
        guard threshold.isFinite, (0...1).contains(threshold), minimumPixels > 0 else {
            throw PhotosMedia.failure("Invalid mask extraction settings.")
        }
        let mask = probabilities.map { $0 >= threshold }
        var visited = [Bool](repeating: false, count: mask.count)
        var regions: [LineageRegion] = []
        for seed in mask.indices where mask[seed] && !visited[seed] {
            try Task.checkCancellation()
            var queue = [seed], cursor = 0
            var minX = width, minY = height, maxX = 0, maxY = 0
            var contour: [LineageSegment] = []
            visited[seed] = true
            while cursor < queue.count {
                if cursor % 4096 == 0 { try Task.checkCancellation() }
                let k = queue[cursor]; cursor += 1
                let x = k % width, y = k / width
                minX = min(minX, x); maxX = max(maxX, x + 1)
                minY = min(minY, y); maxY = max(maxY, y + 1)
                for (nx, ny) in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)] {
                    if nx >= 0 && ny >= 0 && nx < width && ny < height && mask[ny*width+nx] {
                        let next = ny*width+nx
                        if !visited[next] { visited[next] = true; queue.append(next) }
                    } else {
                        let a: CGPoint, b: CGPoint
                        if nx < x { a = CGPoint(x:x,y:y); b = CGPoint(x:x,y:y+1) }
                        else if nx > x { a = CGPoint(x:x+1,y:y); b = CGPoint(x:x+1,y:y+1) }
                        else if ny < y { a = CGPoint(x:x,y:y); b = CGPoint(x:x+1,y:y) }
                        else { a = CGPoint(x:x,y:y+1); b = CGPoint(x:x+1,y:y+1) }
                        contour.append(.init(a:a,b:b))
                    }
                }
            }
            if queue.count >= minimumPixels {
                regions.append(.init(id:seed, bounds:CGRect(x:minX,y:minY,width:maxX-minX,height:maxY-minY),
                                     pixelCount:queue.count, contour:contour))
            }
        }
        return regions
    }
    func nativeBounds(_ region: LineageRegion, size: CGSize) -> CGRect {
        region.bounds.applying(CGAffineTransform(scaleX:size.width / CGFloat(width), y:size.height / CGFloat(height)))
    }
}

/// Lightweight group story facts; these are observations from genuine masks,
/// never an authenticity score or a prediction of editing chronology.
struct LineageSafireSummary:Codable,Sendable {
    let contentIdentity:String
    let modelFingerprint:String
    let flaggedPixels:Int
    let maskPixels:Int
    var ratio:Double { Double(flaggedPixels)/Double(max(1,maskPixels)) }
    init(_ result:SafireSpatialResult) {
        contentIdentity=result.contentIdentity;modelFingerprint=result.modelFingerprint
        flaggedPixels=result.probabilities.reduce(0) { $0+($1>=0.5 ? 1 : 0) }
        maskPixels=result.width*result.height
    }
}

enum LineageIdentity {
    static func hash(_ value: String) -> String { SHA256.hash(data:Data(value.utf8)).map { String(format:"%02x",$0) }.joined() }
}

struct LineagePhoto: Identifiable, Codable, Equatable, Sendable {
    let id: String
    let contentIdentity: String
    let name: String
    let width: Int
    let height: Int
    let format: String
    let url: URL
    let offset: Double
    var photosDate: Date? = nil
    var size: CGSize { CGSize(width:width,height:height) }
}
struct LineageRelation: Identifiable, Codable, Equatable, Sendable {
    let a: String
    let b: String
    let correlation: Double
    let meanAbsoluteDifference: Double
    let overlap: Double
    let registration: CrystalOrbRegistrationRecord
    var dimensionChange = false
    var confirmedParent: String? = nil
    var inferredParent: String? = nil
    var sourceEvidence: LineageSourceEvidence? = nil
    var parent: String? { confirmedParent ?? inferredParent }
    var id: String { LineageIdentity.hash([a,b].sorted().joined(separator:"\n")) }
    var label: String {
        if confirmedParent != nil { return "User-confirmed relationship" }
        if let sourceEvidence {
            if sourceEvidence.method == "contained crop correspondence" { return "Possible crop · date order tentative" }
            return sourceEvidence.dimensionsChanged ? "Possible resampling · date order tentative" : "Pixel-equivalent · date order tentative" }
        return dimensionChange ? "Resampling consistent · order uncertain" : "Shared details · order uncertain"
    }
    var observed: String {
        "Registered photo previews share spatial detail across \(Int(overlap * 100))% of the comparison. Dimensions and pixel differences can reflect resizing, tone changes or recompression; they do not identify an editor."
    }
}
struct CrystalOrbRegistrationRecord: Codable, Equatable, Sendable {
    var x: Double = 0; var y: Double = 0
    var scale: Double = 1
    var value: CrystalOrbRegistration { .init(x:x,y:y,scale:scale) }
}
struct LineageFamily: Identifiable, Codable, Equatable, Sendable {
    var id: String
    var photoIDs: [String]
    var relations: [LineageRelation]
    /// Only a supported or confirmed partial order can suggest a source here.
    var sourceCandidate: String? {
        let directed = relations.filter { $0.parent != nil }
        guard !directed.isEmpty else { return nil }
        let children = Set(directed.map { $0.parent == $0.a ? $0.b : $0.a })
        let roots = photoIDs.filter { !children.contains($0) }
        return roots.count == 1 ? roots[0] : nil
    }
    func reachable(from start: String, descendants: Bool) -> Set<String> {
        var visited: Set<String> = [start], queue = [start], cursor = 0
        while cursor < queue.count {
            let current = queue[cursor]; cursor += 1
            for relation in relations {
                guard let parent = relation.parent else { continue }
                let child = parent == relation.a ? relation.b : relation.a
                let from = descendants ? parent : child, to = descendants ? child : parent
                if from == current && visited.insert(to).inserted { queue.append(to) }
            }
        }
        return visited
    }
    mutating func confirm(_ relationID: String, parent: String) -> Bool {
        guard let index = relations.firstIndex(where: { $0.id == relationID }),
              [relations[index].a,relations[index].b].contains(parent) else { return false }
        let child = parent == relations[index].a ? relations[index].b : relations[index].a
        // Remove a previous direction before checking the new one.
        var candidate = self; candidate.relations[index].confirmedParent = nil; candidate.relations[index].inferredParent = nil
        guard !candidate.reachable(from:child,descendants:true).contains(parent) else { return false }
        relations[index].confirmedParent = parent
        return true
    }
}

struct LineageInvestigation: Codable, Sendable {
    var photos: [LineagePhoto] = []
    var families: [LineageFamily] = []
    var keepers: Set<String> = []
    var excluded: Set<String> = []
    var evidence: [String:SafireSpatialResult] = [:]
    var findings: [String:LineageSafireSummary]? = nil
    var selectedPhotoID: String?
    var selectedFamilyID: String?

    /// Success-only deletion transaction. Also used for external Photos changes.
    mutating func remove(_ ids: Set<String>) {
        photos.removeAll { ids.contains($0.id) }
        keepers.subtract(ids); excluded.subtract(ids)
        for id in ids { evidence[id] = nil;findings?[id]=nil }
        families = families.compactMap { family in
            var family = family
            family.photoIDs.removeAll { ids.contains($0) }
            family.relations.removeAll { ids.contains($0.a) || ids.contains($0.b) }
            return family.photoIDs.isEmpty ? nil : family
        }
        // A deleted bridge cannot keep otherwise disconnected photos in one
        // asserted family. Split only the affected connected components.
        families = families.flatMap { family -> [LineageFamily] in
            var remaining=Set(family.photoIDs),components:[LineageFamily]=[]
            for seed in family.photoIDs where remaining.remove(seed) != nil {
                var members:Set<String>=[seed],queue=[seed],cursor=0
                while cursor<queue.count {
                    let current=queue[cursor];cursor+=1
                    for relation in family.relations {
                        let other=relation.a==current ? relation.b : relation.b==current ? relation.a : ""
                        if remaining.remove(other) != nil { members.insert(other);queue.append(other) }
                    }
                }
                components.append(.init(id:components.isEmpty ? family.id : LineageIdentity.hash(family.id+"|"+seed),photoIDs:family.photoIDs.filter(members.contains),relations:family.relations.filter{members.contains($0.a) && members.contains($0.b)}))
            }
            return components
        }
        if let id = selectedPhotoID, ids.contains(id) { selectedPhotoID = nil }
        if let selectedPhotoID,let selectedFamily=families.first(where:{$0.photoIDs.contains(selectedPhotoID)}) { selectedFamilyID=selectedFamily.id }
        else if !families.contains(where: { $0.id == selectedFamilyID }) { selectedFamilyID = families.first?.id }
        if selectedPhotoID == nil { selectedPhotoID = families.first(where: { $0.id == selectedFamilyID })?.photoIDs.first }
    }
    mutating func accept(_ result: SafireSpatialResult) throws {
        try result.validate()
        guard let photo = photos.first(where: { $0.id == result.assetID }),
              photo.contentIdentity == result.contentIdentity else { throw CancellationError() }
        evidence[photo.id] = result
    }
    /// Stable IDs/order survive progressive discoveries. Confirmations survive only
    /// while the exact pair's content identities are unchanged (checked by caller).
    mutating func merge(photos next: [LineagePhoto], relations: [LineageRelation]) {
        let newMap = Dictionary(uniqueKeysWithValues:next.map { ($0.id,$0) })
        let changed = Set(photos.filter { newMap[$0.id]?.contentIdentity != $0.contentIdentity }.map(\.id))
        remove(Set(photos.map(\.id)).subtracting(newMap.keys))
        for id in changed { evidence[id] = nil;findings?[id]=nil }
        let oldFamilies = families
        let oldRelations = Dictionary(oldFamilies.flatMap(\.relations).map { ($0.id,$0) }, uniquingKeysWith: { a,_ in a })
        self.photos = next
        var remaining = Set(next.map(\.id)).subtracting(excluded)
        var grouped: [LineageFamily] = []
        for photo in next where remaining.contains(photo.id) {
            var group: Set<String> = [photo.id], queue = [photo.id], cursor = 0
            remaining.remove(photo.id)
            while cursor < queue.count {
                let current = queue[cursor]; cursor += 1
                for r in relations where !excluded.contains(r.a) && !excluded.contains(r.b) {
                    let other = r.a == current ? r.b : r.b == current ? r.a : ""
                    if remaining.remove(other) != nil { group.insert(other); queue.append(other) }
                }
            }
            let previous = oldFamilies.first { !Set($0.photoIDs).isDisjoint(with:group) && !grouped.map(\.id).contains($0.id) }
            var ids = (previous?.photoIDs ?? []).filter(group.contains)
            ids += next.map(\.id).filter { group.contains($0) && !ids.contains($0) }
            var links = relations.filter { group.contains($0.a) && group.contains($0.b) }
            for index in links.indices where !changed.contains(links[index].a) && !changed.contains(links[index].b) {
                links[index].confirmedParent = oldRelations[links[index].id]?.confirmedParent
                if links[index].sourceEvidence == nil {
                    links[index].sourceEvidence = oldRelations[links[index].id]?.sourceEvidence
                    links[index].inferredParent = oldRelations[links[index].id]?.inferredParent
                }
            }
            grouped.append(.init(id:previous?.id ?? LineageIdentity.hash(photo.id),photoIDs:ids,relations:links))
        }
        families = grouped
        if let selectedPhotoID,let selectedFamily=families.first(where:{$0.photoIDs.contains(selectedPhotoID)}) { selectedFamilyID=selectedFamily.id }
        else if !families.contains(where: { $0.id == selectedFamilyID }) { selectedFamilyID = families.first?.id }
        if !next.contains(where: { $0.id == selectedPhotoID }) { selectedPhotoID = families.first?.photoIDs.first }
    }
}
