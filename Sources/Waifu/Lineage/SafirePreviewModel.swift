import AppKit
import Combine

@MainActor final class SafirePreviewModel:ObservableObject {
    @Published private(set) var inputID: UUID?
    @Published private(set) var overlayThumbnail: CGImage?
    @Published private(set) var maskThumbnail: CGImage?
    @Published private(set) var photoURL:URL?
    @Published private(set) var overlay:NSImage?
    @Published private(set) var mask:NSImage?
    @Published private(set) var status=""
    @Published private(set) var loading=false
    private var ticket=UUID()
    func load(url:URL?,enabled:Bool,input:PreviewEvidenceInput?) async {
        let token=UUID();ticket=token;photoURL=url;inputID=input?.id;overlayThumbnail=nil;maskThumbnail=nil;overlay=nil;mask=nil;status="";loading=false
        guard enabled,let url,let item=PhotosMedia.shared.item(url),item.isStill else { return }
        guard let input,input.photoURL==url else { status="Loading photo…";return }
        loading=true
        defer { if ticket==token { loading=false } }
        do {
            let base=await Task.detached(priority:.utility){SafirePhotoIdentity.make(item)}.value
            let (png,photo)=try await Task.detached(priority:.utility) {
                let png=try SafireBackend.png(input.image)
                return (png,input.photo(base,png:png))
            }.value
            let local=SafireConfiguration.load()
            let remote=RemoteInferencePool.shared.safireClients.first
            guard let fingerprint=remote?.capabilities?.safire?.modelFingerprint ?? local?.fingerprint else {
                status="Install SAFIRE in Settings → Models";return
            }
            let result:SafireSpatialResult
            if let cached=await SafireEvidenceCache.shared.read(photo:photo,fingerprint:fingerprint) { result=cached }
            else {
                // Rapid preview navigation never queues analysis of passed photos.
                try await Task.sleep(for:.milliseconds(200));try Task.checkCancellation()
                status="SAFIRE · analyzing this photo…"
                result=try await SafireJobWait.run(onWaiting:{
                    self.status="SAFIRE · waiting for another job; resumes automatically"
                }) {
                    if let remote { return try await remote.safire(png:png,photo:photo) }
                    if local != nil { return try await SafireBackend.shared.infer(png:png,photo:photo) }
                    throw RemoteError.modelUnavailable("Install SAFIRE in Settings → Models.")
                }
                try Task.checkCancellation()
                do { try await SafireEvidenceCache.shared.save(result) }
                catch { Diag.trace(.preview,message:"SAFIRE cache write: \(error)") }
            }
            let material=try await SafireCardRenderer.shared.make(image:input.image,result:result,maximumSide:1600)
            let summary=await Task.detached(priority:.utility){LineageSafireSummary(result)}.value
            guard !Task.isCancelled,ticket==token else { return }
            overlay=NSImage(cgImage:material.highlighted,size:NSSize(width:material.highlighted.width,height:material.highlighted.height))
            mask=NSImage(cgImage:material.mask,size:NSSize(width:material.mask.width,height:material.mask.height))
            overlayThumbnail = EvidenceThumbnailRaster.make(material.highlighted)
            maskThumbnail = EvidenceThumbnailRaster.make(material.mask)
            status="SAFIRE · "+summary.ratio.formatted(.percent.precision(.fractionLength(2)))+" mask area"
        } catch is CancellationError { }
        catch {
            Diag.trace(.preview,message:"SAFIRE preview: \(error)")
            if ticket==token { status="SAFIRE evidence unavailable" }
        }
    }
}
