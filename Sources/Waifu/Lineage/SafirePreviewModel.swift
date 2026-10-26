import AppKit
import Combine

@MainActor final class SafirePreviewModel:ObservableObject {
    @Published private(set) var photoURL:URL?
    @Published private(set) var overlay:NSImage?
    @Published private(set) var mask:NSImage?
    @Published private(set) var status=""
    @Published private(set) var loading=false
    private var ticket=UUID()
    func load(url:URL?,enabled:Bool) async {
        let token=UUID();ticket=token;photoURL=url;overlay=nil;mask=nil;status="";loading=false
        guard enabled,let url,let item=PhotosMedia.shared.item(url),item.isStill else { return }
        loading=true
        defer { if ticket==token { loading=false } }
        do {
            let photo=await Task.detached(priority:.utility){SafirePhotoIdentity.make(item)}.value
            let local=SafireConfiguration.load()
            let remote=RemoteInferencePool.shared.entries.map(\.client).first {
                $0.offersSafire && $0.capabilities?.safire?.runtimeRevision==SafireSpatialResult.runtimeRevision
                    && $0.capabilities?.safire?.preprocessingVersion==SafireSpatialResult.preprocessingVersion
            }
            guard let fingerprint=local?.fingerprint ?? remote?.capabilities?.safire?.modelFingerprint else {
                status="Install SAFIRE in Settings → Models";return
            }
            let result:SafireSpatialResult
            if let cached=await SafireEvidenceCache.shared.read(photo:photo,fingerprint:fingerprint) { result=cached }
            else {
                // Rapid preview navigation never queues analysis of passed photos.
                try await Task.sleep(for:.milliseconds(200));try Task.checkCancellation()
                status="SAFIRE · analyzing this photo…"
                let original=try await PhotosMedia.shared.image(url,seconds:0)
                let png=try await Task.detached(priority:.utility){try SafireBackend.png(original)}.value
                result=try await SafireJobWait.run(onWaiting:{
                    self.status="SAFIRE · waiting for another job; resumes automatically"
                }) {
                    if local != nil { return try await SafireBackend.shared.infer(png:png,photo:photo) }
                    if let remote { return try await remote.safire(png:png,photo:photo) }
                    throw RemoteError.modelUnavailable("Install SAFIRE in Settings → Models.")
                }
                try Task.checkCancellation();try await SafireEvidenceCache.shared.save(result)
            }
            let image=try await PhotosMedia.shared.analysisImage(url,seconds:0)
            let material=try await SafireCardRenderer.shared.make(image:image,result:result,maximumSide:1600)
            let summary=await Task.detached(priority:.utility){LineageSafireSummary(result)}.value
            guard !Task.isCancelled,ticket==token else { return }
            overlay=NSImage(cgImage:material.highlighted,size:NSSize(width:material.highlighted.width,height:material.highlighted.height))
            mask=NSImage(cgImage:material.mask,size:NSSize(width:material.mask.width,height:material.mask.height))
            status="SAFIRE · "+summary.ratio.formatted(.percent.precision(.fractionLength(2)))+" mask area"
        } catch is CancellationError { }
        catch { if ticket==token { status="SAFIRE evidence unavailable" } }
    }
}
