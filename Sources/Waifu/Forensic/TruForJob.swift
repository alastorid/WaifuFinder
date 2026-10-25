import AppKit
import Combine

struct ForensicProgress:Sendable,Equatable {
    var isRunning=false,processed=0,total=0
    var elapsed:TimeInterval=0
    var fraction:Double { total>0 ? min(1,Double(processed)/Double(total)):0 }
    var framesPerSecond:Double { elapsed>0 ? Double(processed)/elapsed:0 }
}
private actor TruForWorkQueue {
    let photos:[TruForPhoto]
    var position=0
    init(_ photos:[TruForPhoto]) { self.photos=photos }
    func next()->[TruForPhoto] { let end=min(position+2,photos.count);defer{position=end};return Array(photos[position..<end]) }
}
@MainActor final class TruForJob:ObservableObject {
    @Published private(set) var progress=ForensicProgress()
    @Published private(set) var checking=false
    @Published private(set) var total=0
    @Published private(set) var scored=0
    @Published private(set) var message:String?
    private var source:URL?
    private var photos:[TruForPhoto]=[]
    private var scores:[String:Float]=[:]
    private var generation=UUID()
    private var operation:Task<Void,Never>?
    private var starting=false
    var remaining:Int { max(0,total-scored) }
    var localAvailable:Bool { TruForConfiguration.load() != nil }
    var remoteClients:[RemoteClient] { RemoteInferencePool.shared.entries.map(\.client).filter(\.offersTruFor) }
    var remotely:Bool { !remoteClients.isEmpty }
    var available:Bool { localAvailable || remotely }
    func refresh(session:WaifuSession)async {
        guard let url=session.photosAlbumURL,let album=PhotosMedia.shared.album(url) else { cancel();source=nil;photos=[];scores=[:];total=0;scored=0;return }
        if progress.isRunning,source==url {
            let current=album.items.compactMap { $0.isStill ? $0.assetCacheKey:nil }
            if current==photos.map(\.assetKey) { return }
            // Edits, deletion and reordered membership invalidate the job's
            // snapshot; its partial scalar cache remains reusable by identity.
            cancel()
        }
        if source != url { cancel();source=url;scores=[:];photos=[] }
        let token=UUID();generation=token;checking=true
        defer { if generation==token { checking=false } }
        let candidates=await Task.detached(priority:.utility) {
            album.items.compactMap { item -> TruForPhoto? in guard item.isStill,let key=item.assetCacheKey else { return nil };return .init(url:item.url,assetKey:key) }
        }.value
        guard !Task.isCancelled,generation==token else { return }
        do {
            let cached=try await TruForScoreMemory.shared.read(candidates,fingerprint:TruForConfiguration.scoreFingerprint)
            guard !Task.isCancelled,generation==token,session.photosAlbumURL==url else { return }
            photos=candidates;scores=cached;total=candidates.count;scored=cached.count
            session.applyForensicScores(cached,photos:candidates,fingerprint:TruForConfiguration.scoreFingerprint,source:url)
        } catch { if generation==token { message=error.localizedDescription } }
    }
    func start(session:WaifuSession,includeLocal:Bool)async {
        guard !progress.isRunning,!starting else { return }
        starting=true
        defer { starting=false }
        await refresh(session:session)
        guard !Task.isCancelled,!progress.isRunning,let source,session.photosAlbumURL==source,available else { return }
        let targets=photos.filter{scores[$0.contentIdentity]==nil}
        guard !targets.isEmpty else { return }
        let clients=remoteClients,includeThisMac=localAvailable && (clients.isEmpty || includeLocal)
        let lanes:[RemoteClient?]=clients.map{Optional($0)}+(includeThisMac ? [nil]:[])
        guard !lanes.isEmpty else { return }
        let token=UUID();generation=token;message=nil;checking=false
        progress = .init(isRunning:true,processed:0,total:targets.count,elapsed:0)
        let queue=TruForWorkQueue(targets),started=Date()
        operation=Task { [weak self,weak session] in
            guard let self,let session else { return }
            do {
                try await withThrowingTaskGroup(of:Void.self) { group in
                    for client in lanes {
                        group.addTask {
                            while true {
                                try Task.checkCancellation();let batch=await queue.next();if batch.isEmpty { return }
                                var pngs:[Data]=[]
                                for photo in batch {
                                    let image=try await PhotosMedia.shared.image(photo.url,seconds:0)
                                    pngs.append(try await Task.detached(priority:.utility){try SafireBackend.png(image)}.value)
                                }
                                let values:[Float]
                                if let client { values=try await client.truforScores(pngs) }
                                else {
                                    let output=try await SafireJobWait.run(onWaiting:{self.message="Waiting for another model job"}) { try await TruForBackend.shared.submit(images:pngs,evidence:false) }
                                    guard case .scores(let returned)=output else { throw ModelInstaller.fail("Unexpected TruFor score response") };values=returned
                                }
                                guard values.count==batch.count else { throw ModelInstaller.fail("Incomplete TruFor batch") }
                                for (i,photo) in batch.enumerated() { try await TruForScoreMemory.shared.remember(photo,score:values[i],fingerprint:TruForConfiguration.scoreFingerprint) }
                                try Task.checkCancellation()
                                await self.accept(batch,values:values,session:session,source:source,token:token,elapsed:Date().timeIntervalSince(started))
                            }
                        }
                    }
                    try await group.waitForAll()
                }
            } catch { if generation==token && !Task.isCancelled { message=error.localizedDescription } }
            if generation==token { progress.isRunning=false;progress.elapsed=Date().timeIntervalSince(started);operation=nil }
        }
    }
    private func accept(_ batch:[TruForPhoto],values:[Float],session:WaifuSession,source:URL,token:UUID,elapsed:TimeInterval) {
        guard generation==token,session.photosAlbumURL==source else { return }
        var updates:[String:Float]=[:]
        for (i,photo) in batch.enumerated() { scores[photo.contentIdentity]=values[i];updates[photo.contentIdentity]=values[i] }
        scored=scores.count;message=nil;progress.processed+=batch.count;progress.elapsed=elapsed
        session.applyForensicScores(updates,photos:batch,fingerprint:TruForConfiguration.scoreFingerprint,source:source)
    }
    func cancel() { operation?.cancel();operation=nil;generation=UUID();progress.isRunning=false;checking=false }
}
