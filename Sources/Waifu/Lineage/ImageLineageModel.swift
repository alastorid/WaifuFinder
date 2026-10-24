import AppKit
import Combine
import Photos

@MainActor final class ImageLineageModel: ObservableObject {
    @Published private(set) var isOpen = false
    @Published var document = LineageInvestigation()
    @Published private(set) var running = false
    @Published private(set) var progress = ""
    @Published var issue: String?
    @Published var selectedRelationID: String?
    @Published var selectedRegionID: Int?
    @Published var showDeleteConfirmation = false
    @Published private(set) var deletionTargets: [LineagePhoto] = []
    @Published private(set) var overlay: LineageOverlayMaterial?
    @Published private(set) var metadata: PhotoMetadata?
    @Published private(set) var metadataLoading = false
    @Published private(set) var storyImage: CGImage?
    @Published var showOutlines = true
    @Published var showMasks = true
    @Published var showContours = false
    @Published var showPartitions = false
    @Published var historyList = false
    @Published var traceDescendants = true
    private let discovery = LineageDiscovery()
    private let sourceAssessor = LineageSourceAssessor()
    private var task: Task<Void,Never>?
    private var inspection: Task<Void,Never>?
    private var saveTask: Task<Void,Never>?
    private var ticket = UUID()
    private var inspectionTicket = UUID()
    private var albumID: String?
    private var source: URL?
    private var targetAssets = Set<String>()
    private var targetAlbum: URL?
    private var scopeID: String?
    private var anchorAsset: String?
    private weak var session: WaifuSession?
    var canDelete: Bool {
        guard let session else { return false }
        return isOpen && selected != nil && !session.isDeletingPhotos && !session.isSyncingPhotosOrder
            && !session.isWritingPhotosTags && !session.progress.isRunning && session.uncover == nil && !session.showExportSheet
    }
    var family: LineageFamily? { document.families.first { $0.id == document.selectedFamilyID } }
    var selected: LineagePhoto? { document.photos.first { $0.id == document.selectedPhotoID } }
    var relation: LineageRelation? { family?.relations.first { $0.id == selectedRelationID } }
    var evidence: SafireSpatialResult? { selected.flatMap { document.evidence[$0.id] } }
    var regions: [LineageRegion] { overlay?.regions ?? [] }
    var highlighted: Set<String> {
        guard let id = document.selectedPhotoID else { return [] }
        return family?.reachable(from:id,descendants:traceDescendants) ?? [id]
    }
    static func allowed(source: URL?) -> Bool { source.flatMap { PhotosMedia.shared.album($0) } != nil }

    func open(session: WaifuSession, anchoredAt frame: AnalyzedFrame) {
        guard Self.allowed(source:session.photosAlbumURL),let url=session.photosAlbumURL,
              let frames=LineageGroupTarget.frames(anchor:frame,frames:session.frames,groups:session.groups,
                  photosSource:true,usesGroupRows:session.usesGroupRows) else { close(); return }
        let ids=Set(frames.compactMap { frame -> String? in
            guard let photo=PhotosMedia.shared.photoItemURL(for:url,seconds:frame.timeSeconds) else { return nil }
            return PhotosMedia.shared.item(photo)?.asset.localIdentifier
        })
        guard !ids.isEmpty,let anchor=PhotosMedia.shared.assetIdentifier(for:url,seconds:frame.timeSeconds),ids.contains(anchor) else { close(); return }
        saveImmediately()
        targetAssets=ids;targetAlbum=url;anchorAsset=anchor
        scopeID=LineageIdentity.hash(ids.sorted().joined(separator:"\n"))
        self.session = session; isOpen = true
        session.previewFrame = nil; session.closeUncover()
        refresh(session:session)
    }
    func close() {
        saveImmediately(); isOpen = false; task?.cancel(); inspection?.cancel()
        ticket = UUID(); inspectionTicket = UUID(); running = false
        overlay=nil;storyImage=nil
    }
    func refresh(session: WaifuSession) {
        self.session = session
        guard isOpen,let scopeID,let url = session.photosAlbumURL,url==targetAlbum,let album = PhotosMedia.shared.album(url) else { close(); return }
        // Selection changes don't trigger inference; only content/membership changes.
        task?.cancel(); inspection?.cancel(); inspectionTicket = UUID()
        let next = UUID(); ticket = next
        let collection = album.collectionIdentifier+"|group|"+scopeID
        source = url; running = true; progress = "Reading Photos identities…"; issue = nil
        let changedAlbum = albumID != collection
        if changedAlbum { saveImmediately() }
        albumID = collection
        if changedAlbum { document = .init(); selectedRelationID = nil; storyImage=nil }
        let requestedAssets=targetAssets
        task = Task {
            if changedAlbum, let saved = await LineageStore.shared.read(album:collection), ticket == next { document = saved }
            // PhotoKit resource/filename enumeration stays off the UI actor.
            let photos = await Task.detached(priority:.utility) {
                album.items.filter { $0.isStill && requestedAssets.contains($0.asset.localIdentifier) }.map(SafirePhotoIdentity.make)
            }.value
            guard !Task.isCancelled, ticket == next else { return }
            // Invalidate old pixels/evidence immediately, including deleted targets.
            let ids = Set(photos.map(\.id))
            document.remove(Set(document.photos.map(\.id)).subtracting(ids))
            let validContent = Dictionary(uniqueKeysWithValues:photos.map { ($0.id,$0.contentIdentity) })
            let priorLinks = document.families.flatMap(\.relations).filter { r in
                [r.a,r.b].allSatisfy { id in document.photos.first { $0.id == id }?.contentIdentity == validContent[id] }
            }
            document.merge(photos:photos,relations:priorLinks)
            consolidateTargetGroup()
            if let anchorAsset,photos.contains(where:{$0.id==anchorAsset}) { document.selectedPhotoID=anchorAsset;self.anchorAsset=nil }
            selectedRelationID = nil; selectedRegionID = nil
            if let selected = document.selectedPhotoID { select(selected) }
            // The existing grouping defines the requested scope. Prioritize the
            // actual neural findings; optional relationship analysis is separate.
            await analyzeSafire(ticket:next)
            if self.ticket==next { running=false;persist() }

        }
    }
    func investigateRelationships() {
        guard isOpen,!running,let session,let source else { return }
        task?.cancel();let next=UUID();ticket=next;running=true
        let photos=document.photos,frames=session.frames,generation=session.thumbnailGeneration
        task=Task {
            defer { if ticket==next { running=false } }
            do {
                let links=try await discovery.discover(photos:photos,frames:frames,source:source,generation:generation) { [weak self] links,done,total,warning in
                    await self?.publish(links,photos:photos,done:done,total:total,warning:warning,ticket:next)
                }
                guard !Task.isCancelled,ticket==next else { return }
                mergeDiscoveries(links,photos:photos);await assessSources(ticket:next)
                if ticket==next { progress="Selected group · relationship evidence ready";persist() }
            } catch is CancellationError { }
            catch { if ticket==next { issue=error.localizedDescription } }
        }
    }
    private func assessSources(ticket:UUID) async {
        running=true
        defer { if self.ticket==ticket { running=false } }
        let relations=document.families.flatMap(\.relations)
        for (index,relation) in relations.enumerated() where relation.sourceEvidence == nil && relation.confirmedParent == nil {
            guard !Task.isCancelled,self.ticket==ticket,isOpen else { return }
            progress="Checking native pixels and supporting dates · \(index+1) of \(relations.count)"
            do {
                guard let evidence=try await sourceAssessor.assess(relation,photos:document.photos) else { continue }
                guard !Task.isCancelled,self.ticket==ticket,
                      let familyIndex=document.families.firstIndex(where:{$0.relations.contains(where:{$0.id==relation.id})}),
                      let link=document.families[familyIndex].relations.firstIndex(where:{$0.id==relation.id}) else { continue }
                let child=evidence.childID,parent=evidence.parentID
                // Native hypotheses always follow strictly increasing dates.
                // Confirmed user edges still take precedence and must stay acyclic.
                guard !document.families[familyIndex].reachable(from:child,descendants:true).contains(parent) else { continue }
                document.families[familyIndex].relations[link].inferredParent=parent
                document.families[familyIndex].relations[link].sourceEvidence=evidence
            } catch is CancellationError { return }
            catch { /* Unavailable originals leave the existing relation uncertain. */ }
        }
        if self.ticket==ticket { persist() }
    }
    private func analyzeSafire(ticket:UUID) async {
        let local=SafireConfiguration.load()
        let remote=RemoteInferencePool.shared.entries.map(\.client).first {
            $0.offersSafire && $0.capabilities?.safire?.runtimeRevision==SafireSpatialResult.runtimeRevision
                && $0.capabilities?.safire?.preprocessingVersion==SafireSpatialResult.preprocessingVersion
        }
        guard let fingerprint=local?.fingerprint ?? remote?.capabilities?.safire?.modelFingerprint else { progress="Install SAFIRE in Settings → Models to read this group’s spatial evidence";return }
        let targets=document.photos.filter { targetAssets.contains($0.id) }
        running=true
        defer { if self.ticket==ticket { running=false } }
        for (index,photo) in targets.enumerated() {
            guard !Task.isCancelled,self.ticket==ticket,isOpen else { return }
            progress="SAFIRE spatial evidence · \(index+1) of \(targets.count)"
            do {
                let result:SafireSpatialResult
                if let cached=await SafireEvidenceCache.shared.read(photo:photo,fingerprint:fingerprint) { result=cached }
                else {
                    let image=try await PhotosMedia.shared.image(photo.url,seconds:0)
                    let png=try await Task.detached(priority:.utility) { try SafireBackend.png(image) }.value
                    result=try await SafireJobWait.run(onWaiting:{
                        self.progress="SAFIRE · waiting for another job; this photo will resume automatically"
                    }) {
                        if local != nil { return try await SafireBackend.shared.infer(png:png,photo:photo) }
                        if let remote { return try await remote.safire(png:png,photo:photo) }
                        throw RemoteError.modelUnavailable("Install SAFIRE in Settings → Models.")
                    }
                    guard !Task.isCancelled,self.ticket==ticket else { return }
                    try await SafireEvidenceCache.shared.save(result)
                }
                guard !Task.isCancelled,self.ticket==ticket else { return }
                let summary=await Task.detached(priority:.utility) { LineageSafireSummary(result) }.value
                guard !Task.isCancelled,self.ticket==ticket else { return }
                if document.findings==nil { document.findings=[:] }
                document.findings?[photo.id]=summary
                try document.accept(result);trimEvidence(keeping:photo.id)
                if document.selectedPhotoID==photo.id {
                    let material=try await Task.detached(priority:.utility) { try LineageOverlayMaterial.make(result) }.value
                    if self.ticket==ticket,document.selectedPhotoID==photo.id { overlay=material }
                }
            } catch is CancellationError { return }
            catch { if self.ticket==ticket { issue=error.localizedDescription } }
        }
        if self.ticket==ticket {
            let completed=targets.filter { document.findings?[$0.id]?.contentIdentity==$0.contentIdentity }.count
            progress="SAFIRE · \(completed) of \(targets.count) photos have findings"
            persist()
        }
    }
    private func mergeDiscoveries(_ links:[LineageRelation],photos:[LineagePhoto]) {
        var verified=Dictionary(document.families.flatMap(\.relations).map{($0.id,$0)},uniquingKeysWith:{a,_ in a})
        for link in links { verified[link.id]=link }
        // Prior links survived the content-identity gate at refresh start. Keep
        // their proven correspondence while new results arrive in another order.
        document.merge(photos:photos,relations:verified.values.sorted{$0.id<$1.id})
        consolidateTargetGroup()
    }
    private func consolidateTargetGroup() {
        guard let scopeID else { return }
        let photos=document.photos.filter { targetAssets.contains($0.id) }
        let links=document.families.flatMap(\.relations)
        // Membership was explicitly chosen from the existing group. Missing
        // correspondence remains unresolved; it is not a fabricated graph edge.
        document.families=photos.isEmpty ? [] : [.init(id:scopeID,photoIDs:photos.map(\.id),relations:links)]
        document.selectedFamilyID=document.families.first?.id
    }
    private func publish(_ links:[LineageRelation],photos:[LineagePhoto],done:Int,total:Int,warning:String?,ticket:UUID) {
        guard self.ticket == ticket, isOpen else { return }
        mergeDiscoveries(links,photos:photos)
        progress = "Matching shared image details · \(done) of \(total)"; issue = warning
    }
    func cancel() { task?.cancel(); ticket = UUID(); running = false; progress = "Discovery stopped · partial results retained"; persist() }
    func chooseFamily(_ id: String) {
        document.selectedFamilyID = id; selectedRelationID = nil
        if let id = family?.photoIDs.first { select(id) }; persist()
    }
    func select(_ id: String, native: Bool = false) {
        guard document.photos.contains(where: { $0.id == id }) else { return }
        document.selectedPhotoID = id; selectedRegionID = nil
        overlay = nil
        loadOverlay(id)

        if let f = document.families.first(where: { $0.photoIDs.contains(id) }) { document.selectedFamilyID = f.id }
        if relation.map({ ![$0.a,$0.b].contains(id) }) == true { selectedRelationID = nil }
        loadStoryPhoto(native:native)
        persist()
    }
    private func loadStoryPhoto(native:Bool=false) {
        inspection?.cancel();let token=UUID();inspectionTicket=token
        storyImage=nil;metadata=nil;metadataLoading=true
        guard let selected else { metadataLoading=false;return }
        inspection=Task {
            do {
                async let basic=PhotoMetadataService.shared.basic(for:selected.url)
                let image=try await (native ? PhotosMedia.shared.image(selected.url,seconds:0) : PhotosMedia.shared.analysisImage(selected.url,seconds:0))
                let info=await basic
                guard !Task.isCancelled,inspectionTicket==token,self.selected?.contentIdentity==selected.contentIdentity else { return }
                storyImage=image;metadata=info;metadataLoading=false
                if let detail=try? await PhotoMetadataService.shared.load(for:selected.url),inspectionTicket==token { metadata=detail }
            } catch { if inspectionTicket==token { metadataLoading=false;issue=error.localizedDescription } }
        }
    }
    private func loadOverlay(_ id:String) {
        guard let photo=document.photos.first(where:{$0.id==id}) else { return }
        let stored=document.evidence[id]
        let fingerprint=stored?.modelFingerprint ?? SafireConfiguration.load()?.fingerprint
            ?? RemoteInferencePool.shared.entries.map(\.client).first(where:{$0.offersSafire})?.capabilities?.safire?.modelFingerprint
        guard let fingerprint else { return }
        let generation=ticket
        Task {
            guard let result=await cachedEvidence(stored,photo:photo,fingerprint:fingerprint) else { return }
            let next=try? await Task.detached(priority:.utility) { try LineageOverlayMaterial.make(result) }.value
            guard ticket==generation,document.selectedPhotoID==id,
                  document.photos.first(where:{$0.id==id})?.contentIdentity==photo.contentIdentity else { return }
            try? document.accept(result);trimEvidence(keeping:id);overlay=next
        }
    }
    private func cachedEvidence(_ stored:SafireSpatialResult?,photo:LineagePhoto,fingerprint:String) async -> SafireSpatialResult? {
        if let stored { return stored }
        return await SafireEvidenceCache.shared.read(photo:photo,fingerprint:fingerprint)
    }
    private func trimEvidence(keeping id:String) {
        let survivors=Set([id,document.selectedPhotoID].compactMap{$0})
        for key in Array(document.evidence.keys) where !survivors.contains(key) { document.evidence[key]=nil }
    }
    func selectRelation(_ id: String) {
        selectedRelationID = id
        guard let r = relation else { return }
        select(r.b); persist()
    }
    func inspect(native: Bool) { loadStoryPhoto(native:native) }
    func selectRegion(_ id:Int) { selectedRegionID=id }
    func toggleKeeper() {
        guard let id = selected?.id else { return }
        if !document.keepers.insert(id).inserted { document.keepers.remove(id) }; persist()
    }
    func confirmRelation(parent:String) {
        guard let id = selectedRelationID, let f = document.families.firstIndex(where: { $0.id == document.selectedFamilyID }) else { return }
        if !document.families[f].confirm(id,parent:parent) { issue = "That relationship would create a cycle. Review the existing ancestry first." }
        persist()
    }
    func requestDelete(_ id: String? = nil) {
        guard canDelete,
              let target = document.photos.first(where: { $0.id == (id ?? document.selectedPhotoID) }) else { return }
        deletionTargets = [target]; showDeleteConfirmation = true
    }
    func deletionTargetsReset() { deletionTargets = [] }
    func deleteConfirmed() {
        guard let session, let source, !deletionTargets.isEmpty else { return }
        let ids = deletionTargets.map(\.id), album = albumID
        deletionTargets = []; showDeleteConfirmation = false
        session.deletePhotos(identifiers:ids,from:source) { [weak self] deleted in
            guard let self, self.albumID == album else { return }
            // Cancellation + new ticket prevents late local/remote results from
            // resurrecting deleted assets, even before the Photos refresh arrives.
            self.cancel(); self.inspection?.cancel(); self.inspectionTicket = UUID()
            self.document.remove(deleted); self.consolidateTargetGroup(); self.selectedRelationID = nil; self.selectedRegionID = nil
            self.storyImage=nil; self.persist()
            if let id = self.document.selectedPhotoID { self.select(id) }
        }
    }
    private func saveImmediately() {
        guard let albumID else { return }
        saveTask?.cancel();let snapshot=document
        Task {
            do { try await LineageStore.shared.save(snapshot,album:albumID) }
            catch { issue="Couldn’t save this investigation: \(error.localizedDescription)" }
        }
    }
    func persist() {
        guard let albumID else { return }
        saveTask?.cancel(); let snapshot = document
        saveTask = Task {
            do { try await Task.sleep(for:.milliseconds(150)); try await LineageStore.shared.save(snapshot,album:albumID) }
            catch is CancellationError { } catch { issue = "Couldn’t save this investigation: \(error.localizedDescription)" }
        }
    }
}
