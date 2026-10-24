import Foundation
import Photos

/// PhotoKit calls on an arbitrary queue. Consumers coalesce notifications and
/// fetch fresh snapshots off the main actor instead of sharing PHChange objects.
/// Each change is also distilled against the fetch results PhotosMedia keeps
/// alive, so consumers can attribute a change to precise assets instead of
/// answering every event with a wholesale reload.
final class PhotosLibraryChanges: NSObject, PHPhotoLibraryChangeObserver, @unchecked Sendable {
    static let shared = PhotosLibraryChanges()
    static let notification = Notification.Name("WaifuPhotosLibraryDidChange")
    private let lock = NSLock()
    private var observing = false
    private var pending = Digest()

    /// What one library change did to the albums being observed. `changedAssetIDs`
    /// covers metadata edits — including our own keyword writes; `structural`
    /// means membership or order changed (additions, deletions, moves), which
    /// requires a fresh snapshot rather than keyword-echo suppression.
    struct Digest: Sendable, Equatable {
        var albumAffected = false
        var structural = false
        var changedAssetIDs: Set<String> = []

        /// True when the only thing that happened is our own keyword writes
        /// landing on photos we know we wrote: no reload can be needed, because
        /// nothing in this workspace derives from Photos-side keywords.
        func isEcho(ofWritten written: Set<String>) -> Bool {
            albumAffected && !structural && !changedAssetIDs.isEmpty && changedAssetIDs.isSubset(of: written)
        }
    }

    /// Called only after Photos authorization has been granted.
    func start() {
        lock.withLock {
            guard !observing else { return }
            observing = true
            PHPhotoLibrary.shared().register(self)
        }
    }

    func photoLibraryDidChange(_ changeInstance: PHChange) {
        var digest = Digest()
        for fetch in PhotosMedia.shared.observedFetchResults() {
            guard let details = changeInstance.changeDetails(for: fetch) else { continue }
            digest.albumAffected = true
            if !details.insertedObjects.isEmpty || !details.removedObjects.isEmpty || details.hasMoves {
                digest.structural = true
            }
            for asset in details.changedObjects {
                digest.changedAssetIDs.insert(asset.localIdentifier)
            }
        }
        lock.withLock {
            pending.albumAffected = pending.albumAffected || digest.albumAffected
            pending.structural = pending.structural || digest.structural
            pending.changedAssetIDs.formUnion(digest.changedAssetIDs)
        }
        NotificationCenter.default.post(name: Self.notification, object: nil)
    }

    /// Merged digest of everything reported since the last consumption, so the
    /// consumers' debouncing cannot drop an earlier structural change between
    /// two keyword-only notifications.
    func consumeDigest() -> Digest {
        lock.withLock {
            let digest = pending
            pending = Digest()
            return digest
        }
    }
}
