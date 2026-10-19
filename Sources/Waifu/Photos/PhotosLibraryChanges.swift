import Foundation
import Photos

/// PhotoKit calls on an arbitrary queue. Consumers coalesce notifications and
/// fetch fresh snapshots off the main actor instead of sharing PHChange objects.
final class PhotosLibraryChanges: NSObject, PHPhotoLibraryChangeObserver, @unchecked Sendable {
    static let shared = PhotosLibraryChanges()
    static let notification = Notification.Name("WaifuPhotosLibraryDidChange")
    private let lock = NSLock()
    private var observing = false

    /// Called only after Photos authorization has been granted.
    func start() {
        lock.withLock {
            guard !observing else { return }
            observing = true
            PHPhotoLibrary.shared().register(self)
        }
    }

    func photoLibraryDidChange(_ changeInstance: PHChange) {
        NotificationCenter.default.post(name: Self.notification, object: nil)
    }
}
