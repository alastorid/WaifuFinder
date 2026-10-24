import CoreServices
import Foundation
import AppKit

/// Photos never idles between one reply and the next request: photos that share
/// one keyword list share a single native set-data event (no script to compile,
/// no extra read requests), and several events stay in flight at once.
enum PhotosKeywordWriter {
    static let mediaItemClass: OSType = 0x49506D69 // IPmi, Photos' public scripting dictionary
    static let keywordsProperty: OSType = 0x49506B77 // IPkw
    /// One rejected event must not lose the whole job, so no batch exceeds this.
    static let batchLimit = 64

    static func openPhotosIfNeeded() async throws {
        guard NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.Photos").isEmpty else { return }
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Photos") else {
            throw PhotosMedia.failure("Photos.app could not be found.")
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false
        _ = try await NSWorkspace.shared.openApplication(at: url, configuration: configuration)
    }

    static func event(for update: PhotosKeywords.Update) throws -> NSAppleEventDescriptor {
        try event(for: [update])
    }

    /// A single photo is addressed directly; a batch uses one property specifier
    /// over a list of photos — `keywords of {media item id …, …}` in AppleScript
    /// terms — so every photo in the batch is confirmed by one reply.
    static func event(for batch: [PhotosKeywords.Update]) throws -> NSAppleEventDescriptor {
        guard let first = batch.first else {
            throw PhotosMedia.failure("A Photos keyword batch cannot be empty.")
        }
        let photos = try batch.map { update in
            try objectSpecifier(desiredClass: mediaItemClass, container: .null(),
                                form: OSType(formUniqueID), key: NSAppleEventDescriptor(string: update.assetIdentifier))
        }
        let photoList = NSAppleEventDescriptor.list()
        for (index, photo) in photos.enumerated() {
            photoList.insert(photo, at: index + 1)
        }
        let container = photos.count == 1 ? photos[0] : photoList
        let property = try objectSpecifier(desiredClass: cProperty, container: container,
                                           form: OSType(formPropertyID), key: NSAppleEventDescriptor(typeCode: keywordsProperty))
        let event = NSAppleEventDescriptor(eventClass: kAECoreSuite, eventID: kAESetData,
            targetDescriptor: NSAppleEventDescriptor(bundleIdentifier: "com.apple.Photos"),
            returnID: AEReturnID(kAutoGenerateReturnID), transactionID: AETransactionID(kAnyTransactionID))
        event.setParam(property, forKeyword: keyDirectObject)
        let keywords = NSAppleEventDescriptor.list()
        for (index, keyword) in first.keywords.enumerated() {
            keywords.insert(NSAppleEventDescriptor(string: keyword), at: index + 1)
        }
        event.setParam(keywords, forKeyword: keyAEData)
        return event
    }

    private static func objectSpecifier(desiredClass: OSType, container: NSAppleEventDescriptor,
                                        form: OSType, key: NSAppleEventDescriptor) throws -> NSAppleEventDescriptor {
        let record = NSAppleEventDescriptor.record()
        record.setDescriptor(NSAppleEventDescriptor(typeCode: desiredClass), forKeyword: OSType(keyAEDesiredClass))
        record.setDescriptor(container, forKeyword: OSType(keyAEContainer))
        record.setDescriptor(NSAppleEventDescriptor(enumCode: form), forKeyword: OSType(keyAEKeyForm))
        record.setDescriptor(key, forKeyword: OSType(keyAEKeyData))
        guard let object = record.coerce(toDescriptorType: typeObjectSpecifier) else {
            throw PhotosMedia.failure("Could not prepare the Photos keyword request.")
        }
        return object
    }

    static func write(_ update: PhotosKeywords.Update) throws {
        try send(event(for: [update]))
    }

    /// A photo list that Photos rejects as a whole must not lose the batch:
    /// retry each photo with the proven single-photo event, then surface the
    /// first real failure.
    static func write(_ batch: [PhotosKeywords.Update]) async throws {
        do {
            try send(event(for: batch))
        } catch {
            guard batch.count > 1 else { throw error }
            for update in batch {
                try Task.checkCancellation()
                try send(event(for: [update]))
            }
        }
    }

    private static func send(_ event: NSAppleEventDescriptor) throws {
        let reply = try event.sendEvent(options: [.waitForReply, .canInteract], timeout: 120)
        let code = reply.paramDescriptor(forKeyword: keyErrorNumber)?.int32Value ?? 0
        guard code == 0 else {
            let message = reply.paramDescriptor(forKeyword: keyErrorString)?.stringValue
                ?? "Allow WaifuFinder to control Photos in System Settings → Privacy & Security → Automation."
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(code), userInfo: [NSLocalizedDescriptionKey: message])
        }
    }

    /// Scheduling units for `run`: an asset repeated in the job keeps only its
    /// final keywords — exactly what a sequential run would leave behind — and
    /// photos with one keyword list share a batch, capped at `batchLimit`.
    static func groups(for updates: [PhotosKeywords.Update], batchLimit: Int = PhotosKeywordWriter.batchLimit) -> [[PhotosKeywords.Update]] {
        var slotByAsset: [String: Int] = [:]
        var latest: [PhotosKeywords.Update] = []
        for update in updates {
            if let slot = slotByAsset[update.assetIdentifier] {
                latest[slot] = update
            } else {
                slotByAsset[update.assetIdentifier] = latest.count
                latest.append(update)
            }
        }
        let limit = max(1, batchLimit)
        var buckets: [[String]: [PhotosKeywords.Update]] = [:]
        for update in latest {
            buckets[update.keywords, default: []].append(update)
        }
        var groups: [[PhotosKeywords.Update]] = []
        var seen: Set<[String]> = []
        for update in latest where seen.insert(update.keywords).inserted {
            let bucket = buckets[update.keywords] ?? []
            groups += stride(from: 0, to: bucket.count, by: limit).map {
                Array(bucket[$0 ..< min($0 + limit, bucket.count)])
            }
        }
        return groups
    }

    /// Report only confirmed writes. Cancellation never claims to undo completed
    /// photos; a failure stops new batches but lets the in-flight photos finish.
    static func run(_ groups: [[PhotosKeywords.Update]],
                    workerCount: Int = 4,
                    write: @escaping @Sendable ([PhotosKeywords.Update]) async throws -> Void = { try await write($0) },
                    progress: @escaping @Sendable (Int) async -> Void) async throws {
        guard !groups.isEmpty else { try Task.checkCancellation(); return }
        let cursor = WriteCursor(batches: groups)
        await withTaskGroup(of: Void.self) { workers in
            for _ in 0..<max(1, min(workerCount, groups.count)) {
                workers.addTask {
                    while true {
                        guard !Task.isCancelled, await cursor.stopped == false,
                              let batch = await cursor.next() else { return }
                        do {
                            try Task.checkCancellation()
                            try await write(batch)
                            await progress(await cursor.confirmed(batch.count))
                        } catch is CancellationError {
                            return
                        } catch {
                            await cursor.fail(error)
                            return
                        }
                    }
                }
            }
            while await workers.next() != nil {}
        }
        try Task.checkCancellation()
        if let failure = await cursor.failure { throw failure }
    }
}

/// Hands each batch to exactly one worker and remembers the first failure so
/// every worker stops pulling new work. Photos-side writes are never interrupted.
private actor WriteCursor {
    private let batches: [[PhotosKeywords.Update]]
    private var nextIndex = 0
    private var confirmedCount = 0
    private(set) var failure: Error?

    init(batches: [[PhotosKeywords.Update]]) { self.batches = batches }

    var stopped: Bool { failure != nil }

    func next() -> [PhotosKeywords.Update]? {
        guard nextIndex < batches.count else { return nil }
        defer { nextIndex += 1 }
        return batches[nextIndex]
    }

    func confirmed(_ count: Int) -> Int {
        confirmedCount += count
        return confirmedCount
    }

    func fail(_ error: Error) {
        if failure == nil { failure = error }
    }
}
