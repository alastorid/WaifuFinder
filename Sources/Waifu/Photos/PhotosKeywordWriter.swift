import CoreServices
import Foundation
import AppKit

/// One native set-data event per photo: no large script to compile and no extra read requests.
enum PhotosKeywordWriter {
    static let mediaItemClass: OSType = 0x49506D69 // IPmi, Photos' public scripting dictionary
    static let keywordsProperty: OSType = 0x49506B77 // IPkw

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
        let photo = try objectSpecifier(desiredClass: mediaItemClass, container: .null(),
                                        form: OSType(formUniqueID), key: NSAppleEventDescriptor(string: update.assetIdentifier))
        let property = try objectSpecifier(desiredClass: cProperty, container: photo,
                                           form: OSType(formPropertyID), key: NSAppleEventDescriptor(typeCode: keywordsProperty))
        let event = NSAppleEventDescriptor(eventClass: kAECoreSuite, eventID: kAESetData,
            targetDescriptor: NSAppleEventDescriptor(bundleIdentifier: "com.apple.Photos"),
            returnID: AEReturnID(kAutoGenerateReturnID), transactionID: AETransactionID(kAnyTransactionID))
        event.setParam(property, forKeyword: keyDirectObject)
        let keywords = NSAppleEventDescriptor.list()
        for (index, keyword) in update.keywords.enumerated() {
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
        let reply = try event(for: update).sendEvent(options: [.waitForReply, .canInteract], timeout: 120)
        let code = reply.paramDescriptor(forKeyword: keyErrorNumber)?.int32Value ?? 0
        guard code == 0 else {
            let message = reply.paramDescriptor(forKeyword: keyErrorString)?.stringValue
                ?? "Allow WaifuFinder to control Photos in System Settings → Privacy & Security → Automation."
            throw NSError(domain: NSOSStatusErrorDomain, code: Int(code), userInfo: [NSLocalizedDescriptionKey: message])
        }
    }

    /// Report only confirmed writes. Cancellation never claims to undo completed photos.
    static func run(_ updates: [PhotosKeywords.Update],
                    write: @Sendable (PhotosKeywords.Update) async throws -> Void = { try PhotosKeywordWriter.write($0) },
                    progress: @Sendable (Int) async -> Void) async throws {
        for (index, update) in updates.enumerated() {
            try Task.checkCancellation()
            try await write(update)
            await progress(index + 1)
        }
        try Task.checkCancellation()
    }
}
