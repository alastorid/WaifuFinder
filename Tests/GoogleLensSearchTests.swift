import AppKit
import Foundation
import ImageIO

func registerGoogleLensSearchTests() {
    Registry.shared.add(suite: "Google Lens", TestCase(name: "online search defaults off and remembers an explicit choice", knownBug: nil) {
        let name = "WaifuFinder-Lens-Tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        check(!GoogleLensSearch.isEnabled(in: defaults), "a fresh settings store disables Lens")
        defaults.set(true, forKey: GoogleLensSearch.enabledKey)
        check(GoogleLensSearch.isEnabled(in: defaults), "explicit opt-in enables Lens")
        defaults.set(false, forKey: GoogleLensSearch.enabledKey)
        check(!GoogleLensSearch.isEnabled(in: defaults), "switching off disables Lens again")
    })

    Registry.shared.add(suite: "Google Lens", TestCase(name: "small thumbnails remain usable and larger images preserve their aspect ratio", knownBug: nil) {
        for (width, height, expectedWidth, expectedHeight) in [
            (320, 180, 320, 180), (180, 320, 180, 320),
            (3000, 2000, 1000, 667), (2000, 3000, 667, 1000)
        ] {
            let image = try ImageFixture.cgImage(width: width, height: height, gray: 0.5)
            let prepared = try GoogleLensSearch.prepare(image)
            guard let source = CGImageSourceCreateWithData(prepared.data as CFData, nil),
                  let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                check(false, "search image must be a decodable JPEG")
                continue
            }
            checkEqual(CGImageSourceGetType(source) as String?, "public.jpeg", "Lens gets a JPEG")
            checkEqual(decoded.width, expectedWidth, "encoded width")
            checkEqual(decoded.height, expectedHeight, "encoded height")
            checkEqual(prepared.width, decoded.width, "form width matches the encoded bytes")
            checkEqual(prepared.height, decoded.height, "form height matches the encoded bytes")
        }
    })

    Registry.shared.add(suite: "Google Lens", TestCase(name: "browser form contains the exact JPEG rather than a result URL from another session", knownBug: nil) {
        let image = try ImageFixture.cgImage(width: 320, height: 180, gray: 0.7)
        let prepared = try GoogleLensSearch.prepare(image)
        let html = GoogleLensSearch.uploadHTML(jpeg: prepared.data, width: prepared.width, height: prepared.height)
        let expression = try NSRegularExpression(pattern: "atob\\('([A-Za-z0-9+/=]+)'\\)")
        let range = NSRange(html.startIndex..., in: html)
        guard let match = expression.firstMatch(in: html, range: range),
              let payloadRange = Range(match.range(at: 1), in: html) else {
            check(false, "browser page contains the image bytes")
            return
        }
        checkEqual(Data(base64Encoded: String(html[payloadRange])), prepared.data, "browser uploads the exact generated JPEG")
        check(html.contains("method=\"POST\" enctype=\"multipart/form-data\""), "browser owns the multipart navigation")
        check(html.contains("https://lens.google.com/v3/upload"), "use Lens's direct upload endpoint")
        check(html.contains("name=\"encoded_image\" type=\"file\""), "upload uses a file part")
        check(html.contains("value=\"320,180\""), "image dimensions travel with the upload")
        check(html.contains("String(Date.now())"), "each submission gets a fresh start time")
        check(html.contains("form.requestSubmit()"), "page navigates through its upload form")
        check(!html.contains("fetch("), "upload and redirects stay in the same browser navigation")
    })

    Registry.shared.add(suite: "Google Lens", TestCase(name: "temporary image pages are private and stale pages are removed without deleting recent pages", knownBug: nil) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-lens-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let stale = directory.appendingPathComponent("stale.html")
        let recent = directory.appendingPathComponent("recent.html")
        try Data().write(to: stale)
        try Data().write(to: recent)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-2 * 24 * 60 * 60)], ofItemAtPath: stale.path)
        let page = try GoogleLensSearch.writeUploadPage(jpeg: Data([0, 127, 255]), width: 1, height: 1, directory: directory)
        check(!FileManager.default.fileExists(atPath: stale.path), "expired image page is removed")
        check(FileManager.default.fileExists(atPath: recent.path), "recent page survives a delayed browser launch")
        let attributes = try FileManager.default.attributesOfItem(atPath: page.path)
        checkEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600, "embedded image is readable only by its owner")
    })
}
