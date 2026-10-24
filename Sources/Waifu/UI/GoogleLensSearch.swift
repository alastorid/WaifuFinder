import AppKit
import Foundation
import ImageIO
import SwiftUI

/// The browser owns the upload and its resulting Lens session. Uploading with
/// URLSession first loses the browser's session when handing off the result URL.
enum GoogleLensSearch {
    static let enabledKey = "googleLensEnabled"

    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: enabledKey)
    }

    @MainActor
    static func search(frame: AnalyzedFrame, preferredImage: CGImage? = nil, in session: WaifuSession) async {
        guard isEnabled() else { return }
        do {
            let identity = session.thumbnailGeneration.uuidString + ":" + frame.thumbnailKey
            if let image = preferredImage ?? GridThumbnailLoader.shared.cachedImage(identity: identity) {
                try await open(image: image)
            } else if let data = await session.thumbnailData(for: frame, retainRegeneratedThumbnail: false) {
                let image = try await Task.detached(priority: .userInitiated) {
                    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                        throw SearchError.invalidImage
                    }
                    return image
                }.value
                try await open(image: image)
            } else {
                throw SearchError.invalidImage
            }
        } catch {
            session.errorMessage = "Couldn’t search with Google Lens: \(error.localizedDescription)"
        }
    }

    @MainActor
    static func search(image: CGImage, in session: WaifuSession) async {
        guard isEnabled() else { return }
        do {
            try await open(image: image)
        } catch {
            session.errorMessage = "Couldn’t search with Google Lens: \(error.localizedDescription)"
        }
    }

    @MainActor
    private static func open(image: CGImage) async throws {
        let fileURL = try await Task.detached(priority: .userInitiated) {
            let prepared = try prepare(image)
            return try writeUploadPage(jpeg: prepared.data, width: prepared.width, height: prepared.height)
        }.value
        // The setting may have been turned off while preparing the image.
        guard isEnabled() else {
            try? FileManager.default.removeItem(at: fileURL)
            return
        }
        let lensURL = URL(string: "https://lens.google.com")!
        guard let browser = NSWorkspace.shared.urlForApplication(toOpen: lensURL) else {
            try? FileManager.default.removeItem(at: fileURL)
            throw SearchError.couldNotOpenBrowser
        }
        do {
            // Explicitly use the default web browser; .html files may otherwise
            // open in the user's source editor instead of submitting the form.
            _ = try await NSWorkspace.shared.open([fileURL], withApplicationAt: browser,
                                                  configuration: NSWorkspace.OpenConfiguration())
        } catch {
            try? FileManager.default.removeItem(at: fileURL)
            throw error
        }
    }

    static func prepare(_ image: CGImage) throws -> (data: Data, width: Int, height: Int) {
        let longestSide = max(image.width, image.height)
        guard longestSide > 0 else { throw SearchError.invalidImage }
        let scale = min(1, 1000.0 / Double(longestSide))
        let width = max(1, Int((Double(image.width) * scale).rounded()))
        let height = max(1, Int((Double(image.height) * scale).rounded()))
        guard let context = CGContext(data: nil, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw SearchError.invalidImage
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resized = context.makeImage(),
              let data = NSBitmapImageRep(cgImage: resized).representation(
                using: .jpeg, properties: [.compressionFactor: 0.8]) else {
            throw SearchError.invalidImage
        }
        return (data, resized.width, resized.height)
    }

    /// A normal form navigation keeps cookies and redirects inside the browser.
    /// Only generated JPEG bytes and integer dimensions are embedded in this page.
    static func uploadHTML(jpeg: Data, width: Int, height: Int) -> String {
        """
        <!doctype html>
        <html lang="en">
        <meta charset="utf-8">
        <meta name="referrer" content="no-referrer">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Search with Google Lens</title>
        <body>
        <p id="status">Opening Google Lens…</p>
        <form id="lens-upload" action="https://lens.google.com/v3/upload" method="POST" enctype="multipart/form-data">
          <input id="image" name="encoded_image" type="file" accept="image/jpeg" hidden>
          <input name="processed_image_dimensions" type="hidden" value="\(width),\(height)">
          <button type="submit">Search with Google Lens</button>
        </form>
        <noscript>Enable JavaScript in your browser to search with this image.</noscript>
        <script>
        (() => {
          const form = document.getElementById('lens-upload');
          const status = document.getElementById('status');
          try {
            const bytes = Uint8Array.from(atob('\(jpeg.base64EncodedString())'), c => c.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'image.jpg', {type: 'image/jpeg'}));
            document.getElementById('image').files = transfer.files;
            form.addEventListener('submit', () => {
              const action = new URL('https://lens.google.com/v3/upload');
              action.searchParams.set('ep', 'ccm');
              action.searchParams.set('st', String(Date.now()));
              action.searchParams.set('hl', navigator.language || 'en');
              action.searchParams.set('vpw', '\(width)');
              action.searchParams.set('vph', '\(height)');
              form.action = action.toString();
              status.textContent = 'Opening Google Lens…';
            });
            form.requestSubmit();
          } catch (error) {
            status.textContent = 'Could not open Google Lens. Return to WaifuFinder and try another browser.';
          }
        })();
        </script>
        </body>
        </html>
        """
    }

    static func writeUploadPage(jpeg: Data, width: Int, height: Int,
                                directory: URL = FileManager.default.temporaryDirectory
                                    .appendingPathComponent("WaifuFinder-GoogleLens", isDirectory: true)) throws -> URL {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true,
                                    attributes: [.posixPermissions: 0o700])
        // Retain a page long enough for a delayed browser launch or a retry, then
        // discard old embedded images on the next search. macOS also manages tmp.
        let cutoff = Date().addingTimeInterval(-24 * 60 * 60)
        let oldPages = try manager.contentsOfDirectory(at: directory,
                                                       includingPropertiesForKeys: [.contentModificationDateKey])
        for page in oldPages where page.pathExtension == "html" {
            if let modified = try? page.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate,
               modified < cutoff {
                try? manager.removeItem(at: page)
            }
        }
        let fileURL = directory.appendingPathComponent(UUID().uuidString).appendingPathExtension("html")
        guard manager.createFile(atPath: fileURL.path,
                                 contents: Data(uploadHTML(jpeg: jpeg, width: width, height: height).utf8),
                                 attributes: [.posixPermissions: 0o600]) else {
            throw SearchError.couldNotPrepareUpload
        }
        return fileURL
    }

    private enum SearchError: LocalizedError {
        case invalidImage, couldNotPrepareUpload, couldNotOpenBrowser
        var errorDescription: String? {
            switch self {
            case .invalidImage: "The selected image is unavailable."
            case .couldNotPrepareUpload: "The selected image could not be prepared for Google Lens."
            case .couldNotOpenBrowser: "macOS could not open your default web browser."
            }
        }
    }
}

extension View {
    /// Avoid installing a contextual menu at all when it has no available items.
    @ViewBuilder
    func contextMenu<MenuItems: View>(isEnabled: Bool, @ViewBuilder menuItems: () -> MenuItems) -> some View {
        if isEnabled { contextMenu(menuItems: menuItems) }
        else { self }
    }

    func googleLensContextMenu(isAvailable: Bool = true, action: @escaping () -> Void) -> some View {
        modifier(GoogleLensContextMenuModifier(isAvailable: isAvailable, action: action))
    }
}

private struct GoogleLensContextMenuModifier: ViewModifier {
    @AppStorage(GoogleLensSearch.enabledKey) private var enabled = false
    let isAvailable: Bool
    let action: () -> Void

    func body(content: Content) -> some View {
        content.contextMenu(isEnabled: enabled && isAvailable) {
            Button("Search with Google Lens", action: action)
        }
    }
}
