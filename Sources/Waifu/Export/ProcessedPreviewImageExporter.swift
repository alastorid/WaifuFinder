import CoreGraphics
import Foundation
import ImageIO

enum ProcessedPreviewImageExporter {
    static func write(_ image: CGImage, sourceStem: String, frameID: FrameID,
                      suffix: String, directory: URL, format: ExportFormat,
                      jpegQuality: Float) throws -> ExportResult {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                throw ExportError.destinationNotADirectory(directory)
            }
        } else {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        let padded = String(format: "%06lld", Int64(frameID.sampleIndex))
        let fileName = "\(sourceStem)-\(padded)-\(suffix).\(format.fileExtension)"
        let baseURL = directory.appendingPathComponent(fileName)
        let destinationURL = uniqueDestinationURL(baseURL)
        guard let destination = CGImageDestinationCreateWithURL(
            destinationURL as CFURL, format.contentType as CFString, 1, nil
        ) else {
            throw ExportError.writeFailed(frameID: frameID, url: destinationURL,
                                          underlying: "Failed to create image destination")
        }

        var properties: [CFString: Any] = [:]
        if format == .jpeg {
            properties[kCGImageDestinationLossyCompressionQuality] = jpegQuality
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: destinationURL)
            throw ExportError.writeFailed(frameID: frameID, url: destinationURL,
                                          underlying: "Image export could not be completed")
        }

        let attributes = try FileManager.default.attributesOfItem(atPath: destinationURL.path)
        let byteCount = attributes[.size] as? Int ?? 0
        return ExportResult(frameID: frameID, url: destinationURL, byteCount: byteCount)
    }

    private static func uniqueDestinationURL(_ baseURL: URL) -> URL {
        guard FileManager.default.fileExists(atPath: baseURL.path) else { return baseURL }
        let directory = baseURL.deletingLastPathComponent()
        let stem = baseURL.deletingPathExtension().lastPathComponent
        let suffix = baseURL.pathExtension
        for index in 1...10_000 {
            let candidate = directory.appendingPathComponent("\(stem)-\(index).\(suffix)")
            if !FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return directory.appendingPathComponent("\(stem)-\(UUID().uuidString).\(suffix)")
    }
}
