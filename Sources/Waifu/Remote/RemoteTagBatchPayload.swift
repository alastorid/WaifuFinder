import CoreGraphics
import Compression
import Foundation

/// Model input preparation runs independently in each server lane. Cached video
/// thumbnails travel verbatim; Photos retain the exact lossless RGBA input.
enum RemoteTagBatchPayload {
    /// Leave the connection's dispatcher free to receive the next tagging batch
    /// while ImageIO decodes this one on the generic executor.
    static func decode(_ payload: [UInt8]) async throws -> [CGImage] {
        var reader = WireReader(payload)
        let count = Int(try reader.u8())
        guard let codec = RemoteTagImageCodec(rawValue: try reader.u8()), try reader.u16() == 0 else {
            throw RemoteError.invalidRequest("Unknown image codec or reserved flags.")
        }
        guard (1...WDTaggerBackend.batchSize).contains(count) else { throw TaggerError.invalidBatch }
        var images: [CGImage] = []
        images.reserveCapacity(count)
        for _ in 0..<count {
            try Task.checkCancellation()
            images.append(try RemoteTagImageDecoder.decode(codec: codec, bytes: reader.raw()))
        }
        guard reader.isAtEnd else { throw RemoteError.framing("trailing bytes after a tagging batch") }
        return images
    }

    static func encode(sources: [TaggingImageSource], losslessCodec: RemoteTagImageCodec = .rgba8) async throws -> [UInt8] {
        guard (1...WDTaggerBackend.batchSize).contains(sources.count) else { throw TaggerError.invalidBatch }
        guard losslessCodec == .rgba8 || losslessCodec == .rgba8LZ4 else { throw TaggerError.preprocessing }
        var thumbnails: [Data] = []
        for source in sources {
            try Task.checkCancellation()
            switch source {
            case .thumbnail(let data): thumbnails.append(data)
            case .photo:
                let images = try await TaggingImageLoader.load(sources)
                return try await encode(images: images, codec: losslessCodec)
            }
        }
        return try encoded(thumbnails, codec: .encoded)
    }

    static func encode(images: [CGImage], codec: RemoteTagImageCodec) async throws -> [UInt8] {
        guard (1...WDTaggerBackend.batchSize).contains(images.count) else { throw TaggerError.invalidBatch }
        var frames: [Data] = []
        for image in images {
            try Task.checkCancellation()
            switch codec {
            case .encoded: frames.append(try ThumbnailCache.encodeJPEG(image))
            case .rgba8: frames.append(Data(try RemoteTagImageDecoder.encodeRGBA8(image)))
            case .rgba8LZ4:
                frames.append(Data(try RemoteTagInputCompression.encode(RemoteTagImageDecoder.encodeRGBA8(image))))
            }
        }
        return try encoded(frames, codec: codec)
    }

    private static func encoded(_ frames: [Data], codec: RemoteTagImageCodec) throws -> [UInt8] {
        let size = 4 + frames.count * 4 + frames.reduce(0) { $0 + $1.count }
        guard size <= RemoteLimits.maximumMessagePayload else { throw RemoteError.tooLarge }
        var writer = WireWriter(capacity: size)
        writer.u8(UInt8(frames.count))
        writer.u8(codec.rawValue)
        writer.u16(0)
        for frame in frames {
            try Task.checkCancellation()
            writer.data(frame)
        }
        return writer.bytes
    }
}

/// One fixed-size model input, with a byte selecting raw (0) or LZ4 (1).
/// Incompressible inputs stay raw rather than growing on the wire. Decode into
/// a fixed, bounded buffer and require the exact 448 × 448 × 4 byte count.
enum RemoteTagInputCompression {
    static let byteCount = WDTaggerBackend.imageSize * WDTaggerBackend.imageSize * 4

    static func encode(_ bytes: [UInt8]) throws -> [UInt8] {
        guard bytes.count == byteCount else { throw TaggerError.preprocessing }
        var output = [UInt8](repeating: 0, count: byteCount)
        let size = output.withUnsafeMutableBufferPointer { destination in
            bytes.withUnsafeBufferPointer { source in
                compression_encode_buffer(destination.baseAddress!, destination.count,
                                          source.baseAddress!, source.count, nil, COMPRESSION_LZ4)
            }
        }
        guard size > 0, size < byteCount else { return [0] + bytes }
        return [1] + output.prefix(size)
    }

    static func decode(_ bytes: [UInt8]) throws -> [UInt8] {
        guard let mode = bytes.first, bytes.count <= byteCount + 1 else { throw TaggerError.preprocessing }
        if mode == 0 {
            guard bytes.count == byteCount + 1 else { throw TaggerError.preprocessing }
            return Array(bytes.dropFirst())
        }
        guard mode == 1, bytes.count > 1, bytes.count <= byteCount else { throw TaggerError.preprocessing }
        var output = [UInt8](repeating: 0, count: byteCount + 1)
        let size = output.withUnsafeMutableBufferPointer { destination in
            bytes.withUnsafeBufferPointer { source in
                compression_decode_buffer(destination.baseAddress!, destination.count,
                                          source.baseAddress!.advanced(by: 1), source.count - 1, nil, COMPRESSION_LZ4)
            }
        }
        guard size == byteCount else { throw TaggerError.preprocessing }
        output.removeLast()
        return output
    }
}
