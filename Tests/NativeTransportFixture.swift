import AVFoundation
import CoreMedia
import Foundation

extension VideoFixture {
    /// Minimal original MPEG-TS fixture muxer. Apple encodes the H.264 samples;
    /// PAT/PMT/PES packet framing is written here without a third-party codec.
    static func makeTransportStream(from source: URL, to output: URL) async throws {
        let asset = AVURLAsset(url: source)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw failure("Transport fixture needs video") }
        let reader = try AVAssetReader(asset: asset)
        let samples = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(samples)
        guard reader.startReading() else { throw reader.error ?? failure("Cannot read native H.264 samples") }
        var stream = Data()
        var continuity: [Int: UInt8] = [:]
        func packetize(_ bytes: [UInt8], pid: Int, timestamp: Int64? = nil) {
            var cursor = 0
            while cursor < bytes.count {
                let first = cursor == 0
                let hasPCR = first && timestamp != nil
                let amount = min(184 - (hasPCR ? 8 : 0), bytes.count - cursor)
                let adaptation = 184 - amount
                let counter = continuity[pid, default: 0]
                continuity[pid] = (counter + 1) & 15
                var packet: [UInt8] = [0x47, UInt8((pid >> 8) & 31) | (first ? 0x40 : 0), UInt8(pid & 255), (adaptation > 0 ? 0x30 : 0x10) | counter]
                if adaptation > 0 {
                    packet.append(UInt8(adaptation - 1))
                    if adaptation > 1 {
                        packet.append(hasPCR ? 0x10 : 0)
                        if hasPCR, let timestamp {
                            let base = UInt64(max(0, timestamp - 9_000))
                            packet += [UInt8(truncatingIfNeeded: base >> 25), UInt8(truncatingIfNeeded: base >> 17),
                                       UInt8(truncatingIfNeeded: base >> 9), UInt8(truncatingIfNeeded: base >> 1),
                                       UInt8((base & 1) << 7) | 0x7e, 0]
                        }
                        while packet.count < 4 + adaptation { packet.append(0xff) }
                    }
                }
                packet += bytes[cursor..<(cursor + amount)]
                stream.append(contentsOf: packet)
                cursor += amount
            }
        }
        func table(_ body: [UInt8]) -> [UInt8] {
            var crc: UInt32 = 0xffff_ffff
            for byte in body {
                crc ^= UInt32(byte) << 24
                for _ in 0..<8 { crc = crc & 0x8000_0000 == 0 ? crc << 1 : (crc << 1) ^ 0x04c1_1db7 }
            }
            return [0] + body + [UInt8(truncatingIfNeeded: crc >> 24), UInt8(truncatingIfNeeded: crc >> 16), UInt8(truncatingIfNeeded: crc >> 8), UInt8(truncatingIfNeeded: crc)]
        }
        while let sample = samples.copyNextSampleBuffer() {
            try Task.checkCancellation()
            // AVAssetReader also emits empty edit/trim marker buffers. They are
            // timeline metadata, not compressed access units for a PES packet.
            if CMSampleBufferGetNumSamples(sample) == 0 { continue }
            packetize(table([0, 0xb0, 13, 0, 1, 0xc1, 0, 0, 0, 1, 0xe1, 0]), pid: 0)
            packetize(table([2, 0xb0, 18, 0, 1, 0xc1, 0, 0, 0xe1, 1, 0xf0, 0, 0x1b, 0xe1, 1, 0xf0, 0]), pid: 0x100)
            guard let format = CMSampleBufferGetFormatDescription(sample), let block = CMSampleBufferGetDataBuffer(sample) else { throw failure("Missing compressed sample") }
            var nalHeaderLength: Int32 = 0
            var count = 0
            var elementary: [UInt8] = [0, 0, 0, 1, 9, 0xf0]
            for index in 0..<2 {
                var pointer: UnsafePointer<UInt8>?
                var size = 0
                let status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(format, parameterSetIndex: index,
                    parameterSetPointerOut: &pointer, parameterSetSizeOut: &size,
                    parameterSetCountOut: &count, nalUnitHeaderLengthOut: &nalHeaderLength)
                guard status == noErr, let pointer else { throw failure("Missing native H.264 parameter sets") }
                elementary += [0, 0, 0, 1] + Array(UnsafeBufferPointer(start: pointer, count: size))
            }
            let length = CMBlockBufferGetDataLength(block)
            var avcc = [UInt8](repeating: 0, count: length)
            let copyStatus = avcc.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            guard copyStatus == noErr, (1...4).contains(nalHeaderLength) else { throw failure("Invalid compressed H.264 data") }
            var cursor = 0
            while cursor < avcc.count {
                let header = Int(nalHeaderLength)
                guard cursor + header <= avcc.count else { throw failure("Truncated H.264 sample") }
                var size = 0
                for offset in 0..<header { size = size << 8 | Int(avcc[cursor + offset]) }
                cursor += header
                guard size <= avcc.count - cursor else { throw failure("Truncated H.264 NAL unit") }
                elementary += [0, 0, 0, 1] + avcc[cursor..<(cursor + size)]
                cursor += size
            }
            let timestamp = UInt64(90_000 + Int64((CMSampleBufferGetPresentationTimeStamp(sample).seconds * 90_000).rounded()))
            let pts: [UInt8] = [0x21 | UInt8((timestamp >> 29) & 14), UInt8(truncatingIfNeeded: timestamp >> 22),
                                UInt8((timestamp >> 14) & 0xfe) | 1, UInt8(truncatingIfNeeded: timestamp >> 7), UInt8((timestamp << 1) & 0xfe) | 1]
            packetize([0, 0, 1, 0xe0, 0, 0, 0x80, 0x80, 5] + pts + elementary, pid: 0x101, timestamp: Int64(timestamp))
        }
        guard reader.status == .completed else { throw reader.error ?? failure("Native transport fixture read failed") }
        try stream.write(to: output)
    }

    static func makeHLS(from source: URL, in directory: URL) async throws {
        try await makeTransportStream(from: source, to: directory.appendingPathComponent("segment0.ts"))
        let duration = try await AVURLAsset(url: source).load(.duration).seconds
        let playlist = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:\(Int(ceil(duration)))\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:\(duration),\nsegment0.ts\n#EXT-X-ENDLIST\n"
        try Data(playlist.utf8).write(to: directory.appendingPathComponent("playlist"))
    }
}
