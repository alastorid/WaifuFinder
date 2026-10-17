import CoreImage
import Foundation

/// At most four small PhotoKit renditions overlap decoding with Vision. The
/// window advances only as the consumer pulls, keeping large albums bounded.
actor PhotosAlbumSampler {
    private let album: PhotosMedia.Album
    private let width: Int
    private let mode: AnalysisMode
    private var itemIndex = 0
    private var iterator: AsyncThrowingStream<SampledFrame, Error>.Iterator?
    private var pending: [Int: Task<CGImage, Error>] = [:]

    init(album: PhotosMedia.Album, width: Int, mode: AnalysisMode) {
        self.album = album
        self.width = min(width, PhotosMedia.analysisPixelSize)
        self.mode = mode
    }
    deinit { for task in pending.values { task.cancel() } }

    func next() async throws -> SampledFrame? {
        try Task.checkCancellation()
        while itemIndex < album.items.count {
            let item = album.items[itemIndex]
            if item.isStill {
                for index in itemIndex..<min(album.items.count, itemIndex + 4) {
                    let next = album.items[index]
                    guard next.isStill else { break }
                    if pending[index] == nil {
                        let width = width
                        pending[index] = Task {
                            try await PhotosMedia.shared.rendition(next.url, maxPixelSize: width, allowNetwork: PhotosMedia.downloadFromICloud)
                        }
                    }
                }
                let request = pending[itemIndex]!
                let requests = Array(pending.values)
                let image: CGImage
                do {
                    image = try await withTaskCancellationHandler {
                        try await request.value
                    } onCancel: { for task in requests { task.cancel() } }
                } catch {
                    if PhotosMedia.isUnavailableLocally(error) {
                        pending.removeValue(forKey: itemIndex)
                        PhotosMedia.shared.noteUnavailable(album.url)
                        itemIndex += 1
                        continue
                    }
                    for task in pending.values { task.cancel() }
                    pending.removeAll()
                    throw error
                }
                try Task.checkCancellation()
                pending.removeValue(forKey: itemIndex)
                itemIndex += 1
                // A photo is scored/tagged from this rendition. Full-resolution
                // pixels are fetched only for explicit preview or output actions.
                return SampledFrame(sampleIndex: PhotosMedia.frameIndex(seconds: item.offset), timeSeconds: item.offset, image: image)
            }
            if iterator == nil {
                iterator = AVFoundationFrameSampler().sample(item.url, analysisWidth: width, mode: mode, onProgress: { _ in }).makeAsyncIterator()
            }
            var active = iterator!
            let frame: SampledFrame?
            do {
                frame = try await active.next(isolation: self)
            } catch {
                if PhotosMedia.isUnavailableLocally(error) {
                    PhotosMedia.shared.noteUnavailable(album.url)
                    iterator = nil; itemIndex += 1
                    continue
                }
                throw error
            }
            iterator = active
            if let frame {
                return SampledFrame(sampleIndex: PhotosMedia.frameIndex(seconds: item.offset + frame.timeSeconds), timeSeconds: item.offset + frame.timeSeconds, image: frame.image, nativeImage: frame.nativeImage)
            }
            iterator = nil; itemIndex += 1
        }
        return nil
    }
}
