import CoreImage
import Foundation

/// At most four small PhotoKit renditions overlap decoding with Vision. The
/// window advances only as the consumer pulls, keeping large albums bounded.
actor PhotosAlbumSampler {
    struct Item: Sendable {
        let url: URL
        let offset: Double
        let isStill: Bool
    }
    private let items: [Item]
    private let width: Int
    private let mode: AnalysisMode
    private let allowNetwork: Bool
    private let photoRendition: @Sendable (URL, Int, Bool) async throws -> CGImage
    private(set) var unavailableItemCount = 0
    private var itemIndex = 0
    private var iterator: AsyncThrowingStream<SampledFrame, Error>.Iterator?
    private var pending: [Int: Task<CGImage, Error>] = [:]
    private var cancelled = false

    init(album: PhotosMedia.Album, width: Int, mode: AnalysisMode,
         completedIndices: Set<Int> = [], allowNetwork: Bool = PhotosMedia.downloadFromICloud) {
        self.init(items: album.items.map { Item(url: $0.url, offset: $0.offset, isStill: $0.isStill) },
                  width: width, mode: mode, completedIndices: completedIndices, allowNetwork: allowNetwork)
    }

    init(items: [Item], width: Int, mode: AnalysisMode, completedIndices: Set<Int> = [], allowNetwork: Bool,
         photoRendition: @escaping @Sendable (URL, Int, Bool) async throws -> CGImage = { url, width, allowNetwork in
             try await PhotosMedia.shared.rendition(url, maxPixelSize: width, allowNetwork: allowNetwork)
         }) {
        // Compact the retry list first so four widely separated gaps can still
        // overlap their requests. Original offsets keep frame identities stable.
        self.items = items.filter {
            !$0.isStill || !completedIndices.contains(PhotosMedia.frameIndex(seconds: $0.offset))
        }
        self.width = min(width, PhotosMedia.analysisPixelSize)
        self.mode = mode
        self.allowNetwork = allowNetwork
        self.photoRendition = photoRendition
    }
    deinit { for task in pending.values { task.cancel() } }

    func next() async throws -> SampledFrame? {
        do {
            return try await nextFrame()
        } catch {
            await cancel()
            throw error
        }
    }

    /// Unstructured rendition tasks must finish cancelling before a replacement
    /// sampler is allowed to start requests under a different network policy.
    func cancel() async {
        cancelled = true
        let requests = Array(pending.values)
        pending.removeAll()
        for request in requests { request.cancel() }
        for request in requests { _ = await request.result }
        iterator = nil
    }

    private func nextFrame() async throws -> SampledFrame? {
        try Task.checkCancellation()
        if cancelled { throw CancellationError() }
        while itemIndex < items.count {
            let item = items[itemIndex]
            if item.isStill {
                for index in itemIndex..<min(items.count, itemIndex + 4) {
                    let next = items[index]
                    guard next.isStill else { break }
                    if pending[index] == nil {
                        let width = width
                        let allowNetwork = allowNetwork
                        let photoRendition = photoRendition
                        pending[index] = Task {
                            try await photoRendition(next.url, width, allowNetwork)
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
                        unavailableItemCount += 1
                        itemIndex += 1
                        continue
                    }
                    throw error
                }
                try Task.checkCancellation()
                if cancelled { throw CancellationError() }
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
                    unavailableItemCount += 1
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
