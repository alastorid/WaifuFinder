import Foundation
import CoreGraphics

/// A `TaggerBackend` that sends the frames to the server and returns its rows.
///
/// This is the whole remote-tagging integration: `WaifuSession.runTaggingPipeline`
/// chooses between this and `WDTaggerBackend` at the factory line, and every
/// other piece of behaviour — batching, prefetch, progress, cancellation, the
/// tag cache — is written against `TaggerBackend` and does not change.
actor RemoteTaggerBackend: TaggerBackend {
    private let client: RemoteClient

    private let threshold: Float
    private let fingerprint: UInt64?

    init(client: RemoteClient, threshold: Float, vocabularyFingerprint: UInt64?) {
        self.client = client
        self.threshold = threshold
        fingerprint = vocabularyFingerprint
    }

    func tag(batch: [CGImage]) async throws -> [TagResult] {
        guard let vocabulary = await client.remoteVocabulary, vocabulary.cacheFingerprint == fingerprint else {
            // Without the labels there is no rating, no top-tag list and no
            // filter vocabulary, so a row alone would be unusable.
            throw RemoteError.modelUnavailable(
                "The server has not sent its tag labels yet. Reconnect to resync.")
        }
        let rows = try await client.tag(images: batch, codec: .rgba8)
        return try rows.map { try $0.result(vocabulary: vocabulary, threshold: threshold) }
    }
}

