import Foundation

/// What the server can actually do, as reported to a connected client.
///
/// This is `UpscalerCapabilities` verbatim, so `UncoverSession.canReconstruct`
/// and `UncoverSession.limitation` compute correctly on the client against the
/// **server's** limits without a remote-specific branch: a region the server
/// cannot reconstruct is refused in the client's own geometry readout.
struct RemoteCapabilities: Codable, Sendable, Equatable {
    static let protocolVersion = Int(RemoteLimits.protocolVersion)

    var protocolVersion: Int
    var appVersion: String
    var serverName: String
    var sessionID: String?

    var tagger: TaggerCapability?
    /// Every algorithm in `UncoverAlgorithm.allCases` order.
    var uncover: [UncoverCapability]
    var vision: VisionCapability

    struct TaggerCapability: Codable, Sendable, Equatable {
        var available: Bool
        var imageSize: Int
        var batchSize: Int
        var headCount: Int
        /// `TagVocabulary.cacheFingerprint` of the labels the server will use.
        ///
        /// The client uses it to decide whether its cached vocabulary copy is the
        /// one this server speaks for, and stores it beside the CSV so a stale
        /// copy is never mistaken for a current one.
        var vocabularyFingerprint: UInt64?
    }

    struct UncoverCapability: Codable, Sendable, Equatable {
        var algorithm: String
        var available: Bool
        var message: String
        var scales: [Int]
        var minimumFrames: Int
        var requiresUnsafeMemoryOptIn: Bool
        /// `nil` rather than zero when the backend states no limit, so "no limit"
        /// and "a limit of zero" cannot be confused.
        var maximumPaddedArea: Int?
        var maximumOutputPixels: Int?
        var outputScale: Int
        var supportsStillPhotos: Bool
    }

    struct VisionCapability: Codable, Sendable, Equatable {
        /// Always `true`: Vision runs in-process on the client and needs no model
        /// files. Said explicitly so the client does not have to guess.
        var available: Bool
    }

    // MARK: Lookup

    func uncover(_ algorithm: UncoverAlgorithm) -> UncoverCapability? {
        uncover.first { $0.algorithm == algorithm.rawValue }
    }

    /// True when the server can run `algorithm` at `scale`.
    func offersRemote(_ algorithm: UncoverAlgorithm, scale: UncoverScale? = nil) -> Bool {
        guard let capability = uncover(algorithm), capability.available else { return false }
        guard let scale else { return true }
        return capability.scales.contains(scale.rawValue)
    }

    var offersAnyUncover: Bool { uncover.contains { $0.available } }

    var offersRemoteTagging: Bool { tagger?.available == true }

    /// The server's `UpscalerCapabilities` for `algorithm`, translated back into
    /// the local type so `RemoteUncoverBackend.capabilities()` can return it
    /// directly.
    func localCapabilities(_ algorithm: UncoverAlgorithm) -> UpscalerCapabilities? {
        guard let capability = uncover(algorithm),
              UncoverAlgorithm(rawValue: capability.algorithm) != nil else { return nil }
        return UpscalerCapabilities(available: capability.available, message: capability.message,
                                    scales: capability.scales.compactMap(UncoverScale.init(rawValue:)),
                                    minimumFrames: capability.minimumFrames,
                                    requiresUnsafeMemoryOptIn: capability.requiresUnsafeMemoryOptIn,
                                    maximumPaddedArea: capability.maximumPaddedArea,
                                    maximumOutputPixels: capability.maximumOutputPixels)
    }

    /// Reads the current model state from this Mac's own library.
    ///
    /// Only `ManagedModel.offered` appears, so a build that is not licensed to
    /// offer a model reports it as **absent** rather than present-and-unavailable —
    /// the same decision `ModelLibrary` already makes in the Settings tab.
    static func current(appVersion: String, serverName: String, sessionID: String?) async -> Self {
        var tagger: TaggerCapability?
        if WDTaggerConfiguration.load() != nil {
            // The fingerprint has to come from the labels the server will actually
            // use, which is `WDTaggerConfiguration.load()`'s directory — the same
            // directory `ModelLibrary` resolved the entry from.
            let fingerprint = try? await RemoteServerResources.shared.vocabularyFingerprint()
            tagger = TaggerCapability(available: true, imageSize: WDTaggerBackend.imageSize,
                                      batchSize: WDTaggerBackend.batchSize,
                                      headCount: TagVocabulary.headCount,
                                      vocabularyFingerprint: fingerprint)
        }
        var uncover: [UncoverCapability] = []
        for algorithm in UncoverAlgorithm.allCases {
            guard ManagedModel.offered.contains(where: { $0.algorithm == algorithm }) else { continue }
            let local = await algorithm.makeBackend().capabilities()
            uncover.append(UncoverCapability(
                algorithm: algorithm.rawValue, available: local.available,
                message: local.message,
                scales: local.scales.map(\.rawValue), minimumFrames: local.minimumFrames,
                requiresUnsafeMemoryOptIn: local.requiresUnsafeMemoryOptIn,
                maximumPaddedArea: local.maximumPaddedArea, maximumOutputPixels: local.maximumOutputPixels,
                outputScale: algorithm == .flashVSR11 ? FlashVSRBackend.outputScale : 1,
                supportsStillPhotos: algorithm.supportsStillPhotos))
        }
        return Self(protocolVersion: protocolVersion, appVersion: appVersion, serverName: serverName,
                    sessionID: sessionID, tagger: tagger, uncover: uncover,
                    vision: VisionCapability(available: true))
    }
}

/// How the client renders an autodiscovered or configured server's key state.
enum RemoteKeyState: Equatable, Sendable {
    case none
    case matches
    case mismatched

    var label: String {
        switch self {
        case .none: "Key needed"
        case .matches: "Ready"
        case .mismatched: "Key changed"
        }
    }
}