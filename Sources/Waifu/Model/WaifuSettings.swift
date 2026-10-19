import Foundation

/// Every knob Waifu exposes, in one persisted place.
///
/// Grouped by the layer that reads it rather than alphabetically, so the
/// defaults sit next to the decision they default. `Codable` and stored as JSON
/// because these are *preferences*: a settings file that fails to decode should
/// fall back to the defaults rather than lose the user their video.
///
/// Every consumer reads a **validated** copy (`WaifuSettings.validated()`) and
/// never the raw value. A hand-edited JSON file with `analysisWidth: 0` must not
/// be able to make the next pass decode a zero-pixel image, and the only place
/// that guarantee can be enforced once is here.
struct WaifuSettings: Sendable, Equatable, Codable {
    // MARK: Analysis

    /// How much of the video one analysis pass covers.
    ///
    /// **Full** — the default — analyses every frame. **Quick**
    /// analyses one frame a second: the decode is the same
    /// front-to-back pass either way, so the mode buys back
    /// Vision work, not decode. That is why the choice is made
    /// before a video loads, on the drop zone, rather than as
    /// a knob on a finished analysis.
    var analysisMode: AnalysisMode = .full
    /// Kept to read older settings files. Tagging now starts from its UI panels.
    var taggerEnabled: Bool = false
    var taggerThreshold: Float = 0.35
    var tagAdvancedMode = true
    /// 99 means all matching frames; applied only in Advanced mode.
    var tagTopCount = 99

    // MARK: Decode

    /// Width, in pixels, at which frames are decoded for analysis.
    ///
    /// **512 is measured, not chosen for tidiness.** Apple's aesthetics model is
    /// trained on and evaluated at roughly this scale, and Vision's own
    /// documentation describes the request as operating on the image as
    /// supplied. Decoding a 1920-px-wide frame to feed a 512-px model is four
    /// times the pixel traffic for the same answer, and the analysis pass runs
    /// over thousands of frames. The export pass re-extracts at native
    /// resolution, so nothing is lost where it is visible.
    var analysisWidth: Int = 512

    // MARK: Grouping

    /// Maximum FeaturePrint distance for two frames to be the same composition.
    var similarityThreshold: Float = 0.35

    /// Largest group accepted before it is split into contiguous runs.
    var maximumGroupSize: Int = 60

    /// Weight given to face capture quality when ranking, 0 = pure art.
    var faceWeight: Double = 0.5

    /// A face counts only if it covers at least this fraction of the frame.
    var minimumFaceAreaFraction: Float = 0.01

    // MARK: Navigation

    /// How many frames of each group the grid keeps by default.
    ///
    /// Groups are runs of near-identical frames — a single shot, a slow pan — so a
    /// group of 40 contributes one moment, not 40 rows. Seven is enough to choose
    /// between: the best few by whatever the order is, with room to see that the
    /// group really is one shot. The rest are not lost; the grid can reveal a group in place.
    var perGroupFrameCount: Int = 7
    /// Lift every group cap while retaining the chosen limited count.
    var revealAllGroups = false

    // MARK: Export

    var exportFormat: ExportFormat = .png
    /// 0…1, JPEG only.
    var exportJPEGQuality: Float = 0.92

    static let `default` = WaifuSettings()

    init() {}

    private enum CodingKeys: String, CodingKey {
        case taggerEnabled, taggerThreshold, tagAdvancedMode, tagTopCount, analysisMode, analysisWidth, similarityThreshold, maximumGroupSize,
             faceWeight, minimumFaceAreaFraction, perGroupFrameCount, revealAllGroups, exportFormat, exportJPEGQuality
    }

    /// Older settings predate analysisMode. Keep every saved preference when
    /// introducing the new field instead of discarding the whole settings file.
    init(from decoder: Decoder) throws {
        self.init()
        let values = try decoder.container(keyedBy: CodingKeys.self)
        analysisMode = try values.decodeIfPresent(AnalysisMode.self, forKey: .analysisMode) ?? .full
        taggerEnabled = try values.decodeIfPresent(Bool.self, forKey: .taggerEnabled) ?? false
        taggerThreshold = try values.decodeIfPresent(Float.self, forKey: .taggerThreshold) ?? 0.35
        tagAdvancedMode = try values.decodeIfPresent(Bool.self, forKey: .tagAdvancedMode) ?? true
        tagTopCount = try values.decodeIfPresent(Int.self, forKey: .tagTopCount) ?? 99
        analysisWidth = try values.decodeIfPresent(Int.self, forKey: .analysisWidth) ?? analysisWidth
        similarityThreshold = try values.decodeIfPresent(Float.self, forKey: .similarityThreshold) ?? similarityThreshold
        maximumGroupSize = try values.decodeIfPresent(Int.self, forKey: .maximumGroupSize) ?? maximumGroupSize
        faceWeight = try values.decodeIfPresent(Double.self, forKey: .faceWeight) ?? faceWeight
        minimumFaceAreaFraction = try values.decodeIfPresent(Float.self, forKey: .minimumFaceAreaFraction) ?? minimumFaceAreaFraction
        perGroupFrameCount = try values.decodeIfPresent(Int.self, forKey: .perGroupFrameCount) ?? perGroupFrameCount
        revealAllGroups = try values.decodeIfPresent(Bool.self, forKey: .revealAllGroups) ?? false
        exportFormat = try values.decodeIfPresent(ExportFormat.self, forKey: .exportFormat) ?? exportFormat
        exportJPEGQuality = try values.decodeIfPresent(Float.self, forKey: .exportJPEGQuality) ?? exportJPEGQuality
    }

    var grouping: GroupingSettings {
        get {
            GroupingSettings(
                similarityThreshold: similarityThreshold,
                maximumGroupSize: maximumGroupSize,
                faceWeight: faceWeight,
                minimumFaceAreaFraction: minimumFaceAreaFraction
            )
        }
        set {
            similarityThreshold = newValue.similarityThreshold
            maximumGroupSize = newValue.maximumGroupSize
            faceWeight = newValue.faceWeight
            minimumFaceAreaFraction = newValue.minimumFaceAreaFraction
        }
    }

    /// Clamps every field into a range that keeps a bad settings file from
    /// making a pass unbounded, inert, or unable to produce a usable image.
    func validated() -> WaifuSettings {
        var copy = self
        // A zero or negative decode width would make every Vision call fail, so
        // the floor is a real one rather than a defensive one.
        copy.taggerThreshold = copy.taggerThreshold.isFinite ? min(max(copy.taggerThreshold, 0.35), 1) : 0.35
        copy.tagTopCount = min(max(copy.tagTopCount, 1), 99)
        copy.analysisWidth = min(max(copy.analysisWidth, 64), 4096)
        copy.similarityThreshold = min(max(copy.similarityThreshold, 0.01), 2.0)
        copy.maximumGroupSize = min(max(copy.maximumGroupSize, 2), 500)
        copy.faceWeight = min(max(copy.faceWeight, 0), 1)
        copy.minimumFaceAreaFraction = min(max(copy.minimumFaceAreaFraction, 0), 0.5)
        copy.perGroupFrameCount = min(max(copy.perGroupFrameCount, 1), 33)
        copy.exportJPEGQuality = min(max(copy.exportJPEGQuality, 0.1), 1.0)
        return copy
    }
}
