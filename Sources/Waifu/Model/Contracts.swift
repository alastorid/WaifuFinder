import CoreGraphics
import Foundation

// The data model every layer of Waifu speaks.
//
// ## Why this file is hand-written and hand-pinned
//
// Waifu is built by five separate agents working in parallel, and the only way
// five people editing five directories end up with one program is if the types
// they exchange are fixed before any of them starts. So the model lives in one
// file, that file is the *only* place a field is defined, and the layers talk to
// each other through the protocols at the bottom of this file rather than
// through each other.
//
// The rule the rest of the codebase follows: a value is copied out of Apple's
// APIs *verbatim*, and every normalisation happens in a named, documented place
// with a reason. `AnalyzedFrame.artScore` is Apple's `overallScore` as returned,
// never rescaled, never blended at the point of capture.

// MARK: - Identity

/// Stable identifier for one analysed frame.
///
/// Derived from the frame's ordinal in the sampling pass (`"f0000123"`) rather
/// than generated, so the same frame has the same identifier across runs of the
/// same video and can be cited in a URL, an export filename, or a cache row.
///
/// Zero-padding to six digits is what keeps lexicographic order equal to numeric
/// order below a million frames. Past that the padding is decorative and the
/// ordering is still correct — just no longer sortable as text, which no code
/// path relies on.
struct FrameID: Sendable, Hashable, Codable, Equatable, CustomStringConvertible {
    let raw: String

    init(sampleIndex: Int) {
        self.raw = String(format: "f%06lld", Int64(max(0, sampleIndex)))
    }

    /// `nil` for anything that is not a well-formed frame identifier, rather
    /// than a fresh identifier that happens to look similar. An identifier that
    /// cannot be parsed must not be able to name a frame.
    init?(raw: String) {
        guard raw.count >= 7, raw.first == "f", raw.dropFirst().allSatisfy({ $0 >= "0" && $0 <= "9" }),
              let index = Int(raw.dropFirst()), index >= 0 else { return nil }
        self.raw = raw
    }

    var sampleIndex: Int { Int(raw.dropFirst()) ?? 0 }

    var description: String { raw }
}

// MARK: - Video

/// What Waifu needs to know about a video before it decides how to sample it.
///
/// Everything here comes from `AVAsset`, and `frameCount` is *estimated*
/// (`duration × frameRate`) rather than read from the container: `nb_frames` is
/// not reliably present in an MP4 track's metadata, and the estimate is only
/// ever used to bound work, never to address a frame. A frame is addressed by
/// its ordinal in the decode pass, which cannot drift.
struct VideoInfo: Sendable, Equatable {
    let url: URL
    /// Seconds. Never `0` for a real asset; a zero-duration video is refused
    /// before it becomes a division.
    let durationSeconds: Double
    let width: Int
    let height: Int
    let frameRate: Double
    /// Estimated decoded frames over the whole file. Used for progress and caps.
    let estimatedFrameCount: Int
    /// True when the file carries no video track at all.
    var hasVideo: Bool { width > 0 && height > 0 }

    /// Pixels per second of source material at the native resolution, which is
    /// what makes the decode width a real decision rather than a nicety: a
    /// 1080p30 file is 62 Mpx/s and a 4K60 file is 500 Mpx/s.
    var pixelsPerSecondAtSource: Double { Double(width * height) * frameRate }
}

// MARK: - One analysed frame

/// One sampled frame with everything Vision said about it.
///
/// **`artScore` and `faceScore` are different instruments measuring different
/// things.** `artScore` is Apple's `CalculateImageAestheticsScoresRequest`
/// `overallScore`, verbatim — measured over roughly −0.996…+1.000 on real
/// material, so it is not a probability and must not be read as one. `faceScore`
/// is Vision's `DetectFaceCaptureQualityRequest` capture quality, 0…1, which
/// Apple documents as describing whether a face is well lit, sharp and centrally
/// positioned, and as intended for comparing a face against other captures of the
/// *same* face.
///
/// They are stored apart and never summed at the point of capture. Ranking
/// happens later, on group-relative ranks, in `FrameRanker`.
struct AnalyzedFrame: Identifiable, Sendable, Equatable, Codable {
    let id: FrameID
    /// Presentation timestamp of this frame in the source, in seconds.
    let timeSeconds: Double
    /// Apple's aesthetics `overallScore`, verbatim. Never rescaled.
    let artScore: Float
    /// Capture quality of the largest significant face in this frame.
    ///
    /// `nil` means **no significant face was detected**, which is a real answer
    /// and not a failure — and is why it is an optional rather than a `0`. A
    /// frame of a landscape must not be able to look like a frame with a
    /// badly-captured face.
    let faceScore: Float?
    /// Native-crop detail estimate, measured during initial scoring.
    let faceDefinitionScore: Float?
    /// How many faces Vision found, before the area filter. Kept so "one face
    /// in the background" is distinguishable from "a crowd".
    let faceCount: Int
    /// Area of the largest detected face as a fraction of the frame. Used to
    /// tell a subject from an incidental background face, never to score.
    let faceAreaFraction: Float?
    /// Compressed FeaturePrint payload, or `nil` when FeaturePrint failed.
    ///
    /// A missing vector is a degradation, not a failure: the frame still scores,
    /// browses, exports and ranks on aesthetics and face quality — it simply
    /// cannot join a group. `FrameGrouper` treats `nil` as ungroupable rather
    /// than as a frame that matches nothing.
    let featurePrint: Data?
    /// The group this frame was placed in, or `nil` when it is not in any group.
    var groupID: FrameGroupID?
    /// Key for the cached analysis-resolution thumbnail held in memory.
    ///
    /// A key, not a path. Thumbnail bytes only live for the current in-memory
    /// analysis session, so this identifier never depends on a filesystem location.
    let thumbnailKey: String
    /// Cropped analysis image width / height, available before its JPEG loads.
    let thumbnailAspectRatio: Double?

    /// Sparse UInt8 model heads compressed with LZ4; absent in older sessions.
    /// Values below the fixed storage threshold reconstruct as zero.
    var tagProbabilities: Data?
    /// Vocabulary index layout that produced `tagProbabilities`.
    var tagVocabularyFingerprint: UInt64?
    var tagRating: TagRating?
    var topTags: [String]?
    /// Sum of the highest 15 general probabilities above the saved threshold.
    var tagGeneralScore: Float?

    /// Official TruFor detection-head sigmoid, never a mask-area heuristic.
    /// Missing means not analyzed. Provenance prevents reuse across changed pixels/models.
    var truforScore: Float?
    var truforFingerprint: String?
    var truforContentIdentity: String?
    var validTruForScore: Float? {
        truforScore.flatMap { $0.isFinite && (0...1).contains($0) ? $0 : nil }
    }

    /// Whether this frame already has usable tagger output, including partial
    /// results restored from an older analysis snapshot.
    var hasTaggingResult: Bool {
        tagProbabilities != nil || tagRating != nil || !(topTags?.isEmpty ?? true) || tagGeneralScore != nil
    }

    init(sampleIndex: Int,
         timeSeconds: Double,
         artScore: Float,
         faceScore: Float?,
         faceCount: Int,
         faceAreaFraction: Float?,
         featurePrint: Data?,
         thumbnailKey: String? = nil,
         groupID: FrameGroupID? = nil,
         faceDefinitionScore: Float? = nil,
         thumbnailAspectRatio: Double? = nil,
         tagProbabilities: Data? = nil,
         tagVocabularyFingerprint: UInt64? = nil,
         tagRating: TagRating? = nil,
         topTags: [String]? = nil,
         tagGeneralScore: Float? = nil,
         truforScore: Float? = nil, truforFingerprint: String? = nil, truforContentIdentity: String? = nil) {
        self.id = FrameID(sampleIndex: sampleIndex)
        self.timeSeconds = timeSeconds
        self.artScore = artScore
        self.faceScore = faceScore
        self.faceDefinitionScore = faceDefinitionScore
        self.faceCount = faceCount
        self.faceAreaFraction = faceAreaFraction
        self.featurePrint = featurePrint
        self.thumbnailKey = thumbnailKey ?? "thumb-\(FrameID(sampleIndex: sampleIndex).raw)"
        self.thumbnailAspectRatio = thumbnailAspectRatio
        self.groupID = groupID
        self.tagProbabilities = tagProbabilities
        self.tagVocabularyFingerprint = tagVocabularyFingerprint
        self.tagRating = tagRating
        self.topTags = topTags
        self.tagGeneralScore = tagGeneralScore
        self.truforScore=truforScore;self.truforFingerprint=truforFingerprint;self.truforContentIdentity=truforContentIdentity
    }

    /// A significant subject face, or a small face measured by definition mode.
    var hasSignificantFace: Bool { faceScore != nil || faceDefinitionScore != nil }
}

// MARK: - Groups

/// Stable identifier for a group of alike frames.
///
/// Derived from the lexicographically smallest member's identifier rather than
/// generated, so a group keeps its identity across rebuilds and can be cited in
/// a URL or an export filename.
struct FrameGroupID: Sendable, Hashable, Codable, Equatable, CustomStringConvertible {
    let raw: String

    init(memberIDs: [String]) {
        self.raw = memberIDs.min() ?? ""
    }

    init(raw: String) { self.raw = raw }

    var description: String { raw }
}

/// A group of frames believed to show approximately the same composition.
///
/// Built by clustering FeaturePrint distances, then ranked internally by
/// `FrameRanker`. The grouping question ("do these belong together?") and the
/// ranking question ("which is better?") are answered by different code and
/// never influence each other.
struct FrameGroup: Identifiable, Sendable, Equatable, Codable {
    let id: FrameGroupID
    /// Member identifiers, ascending — deterministic regardless of the order
    /// the clustering visited them in.
    let memberIDs: [String]

    init(id: FrameGroupID, memberIDs: [String]) {
        self.id = id
        self.memberIDs = memberIDs.sorted()
    }

    /// Human-readable label, when one can be derived. `nil` when it cannot:
    /// an unlabelled group is not a broken one.
    var label: String? { nil }
}

// MARK: - Ranking

/// One frame as the ranker is allowed to see it.
///
/// **Ranks, not raw scores.** `artScore` is measured over −0.996…+1.000 and
/// face capture quality over 0…1. They are different instruments on different
/// scales, and a formula like `artScore * 0.6 + faceScore * 0.4` silently
/// asserts a relationship between their units that does not exist. Each signal
/// is first reduced to its **rank within the set being ranked** — 1 for the
/// best, 0 for the worst — and the ranks are blended. Rank is unit-free, so the
/// blend means what it says.
struct RankedFrame: Sendable, Equatable {
    let id: FrameID
    /// Group-relative art rank: 1.0 best, 0.0 worst.
    let artRank: Double
    /// Group-relative face rank, or `nil` when this set contains no frame with a
    /// significant face — so a landscape scene is not dragged around by a
    /// portrait's absence.
    let faceRank: Double?
    /// The blended value, in the same 1.0-best/0.0-worst space.
    let combined: Double
}

// MARK: - Progress

/// Where an analysis pass currently is.
///
/// A closed enum rather than a string, so an unrecognised state cannot be
/// reported as a *successful* one. `phase` is the thing a UI switches on;
/// `message` is never parsed by anything.
struct AnalysisProgress: Sendable, Equatable {
    enum Phase: String, Sendable, Equatable {
        case idle
        case probing       // reading duration / dimensions / fps
        case sampling      // decoding frames and writing thumbnails
        case scoring       // running Vision over decoded frames
        case grouping      // clustering feature prints
        case complete
        case failed
        case cancelled
    }

    var phase: Phase = .idle
    var processed: Int = 0
    var total: Int = 0
    /// `true` when the pass cannot know its own frame total, so
    /// `total` carries no meaning and `fraction` must not be drawn
    /// as a ratio. The UI shows a looping bar instead.
    ///
    /// Only the keyframe pass sets this. Its kept frames are the
    /// encoder's sync samples, and how many of those a file holds is
    /// a property of the encode — unknowable until the file has been
    /// read. Full and Quick derive their totals exactly from the
    /// probe, so a ratio is honest for them and this stays `false`.
    var isIndeterminate: Bool = false
    /// Seconds of wall clock spent in the current pass, for an ETA readout.
    var elapsed: Double = 0
    var message: String = ""
    var facesProcessed: Int = 0
    var framesWithFaces: Int = 0
    var groupedFrames: Int = 0
    var groupsIdentified: Int = 0

    var isRunning: Bool { [.probing, .sampling, .scoring, .grouping].contains(phase) }

    /// 0…1, or 0 when the total is unknown. Reported rather than derived at
    /// every call site so one division cannot be done four different ways.
    /// Always `0` while `isIndeterminate`, because a fraction of a
    /// count the pass never had is worse than no bar at all.
    var fraction: Double {
        guard total > 0, processed >= 0, !isIndeterminate else { return 0 }
        return min(1.0, Double(processed) / Double(total))
    }

    /// Frames per second the pass is sustaining, measured as `processed`
    /// over `elapsed`. Zero until the first frame has landed.
    var framesPerSecond: Double {
        guard elapsed > 0, processed > 0 else { return 0 }
        return Double(processed) / elapsed
    }

    /// Seconds the pass is expected to take yet, at the current speed.
    /// `nil` while the speed is unknown, once the pass has reached its
    /// total, or while it is indeterminate — dividing a duration by a
    /// frame rate it does not have would only produce a confident
    /// wrong number.
    var estimatedRemaining: Double? {
        let speed = framesPerSecond
        guard !isIndeterminate, speed > 0, total > processed else { return nil }
        return Double(total - processed) / speed
    }

    var isTerminal: Bool {
        phase == .complete || phase == .failed || phase == .cancelled
    }
}

// MARK: - Analysis mode

/// How much of a video one analysis pass covers.
///
/// **Full** — the default — decodes and scores *every* frame the
/// file holds. **Quick** scores one frame a second. The decode is
/// the same front-to-back pass in all three modes (a sequential
/// reader cannot skip frames cheaply), so the narrower modes buy
/// back the Vision work, not the decode. That is why the choice is
/// made before a video loads, on the drop zone, rather than as a
/// knob on a finished analysis.
///
/// **Keyframe** scores the encoder's sync samples — the points it
/// can decode from on its own. Unlike Quick's fixed one-per-second
/// grid, this follows the encode: keyframes cluster after a scene
/// cut and thin out through a static shot, so its coverage is
/// content-driven rather than even. Its purpose is throughput
/// rather than coverage. Quick on a long file sits far below the
/// Neural Engine's ceiling, and keyframes are the subset of frames
/// that actually changes, so the scoring pass runs at a rate the
/// hardware actually sustains.
enum AnalysisMode: String, CaseIterable, Identifiable, Sendable, Codable {
    case quick
    case keyframe
    case full

    var id: String { rawValue }

    var label: String {
        switch self {
        case .quick: return "Quick"
        case .keyframe: return "Keyframe"
        case .full: return "Full"
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        if value == "faceDefinition" { self = .full }
        else if let mode = Self(rawValue: value) { self = mode }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unknown analysis mode") }
    }

    /// Frames per second a quick pass analyses.
    ///
    /// The rate the app shipped with for years: fine enough to catch
    /// a gesture, a glance or a lighting change, coarse enough that
    /// the 47-minute reference is ~2,800 scored frames instead of
    /// 84,067.
    static let quickRate: Double = 1.0

    /// Seconds between keyframes assumed when sizing the progress
    /// bar, for the keyframe pass only.
    ///
    /// A real GOP is a property of the encode — 1s on a screen
    /// recording, 10s across a long static take — and is not knowable
    /// before the pass has read the file. Two seconds is the common
    /// encoder default. The figure is never shown to anyone: a
    /// keyframe pass reports itself as `isIndeterminate` and loops
    /// instead. This only keeps `total` a sane non-zero number for
    /// anything that reads it without checking the flag.
    static let nominalKeyframeInterval: Double = 2.0

    /// How many decoded frames sit between the frames a pass
    /// analyses, for a source at `frameRate`. `1` — every frame —
    /// is the full pass.
    ///
    /// The step is a grid over *decoded ordinals*, not a timestamp
    /// match, so the same video yields the same kept frames on every
    /// run; at 29.97 fps a quick pass analyses frames 0, 30, 60…,
    /// which land within a frame of each whole second.
    ///
    /// Keyframe mode is deliberately `1` as well: it keeps no ordinal
    /// grid at all. The sampler filters on the reader's sync-sample
    /// flag instead, which cannot be expressed as a fixed step
    /// because it does not repeat.
    func frameStep(frameRate: Double) -> Int {
        guard self == .quick, frameRate > 0 else { return 1 }
        return max(1, Int((frameRate / Self.quickRate).rounded()))
    }

    /// How many frames a pass analyses for a video whose `frameCount`
    /// decoded frames run at `frameRate`: every frame, or one a
    /// second. An estimate against the probe's own estimate of
    /// `frameCount`, used to size the progress bar.
    ///
    /// `durationSeconds` is only consulted by the keyframe pass,
    /// whose kept-frame count is not derivable from a frame rate at
    /// all. It defaults to `0` so the frame-rate modes keep their
    /// existing signature.
    func analysedFrameCount(frameCount: Int, frameRate: Double, durationSeconds: Double = 0) -> Int {
        if self == .keyframe {
            guard durationSeconds.isFinite, durationSeconds > 0 else { return 1 }
            return max(1, Int((durationSeconds / Self.nominalKeyframeInterval).rounded()))
        }
        let step = frameStep(frameRate: frameRate)
        return max(1, (frameCount + step - 1) / step)
    }
}

// MARK: - Export

/// Format for exported stills.
enum ExportFormat: String, Sendable, Equatable, Codable, CaseIterable {
    case png
    case jpeg

    var fileExtension: String {
        switch self {
        case .png: return "png"
        case .jpeg: return "jpg"
        }
    }

    /// Uniform type identifier, for `NSAttributedString`-free save panels and
    /// for `CGImageDestination`.
    var contentType: String {
        switch self {
        case .png: return "public.png"
        case .jpeg: return "public.jpeg"
        }
    }
}

/// What one export run is asked to produce.
///
/// **Naming is derived from the source, never asked for.** A user exporting
/// 40 frames does not want to type 40 filenames; they want files that sort into
/// the order they were selected in and say where they came from. The pattern is
/// therefore fixed here and the counter is zero-padded so a directory listing
/// reads in order.
struct ExportRequest: Sendable, Equatable {
    let frameIDs: [FrameID]
    let destinationDirectory: URL
    var format: ExportFormat = .png
    /// 0…1, JPEG only. PNG ignores it.
    var jpegQuality: Float = 0.92
    /// Write the analysis-resolution thumbnail instead of re-extracting at
    /// native resolution. Far cheaper and always available; loses detail.
    var useThumbnails: Bool = false
    /// Overwrite rather than skip-and-suffix when a file already exists.
    var overwrite: Bool = false
    /// Actual decoded timestamps, required for exact VFR exports from the browser.
    var frameTimes: [FrameID: Double] = [:]

    /// `"<stem>-000123.png"` for frame `f000123` of `sample-video.mp4`.
    ///
    /// The stem is the source file's name without its extension, so exports
    /// from two different videos never collide by name in one directory.
    func fileName(for frame: FrameID, sourceStem: String) -> String {
        let padded = String(format: "%06lld", Int64(frame.sampleIndex))
        return "\(sourceStem)-\(padded).\(format.fileExtension)"
    }
}

/// One exported file, for the results panel.
struct ExportResult: Sendable, Equatable {
    let frameID: FrameID
    let url: URL
    let byteCount: Int
}

// MARK: - Layer boundaries
//
// The five layers of Waifu, in the only order they may depend on each other:
//
//     Video ──► Vision ──► Grouping ──► Engine ──► UI
//                            │                        │
//                            └──────── Export ◄────────┘
//
// `Model` is depended on by all of them and depends on none. No layer imports
// another layer's concrete type; they meet at these protocols. That is what let
// them be built concurrently, and it is what keeps them independently testable.

/// One decoded frame, handed from the sampling layer to the scoring layer.
///
/// `image` is the *analysis-resolution* decode — the width given by
/// `WaifuSettings.analysisWidth`, not the source width. Vision needs roughly
/// 512 px to do its best work and the source is 1920 px wide; decoding at
/// native resolution and downscaling afterwards would spend four times the
/// memory bandwidth to arrive at the same pixels.
struct SampledFrame: Sendable {
    let sampleIndex: Int
    let timeSeconds: Double
    let image: CGImage
    /// Native visible-picture pixels for the shared face scoring request.
    var nativeImage: CGImage? = nil
}

/// Reads a video and hands over frames to analyse.
protocol FrameSampling: Sendable {
    /// Reads metadata without decoding any frame.
    func probe(_ url: URL) async throws -> VideoInfo

    /// Streams the frames the mode selects, in ascending timestamp
    /// order. `full` is frame by frame — every decoded frame, none
    /// skipped; `quick` keeps one frame a second; `keyframe` keeps
    /// the encoder's sync samples. All three are one sequential
    /// pass.
    ///
    /// - Parameters:
    ///   - mode: Which frames the pass analyses.
    ///   - onProgress: Called with the count of frames *written*, so a
    ///     caller can drive a progress bar without the sampler having
    ///     to know what a progress bar is.
    func sample(_ url: URL,
                analysisWidth: Int,
                mode: AnalysisMode,
                onProgress: @escaping @Sendable (Int) -> Void) -> AsyncThrowingStream<SampledFrame, Error>
}

/// Runs Apple's Vision models over one decoded frame.
protocol FrameScoring: Sendable {
    /// Scores one frame: aesthetics, face capture quality, and a FeaturePrint
    /// for grouping.
    ///
    /// The three models run concurrently because they are independent passes over
    /// pixels that are already decoded, and measured together they cost
    /// substantially less than three sequential passes.
    ///
    /// A FeaturePrint failure is *not* a failure of this call: the aesthetics
    /// score is the primary observation and a frame without a vector is still
    /// perfectly browsable. Only a failure of the aesthetics pass propagates,
    /// so a single unreadable frame does not abort the run.
    func score(_ frame: SampledFrame) async throws -> FrameScoreResult
}

/// What `FrameScoring` returns for one frame.
struct FrameScoreResult: Sendable, Equatable {
    let artScore: Float
    let faceScore: Float?
    let faceCount: Int
    let faceAreaFraction: Float?
    let featurePrint: Data?
    var faceDefinitionScore: Float? = nil
}

/// Decides which frames belong together, and how they rank against each other.
protocol FrameGrouping: Sendable {
    /// Clusters frames into groups of two or more.
    ///
    /// A frame whose FeaturePrint is missing is ungroupable and is never placed
    /// in a group. A lone frame is not a group and is never returned: there is
    /// nothing for a group browser to show.
    func group(_ frames: [AnalyzedFrame],
               settings: GroupingSettings) -> [FrameGroup]

    /// Ranks a set of frames best-first, on group-relative ranks.
    func rank(_ frames: [AnalyzedFrame], settings: GroupingSettings) -> [RankedFrame]
}

/// The knobs that decide what counts as "alike" and how a group is ranked.
struct GroupingSettings: Sendable, Equatable, Codable {
    /// Maximum FeaturePrint distance for two frames to be the same composition.
    ///
    /// Single linkage is used, so chains merge: A joins B, B joins C, and A and
    /// C end up together even if they are further apart than the threshold.
    /// That is correct for a slow pan, where consecutive frames are close and
    /// the ends of the shot are not — but it is also what lets one dense scene
    /// collapse into a single unusable group, which `maximumGroupSize` bounds.
    var similarityThreshold: Float

    /// Largest group accepted. Oversized groups are split into contiguous runs
    /// rather than truncated, so each group stays a run of neighbouring frames.
    var maximumGroupSize: Int

    /// Weight given to face capture quality when ranking. 0 is pure art, 1 is
    /// pure face. Applied to ranks, never to raw scores.
    var faceWeight: Double

    /// A face counts only if it covers at least this fraction of the frame.
    var minimumFaceAreaFraction: Float

    static let `default` = GroupingSettings(
        similarityThreshold: 0.35,
        maximumGroupSize: 60,
        faceWeight: 0.5,
        minimumFaceAreaFraction: 0.01
    )

    /// Clamps every field into a range that keeps a bad settings file from
    /// making grouping unbounded or inert.
    func validated() -> GroupingSettings {
        var copy = self
        copy.similarityThreshold = min(max(similarityThreshold, 0.01), 2.0)
        copy.maximumGroupSize = min(max(maximumGroupSize, 2), 500)
        copy.faceWeight = min(max(faceWeight, 0), 1)
        copy.minimumFaceAreaFraction = min(max(minimumFaceAreaFraction, 0), 0.5)
        return copy
    }
}

/// Writes frames out of the app.
protocol FrameExporting: Sendable {
    /// Re-extracts the given frames at native resolution and writes them to disk.
    ///
    /// - Returns: one result per frame, in the order requested. A frame that
    ///   failed is reported as an error in the stream rather than silently
    ///   dropped, so the user learns which of their selections did not survive.
    func export(_ request: ExportRequest, sourceURL: URL) -> AsyncThrowingStream<ExportResult, Error>
}
