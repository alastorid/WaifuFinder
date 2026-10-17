import AVFoundation
import Combine
import Foundation

/// Owns the preview's `AVPlayer`: the video it is pointed at, where the playhead
/// is, and whether it is running.
///
/// The player lives here rather than inside the SwiftUI view for one reason: the
/// rest of the window has to be able to move it. Clicking a grid cell and dragging
/// the timeline both mean "show me the video at this second", and neither is inside
/// the preview. Keeping the player in a small controller lets them say that without
/// either touching AVFoundation or reaching into another view's state.
///
/// `@MainActor` because every published value is read from a `View.body` and the
/// player is only ever touched from the main thread. The AVFoundation callbacks
/// arrive on other threads, so each one hops back here rather than assuming.
@MainActor
final class PreviewController: ObservableObject {
    /// The player, or `nil` when no video is loaded.
    private(set) var player: AVPlayer?

    /// Where the playhead is, in seconds, updated while playing.
    @Published private(set) var currentTime: Double = 0

    /// The asset's duration in seconds, or `0` until it is known.
    @Published private(set) var duration: Double = 0

    @Published private(set) var isPlaying = false
    @Published private(set) var isSeeking = false

    private var timelineOffset: Double = 0
    private var timelineDuration: Double?
    private var mediaDuration: Double?
    private var timeObserver: Any?
    private var rateObservation: NSKeyValueObservation?
    private var statusObservation: NSKeyValueObservation?
    private var endObserver: NSObjectProtocol?
    private var seekGeneration = UUID()

    /// Points the preview at `url`, replacing whatever was there, paused at the
    /// start.
    func load(url: URL) {
        unload()
        let player = AVPlayer(playerItem: AVPlayerItem(asset: VideoAssetFactory.asset(for: url)))
        // Reaching the end should stop, not rewind — the preview is for inspecting
        // a frame, and a video that loops back to t=0 hides it.
        player.actionAtItemEnd = .pause
        self.player = player
        observe(player)
    }

    func load(asset: AVAsset, offset: Double = 0, mediaDuration: Double? = nil, timelineDuration: Double? = nil) {
        unload()
        self.timelineOffset = offset
        self.timelineDuration = timelineDuration
        self.mediaDuration = mediaDuration
        currentTime = offset
        if let timelineDuration { duration = timelineDuration }
        let player = AVPlayer(playerItem: AVPlayerItem(asset: asset))
        player.actionAtItemEnd = .pause
        self.player = player
        observe(player)
    }

    func loadStill(time: Double, duration: Double) {
        unload()
        currentTime = time
        self.duration = duration
    }

    /// Releases the player and its observations.
    func unload() {
        if let timeObserver, let player { player.removeTimeObserver(timeObserver) }
        timeObserver = nil
        rateObservation = nil
        statusObservation = nil
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = nil

        player?.pause()
        player = nil
        currentTime = 0
        duration = 0
        isPlaying = false
        isSeeking = false
        seekGeneration = UUID()
        timelineOffset = 0; timelineDuration = nil; mediaDuration = nil
    }

    /// Moves the playhead to `seconds`.
    ///
    /// Land within one display frame in the player. The zoomable frame preview uses
    /// the exact native-resolution decoder when pixel-accurate inspection is needed.
    func seek(to seconds: Double) {
        guard let player else { return }
        let ceiling = mediaDuration.map { timelineOffset + $0 } ?? (duration > 0 ? duration : seconds)
        let clamped = max(timelineOffset, min(seconds, ceiling))
        let generation = UUID()
        seekGeneration = generation
        isSeeking = true
        player.seek(to: CMTime(seconds: max(0, clamped - timelineOffset), preferredTimescale: 600),
                    // The frame preview uses a separate exact full-resolution
                    // decoder. The player panel only needs to land within one
                    // display frame; a zero-tolerance seek can decode a long GOP
                    // synchronously enough to make frame selection feel stuck.
                    toleranceBefore: CMTime(value: 1, timescale: 30),
                    toleranceAfter: CMTime(value: 1, timescale: 30)) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.seekGeneration == generation else { return }
                self.isSeeking = false
            }
        }
        currentTime = clamped
    }

    /// Plays from wherever the playhead is (which a frame click just set).
    func play() {
        player?.play()
        isPlaying = true
    }

    func pause() {
        player?.pause()
        isPlaying = false
    }

    func togglePlayPause() {
        isPlaying ? pause() : play()
    }

    private func observe(_ player: AVPlayer) {
        // ~20 Hz: fast enough that the timeline thumb looks live, cheap enough to
        // run for the length of a film.
        let interval = CMTime(seconds: 0.05, preferredTimescale: 600)
        timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            let seconds = time.seconds
            Task { @MainActor in
                guard let self else { return }
                self.currentTime = (seconds.isFinite ? max(0, seconds) : 0) + self.timelineOffset
            }
        }

        rateObservation = player.observe(\.rate, options: [.initial, .new]) { [weak self] player, _ in
            let playing = player.rate != 0
            Task { @MainActor in self?.isPlaying = playing }
        }

        if let item = player.currentItem {
            statusObservation = item.observe(\.status, options: [.initial, .new]) { [weak self] item, _ in
                guard item.status == .readyToPlay else { return }
                let seconds = item.duration.seconds
                Task { @MainActor in
                    guard let self else { return }
                    self.duration = self.timelineDuration ?? (seconds.isFinite && seconds > 0 ? seconds : 0)
                }
            }

            endObserver = NotificationCenter.default.addObserver(
                forName: .AVPlayerItemDidPlayToEndTime,
                object: item,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor in self?.isPlaying = false }
            }
        }
    }
}
