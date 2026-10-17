import SwiftUI

/// The indeterminate bar for a pass that cannot know its own frame
/// total — the keyframe pass, whose kept frames are the encoder's
/// sync samples and are not countable in advance.
///
/// The XP boot bar: a run of segments lights up, sweeps left to
/// right, and the whole run wipes and starts over. It never claims to
/// be a percentage, because there is no percentage to claim.
///
/// Driven by `TimelineView(.animation)` rather than a `@State` +
/// `withAnimation` toggle. The lit position is computed per segment
/// inside the body, so SwiftUI has no animatable value to interpolate
/// — an animated state would cross-fade between two static renders of
/// the sweep instead of moving it. A timeline redraws per frame, which
/// is what a marquee needs.
private struct LoopingAnalysisBar: View {
    private static let segmentCount = 20
    /// How many segments are lit at once, as the head travels.
    private static let litRun: Double = 6
    /// Seconds for one full left-to-right pass, wipe included.
    private static let cycle: Double = 1.8

    /// How lit a segment is, given where the sweeping head has reached.
    ///
    /// Lit segments trail the head and fade behind it. When the head
    /// wraps past the right edge every distance goes negative at once
    /// and the bar is dark for an instant — that wipe is the beat that
    /// makes the loop read as a loop rather than a flicker.
    private func opacity(for index: Int, head: Double) -> Double {
        let distance = head - Double(index)
        guard distance >= 0, distance <= Self.litRun else { return 0.08 }
        return 1 - 0.72 * (distance / Self.litRun)
    }

    var body: some View {
        TimelineView(.animation) { context in
            // Reference date rather than a stored start, so a redraw
            // mid-run continues the sweep instead of restarting it.
            let now = context.date.timeIntervalSinceReferenceDate
            let head = (now.truncatingRemainder(dividingBy: Self.cycle) / Self.cycle)
                * Double(Self.segmentCount)

            GeometryReader { proxy in
                let gap: CGFloat = 2
                let width = (proxy.size.width - gap * CGFloat(Self.segmentCount - 1)) / CGFloat(Self.segmentCount)
                HStack(spacing: gap) {
                    ForEach(0..<Self.segmentCount, id: \.self) { index in
                        Rectangle()
                            .fill(Color.accentColor.opacity(opacity(for: index, head: head)))
                            .frame(width: max(1, width))
                    }
                }
                .frame(width: proxy.size.width, alignment: .leading)
            }
        }
        .frame(height: 6)
        .clipShape(RoundedRectangle(cornerRadius: 3))
        .accessibilityElement()
        .accessibilityLabel("Analysing keyframes")
    }
}

/// The grid and video player are unmounted during analysis. Four small progress
/// updates per second render scalar values, without sorting or decoding thumbnails.
struct AnalysisProgressView: View {
    let progress: AnalysisProgress
    let video: VideoInfo?
    let cancel: () -> Void

    /// Reduce Motion replaces the sweep with a plain spinner: the bar's
    /// whole meaning is motion, so there is nothing left to keep.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "viewfinder")
                .font(.system(size: 38, weight: .light))
                .foregroundStyle(.tint)
            VStack(spacing: 6) {
                Text(progress.message).font(.title3.weight(.medium))
                if let video {
                    Text(PhotosMedia.shared.album(video.url)?.title ?? video.url.lastPathComponent).foregroundStyle(.secondary).lineLimit(1)
                    Text("\(video.width) × \(video.height) · \(video.frameRate, specifier: "%.2f") fps · \(timecode(video.durationSeconds))")
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            if progress.phase == .grouping || progress.phase == .probing {
                ProgressView().controlSize(.small)
            } else if progress.isIndeterminate {
                if reduceMotion {
                    ProgressView().controlSize(.small)
                } else {
                    LoopingAnalysisBar().frame(maxWidth: 440)
                }
            } else {
                ProgressView(value: progress.fraction).frame(maxWidth: 440)
            }
            HStack(spacing: 32) {
                stat("Frames", progress.processed.formatted())
                stat("Faces", progress.facesProcessed.formatted())
                if progress.phase == .grouping || progress.phase == .scoring {
                    if progress.phase == .scoring {
                        stat("Frames with faces", progress.framesWithFaces.formatted())
                    }
                    stat("Grouped frames", progress.groupedFrames.formatted())
                    stat("Linked groups", progress.groupsIdentified.formatted())
                } else {
                    stat("Frames with faces", progress.framesWithFaces.formatted())
                }
            }
            HStack(spacing: 12) {
                Text(timecode(progress.elapsed))
                if progress.phase == .scoring {
                    Text("\(progress.framesPerSecond, specifier: "%.1f") frames/s")
                    // "of ~2,800 frames" is only true for the two modes that
                    // can count their frames. A keyframe pass never knows,
                    // so it reports what it is instead of a total it guessed.
                    if progress.isIndeterminate {
                        Text("keyframes only")
                    } else {
                        Text("of ~\(progress.total.formatted()) frames")
                    }
                }
            }.font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            Text("Results will appear when processing finishes.")
                .font(.caption).foregroundStyle(.tertiary)
            Button("Cancel", action: cancel)
        }
        .padding(36)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(spacing: 5) {
            Text(value).font(.title2.monospacedDigit().weight(.medium))
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
    }
}
