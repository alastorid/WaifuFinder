import AVFoundation
import Foundation

/// Each seek owns its decoder. Cancellation and timeout complete the waiter even
/// if AVFoundation never calls back, so abandoned reads cannot strand the preview.
final class PreviewFrameReader: @unchecked Sendable {
    private let generator: AVAssetImageGenerator
    private let lock = NSLock()
    private var continuation: CheckedContinuation<CGImage, Error>?
    private var result: Result<CGImage, Error>?
    private let queue = DispatchQueue(label: "Waifu.preview.decode")

    init(asset: AVAsset, maximumPixelSize: Int? = nil) {
        generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        // Zero means native dimensions. Analysis thumbnails must never constrain
        // the original frame used for pixel inspection and zooming.
        generator.maximumSize = maximumPixelSize.map { CGSize(width: $0, height: $0) } ?? .zero
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
    }

    func image(at seconds: Double, timeout: TimeInterval = 15) async throws -> CGImage {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                if let result {
                    lock.unlock()
                    continuation.resume(with: result)
                    return
                }
                self.continuation = continuation
                lock.unlock()
                queue.async { [self] in
                    lock.lock()
                    let finished = result != nil
                    lock.unlock()
                    guard !finished else { return }
                    generator.generateCGImageAsynchronously(for: CMTime(seconds: seconds, preferredTimescale: 60000)) { [self] image, _, error in
                        if let image { finish(.success(image)) }
                        else { finish(.failure(error ?? PreviewReadError.unreadable)) }
                    }
                }
                DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + timeout) { [weak self] in
                    self?.finish(.failure(PreviewReadError.timedOut))
                }
            }
        } onCancel: {
            self.finish(.failure(CancellationError()))
        }
    }

    private func finish(_ value: Result<CGImage, Error>) {
        lock.lock()
        guard result == nil else { lock.unlock(); return }
        result = value
        let waiter = continuation
        continuation = nil
        lock.unlock()
        waiter?.resume(with: value)
        queue.async { [self] in generator.cancelAllCGImageGeneration() }
    }
}

private enum PreviewReadError: LocalizedError {
    case unreadable, timedOut
    var errorDescription: String? {
        switch self {
        case .unreadable: "This frame could not be read. Try another frame."
        case .timedOut: "Reading this frame took too long. Try another frame."
        }
    }
}

/// Physical presses may supersede a read; autorepeat waits for completion.
struct PreviewArrowNavigation {
    private(set) var held: Int?
    private var repeating = false

    mutating func down(_ direction: Int, isRepeat: Bool, loading: Bool) -> Int? {
        if !isRepeat {
            held = direction
            repeating = false
            return direction
        }
        guard held == direction else { return nil }
        repeating = true
        return loading ? nil : direction
    }

    mutating func up(_ direction: Int) {
        if held == direction { reset() }
    }

    mutating func reset() { held = nil; repeating = false }
    func loaded() -> Int? { repeating ? held : nil }
}

/// Drive held navigation independently of macOS generating key-repeat events.
/// The view still gates each tick on its current frame read.
@MainActor
final class PreviewArrowRepeater {
    private var task: Task<Void, Never>?
    private let delay: Duration
    private let interval: Duration

    init(delay: Duration = .milliseconds(250), interval: Duration = .milliseconds(80)) {
        self.delay = delay
        self.interval = interval
    }

    func start(_ direction: Int, tick: @escaping @MainActor (Int) -> Void) {
        stop()
        let delay = delay, interval = interval
        task = Task { @MainActor in
            do {
                try await Task.sleep(for: delay)
                while !Task.isCancelled {
                    tick(direction)
                    try await Task.sleep(for: interval)
                }
            } catch { /* Release or focus loss cancels the repeat. */ }
        }
    }

    func stop() { task?.cancel(); task = nil }
}
