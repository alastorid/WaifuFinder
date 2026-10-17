import Foundation

/// Why the Vision layer can refuse to run at all, and what it does instead.
///
/// Every request this layer uses — `CalculateImageAestheticsScoresRequest`,
/// `DetectFaceCaptureQualityRequest`, `GenerateImageFeaturePrintRequest` — was
/// introduced in macOS 15. The package's deployment target is macOS 15, so on
/// any OS that can run the binary the guard is always true; it exists so the
/// one configuration where it is not true (an old OS, or a future SDK that
/// moves the requests) fails with a message naming the requirement rather
/// than a crash or a silently empty analysis.
enum VisionError: Error, CustomStringConvertible {
    /// The aesthetics request is unavailable; macOS 15 or later is required.
    case unavailable

    /// The analysis ran but produced no observation. Kept distinct from
    /// `unavailable` so a Vision hiccup is recorded against the frame instead
    /// of failing the whole run.
    case noObservation(String)

    var description: String {
        switch self {
        case .unavailable:
            return "CalculateImageAestheticsScoresRequest is unavailable; macOS 15 or later is required"
        case .noObservation(let detail):
            return "Vision returned no observation: \(detail)"
        }
    }
}

/// The single availability check every Vision entry point makes before it
/// touches a request.
///
/// A function rather than a bare `#available` at each call site, so the
/// decision — and the error it throws — is written once. The deployment
/// target already requires macOS 15, so this is a guard against a
/// misconfigured target and against the future, not a branch that ever runs
/// in practice.
enum VisionAvailability {
    /// Throws `VisionError.unavailable` unless the macOS 15 Vision requests
    /// are present.
    static func require() throws {
        guard #available(macOS 15.0, *) else { throw VisionError.unavailable }
    }
}
