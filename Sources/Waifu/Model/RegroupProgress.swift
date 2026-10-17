import Combine
import Foundation

/// Observed only by the small ring, so progress pulses do not invalidate the
/// frame grid, player or the rest of the panel.
@MainActor
final class RegroupProgress: ObservableObject {
    @Published private(set) var fraction: Double = 0

    func reset() { fraction = 0 }

    func update(processed: Int, total: Int) {
        fraction = max(fraction, min(1, Double(processed) / Double(max(1, total))))
    }
}
