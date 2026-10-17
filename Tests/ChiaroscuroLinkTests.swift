import AppKit
import Foundation

// The east egg in the Sort and Filter fraction.
//
// A wave of light that runs from the "shown" fraction to the switched-off iCloud
// cloud in the title bar, once the pointer has rested on a fraction that is hiding
// frames for three seconds.
//
// None of that needs a window to pin down, because every part of it that can be
// wrong without one is a value: whether the fraction may link at all, whether the
// hold is long enough and continuous, and the shape of the light itself — where its
// head is at a given moment, that its wake only ever reaches backwards from that
// head and along the arc, that its crests fade with age, and that nothing at all is
// drawn before it leaves or after it has gone. What the cases below check is those
// contracts, so an east egg that turns into a loud streak across the window, or one
// that fires on a pointer merely crossing the panel, fails here rather than in the
// user's face.

func registerChiaroscuroLinkTests() {
    let suite = "chiaroscuro link"

    Registry.shared.add(suite: suite, TestCase(name: "only a fraction that is hiding frames may be linked", knownBug: nil) {
        check(Chiaroscuro.canLink(shown: 4, total: 5), "a filtered fraction links")
        check(Chiaroscuro.canLink(shown: 0, total: 5), "a fraction showing nothing still links")
        check(!Chiaroscuro.canLink(shown: 5, total: 5),
              "a fraction that hid nothing is a finished filter, not a link")
        check(!Chiaroscuro.canLink(shown: 0, total: 0), "an empty library has no fraction to link")
        check(!Chiaroscuro.canLink(shown: 7, total: 5),
              "a fraction larger than its library cannot be a real state")
    })

    Registry.shared.add(suite: suite, TestCase(name: "the light waits three seconds and fires once", knownBug: nil) {
        var arm = ChiaroscuroArm()
        check(arm.rested(for: 10) == nil, "a pointer that is not there has rested for nothing")
        arm.hoverBegan(at: 100)
        check(arm.isHovering, "the hold begins when the pointer arrives")
        checkClose(arm.rested(for: 101) ?? -1, 1, tolerance: 1e-9, "one second in, one second held")
        check(!arm.fire(at: 103 - 0.001), "2.999 seconds is not three seconds")
        check(arm.fire(at: 103), "exactly three seconds fires")
        check(arm.hasFired, "the hold is spent once it has fired")
        check(!arm.fire(at: 104), "a second fire for one hover is not a second wave")
    })

    Registry.shared.add(suite: suite, TestCase(name: "leaving before three seconds cancels the wave", knownBug: nil) {
        var arm = ChiaroscuroArm()
        arm.hoverBegan(at: 0)
        check(!arm.fire(at: 2.9), "still too short")
        arm.hoverEnded()
        check(!arm.isHovering, "leaving forgets the hold")
        // A pointer crossing the panel must not inherit the seconds it spent in
        // flight, so the timer after leaving cannot fire at all.
        check(!arm.fire(at: 3), "the seconds spent passing over are not the seconds spent resting")
        check(arm.rested(for: 9) == nil, "nothing is being held")

        // Coming back starts a fresh three seconds.
        arm.hoverBegan(at: 10)
        check(!arm.fire(at: 12), "the new hold starts from nothing")
        check(arm.fire(at: 13), "and fires on its own three seconds")
    })

    Registry.shared.add(suite: suite, TestCase(name: "re-entering without leaving does not restart the three seconds", knownBug: nil) {
        var arm = ChiaroscuroArm()
        arm.hoverBegan(at: 0)
        arm.hoverBegan(at: 2)
        check(!arm.fire(at: 2.5), "a second entry is not a second hold")
        check(arm.fire(at: 3), "the original hold is the one that counts")
    })

    Registry.shared.add(suite: suite, TestCase(name: "the wave arcs from the fraction up towards the cloud", knownBug: nil) {
        // Fraction at the bottom of the right-hand panel, cloud in the title bar.
        // Window coordinates: the origin is the window's bottom left, so the cloud
        // in the title bar is *above* the fraction and has the larger y.
        let source = CGPoint(x: 1180, y: 180)
        let destination = CGPoint(x: 1240, y: 862)
        let beam = ChiaroscuroBeam(source: source, destination: destination, travel: 0.5)
        checkClose(Double(beam.point(at: 0).x), Double(source.x), tolerance: 1e-9,
                   "the arc starts at the fraction")
        checkClose(Double(beam.point(at: 0).y), Double(source.y), tolerance: 1e-9,
                   "the arc starts at the fraction")
        checkClose(Double(beam.point(at: 1).x), Double(destination.x), tolerance: 1e-9,
                   "the arc ends on the cloud")
        checkClose(Double(beam.point(at: 1).y), Double(destination.y), tolerance: 1e-9,
                   "the arc ends on the cloud")
        // Bowed towards the top of the window: a straight line would read as a
        // progress bar rather than a wave of light.
        check(beam.control.y >= (source.y + destination.y) / 2 - 1e-9,
              "the arc bows towards the title bar")
        check(beam.control.y > (source.y + destination.y) / 2 + 1,
              "and it bows a real distance, not a rounding error")
        // Nothing may be asked for a position off the ends of the light's own path.
        check(beam.point(at: -4) == beam.point(at: 0), "before the start is the start")
        check(beam.point(at: 9) == beam.point(at: 1), "past the end is the end")
        // A short hop must not become a loop.
        let near = ChiaroscuroBeam(source: CGPoint(x: 100, y: 100), destination: CGPoint(x: 140, y: 108), travel: 0.5)
        check(near.control.y >= 100 - 1e-9, "a short hop still bows upwards")
    })

    Registry.shared.add(suite: suite, TestCase(name: "there is no wave without a switched-off cloud to fly to", knownBug: nil) {
        // The only case that waits: it is the only one that has to let three real
        // seconds pass for the link to decide something, which is the whole rule.
        try await Task { @MainActor in
            let fraction = CGRect(x: 1000, y: 200, width: 44, height: 14)
            let cloud = CGRect(x: 1220, y: 806, width: 30, height: 30)
            let link = ChiaroscuroLink()
            link.setSource(fraction)
            check(link.beam == nil, "with no cloud in the title bar there is nothing to draw")

            // A fraction that hides frames is still not enough on its own.
            link.hoverBegan()
            try? await Task.sleep(for: .milliseconds(3_300))
            check(link.beam == nil, "however long the pointer rests, no destination means no light")
            link.hoverEnded()

            // With the switched-off cloud there, the same three seconds do send it.
            link.setDestination(cloud)
            link.hoverBegan()
            try? await Task.sleep(for: .milliseconds(3_300))
            guard let beam = link.beam else {
                check(false, "three seconds over a fraction that hides frames, with the cloud switched off, sends the light")
                return
            }
            check(beam.isLit, "and what arrives is a light that is on the window")
            checkEqual(beam.source, CGPoint(x: fraction.midX, y: fraction.midY),
                       "the light leaves from the middle of the fraction")
            checkEqual(beam.destination, CGPoint(x: cloud.midX, y: cloud.midY),
                       "and lands on the middle of the cloud, in the title bar")

            // Switching iCloud downloads on takes the destination away, and with it the
            // light — mid-flight included, so it cannot outlive its cloud.
            link.setDestination(nil)
            check(link.beam == nil, "a cloud that is downloading is not a destination")
        }.value
    })

    Registry.shared.add(suite: suite, TestCase(name: "the head leaves the fraction, arrives at the cloud, and stops there", knownBug: nil) {
        let beam = ChiaroscuroBeam(source: CGPoint(x: 900, y: 300), destination: CGPoint(x: 1200, y: 870), travel: 0)
        checkClose(beam.head, 0, tolerance: 1e-9, "at rest the light is on the fraction")
        var previous = -1.0
        for step in 0...100 {
            let moving = ChiaroscuroBeam(source: beam.source, destination: beam.destination,
                                         travel: Double(step) / 100)
            check(moving.head >= previous, "the head never travels backwards")
            previous = moving.head
        }
        let arrived = ChiaroscuroBeam(source: beam.source, destination: beam.destination, travel: 1)
        checkClose(arrived.head, 1, tolerance: 1e-9, "the head ends on the cloud")
        // Past arrival the head stays put while the light fades out.
        let gone = ChiaroscuroBeam(source: beam.source, destination: beam.destination,
                                   travel: 1 + ChiaroscuroBeam.fadeOut / 2)
        checkClose(gone.head, 1, tolerance: 1e-9, "the light does not travel past the cloud")
    })

    Registry.shared.add(suite: suite, TestCase(name: "the wake only reaches backwards from the head, along the arc", knownBug: nil) {
        let source = CGPoint(x: 1000, y: 200)
        let destination = CGPoint(x: 1230, y: 822)
        for step in 0...40 {
            let beam = ChiaroscuroBeam(source: source, destination: destination, travel: Double(step) / 20)
            let range = beam.litRange
            check(range.upperBound <= beam.head + 1e-9, "the lit stretch ends at the head")
            check(range.lowerBound >= 0, "and never runs off the start of the arc")
            check(range.upperBound - range.lowerBound <= ChiaroscuroBeam.trail + 1e-9,
                  "the wake is as long as it says it is")
            for (index, crest) in beam.wake.enumerated() {
                // Every crest sits on the arc, at the age it claims — measured
                // against the arc rather than restated from the production figure.
                let expected = max(0, beam.head - ChiaroscuroBeam.trail * ChiaroscuroBeam.crestAges[index])
                let position = arcParameter(beam, crest.point)
                check(position != nil, "a crest is on the arc at all")
                guard let parameter = position else { continue }
                check(abs(parameter - expected) < 2 * samplingStep,
                      "crest \(index) is on the arc at its own age")
                check(parameter <= beam.head + samplingStep,
                      "a crest is behind the head, never ahead of it")
                check(beam.head - parameter <= ChiaroscuroBeam.trail + samplingStep,
                      "and inside the trail")
            }
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "the crests fade with age, and the wave swells without shouting", knownBug: nil) {
        let beam = ChiaroscuroBeam(source: CGPoint(x: 1000, y: 200), destination: CGPoint(x: 1230, y: 822),
                                   travel: 0.5)
        let wake = beam.wake
        checkEqual(wake.count, ChiaroscuroBeam.crestAges.count, "every crest of the wave is drawn")
        for earlier in 0..<(wake.count - 1) {
            let young = wake[earlier], old = wake[earlier + 1]
            check(young.radius > old.radius, "a crest is largest where the light is brightest")
            check(young.brightness > old.brightness, "and dimmer the further behind it is")
        }
        // Restrained: this is a candle in a dark room, not a glow over the window.
        for crest in wake {
            check(crest.brightness <= 0.5, "a single crest never exceeds half strength")
            check(crest.radius <= 15, "and stays small enough to read as light, not a shape")
        }
        // The swell breathes over the flight without ever leaving its envelope. Measured
        // between the ease-in and the arrival, because the light is *supposed* to be
        // dark at both ends of the flight — that is the presence curve's own job,
        // checked in the case below.
        var lowest = 1.0, highest = 0.0
        for step in 10...100 {
            let swell = ChiaroscuroBeam(source: beam.source, destination: beam.destination,
                                        travel: Double(step) / 100).wake.first?.brightness ?? 0
            lowest = min(lowest, swell)
            highest = max(highest, swell)
        }
        check(lowest > 0.1, "the wave keeps a floor of light while it is crossing")
        check(highest <= 0.5, "and never rises past its crest")
    })

    Registry.shared.add(suite: suite, TestCase(name: "nothing is drawn before the light leaves or after it is gone", knownBug: nil) {
        let source = CGPoint(x: 1000, y: 200)
        let destination = CGPoint(x: 1230, y: 822)
        for travel in [0.0, -0.5, 1 + ChiaroscuroBeam.fadeOut, 2.0] {
            let beam = ChiaroscuroBeam(source: source, destination: destination, travel: travel)
            check(!beam.isLit, "travel \(travel) is not on the window at all")
            checkClose(beam.presence, 0, tolerance: 1e-9, "no light at travel \(travel)")
            checkClose(beam.bloom, 0, tolerance: 1e-9, "no bloom at travel \(travel)")
            check(beam.wake.allSatisfy { $0.brightness <= 1e-9 },
                  "no crest is lit at travel \(travel)")
        }
        // Rises quickly but not instantly, and leaves slowly.
        let arriving = ChiaroscuroBeam(source: source, destination: destination, travel: 0.01)
        check(arriving.presence > 0 && arriving.presence < 0.2, "the light eases in")
        let leaving = ChiaroscuroBeam(source: source, destination: destination, travel: 1 + ChiaroscuroBeam.fadeOut * 0.5)
        check(leaving.presence > 0.2 && leaving.presence < 0.8, "and goes out over time, not in a frame")
    })

    Registry.shared.add(suite: suite, TestCase(name: "the cloud lights up when the light arrives, and not before", knownBug: nil) {
        let source = CGPoint(x: 1000, y: 200)
        let destination = CGPoint(x: 1230, y: 822)
        for travel in stride(from: 0.0, through: 0.6, by: 0.1) {
            check(ChiaroscuroBeam(source: source, destination: destination, travel: travel).bloom == 0,
                  "the cloud is untouched at travel \(travel)")
        }
        var previous = -1.0
        for step in 7...10 {
            let bloom = ChiaroscuroBeam(source: source, destination: destination,
                                       travel: Double(step) / 10).bloom
            check(bloom >= previous, "the bloom grows as the light closes on the cloud")
            previous = bloom
        }
        check(ChiaroscuroBeam(source: source, destination: destination, travel: 1).bloom > 0.5,
              "the light is on the cloud when it arrives")
        check(ChiaroscuroBeam(source: source, destination: destination,
                              travel: 1 + ChiaroscuroBeam.fadeOut).bloom == 0,
              "and gone once the flight is over")
    })
}

/// How finely the arc is sampled when a point is measured against it, as a fraction
/// of the arc's own length. Comparisons on the arc allow for one step of it: the
/// point under test is not snapped to a sample, it is measured between them.
private let samples = 400
private let samplingStep = 1.0 / Double(samples) * 2

/// Where on the arc `point` lies, found by sampling rather than by re-deriving what
/// production computed — so this measures where a crest actually landed instead of
/// restating where it was asked to land. `nil` when the point is nowhere near the arc.
private func arcParameter(_ beam: ChiaroscuroBeam, _ point: CGPoint) -> Double? {
    var best = (t: 0.0, distance: Double.greatestFiniteMagnitude)
    for step in 0...samples {
        let t = Double(step) / Double(samples)
        let onArc = beam.point(at: t)
        let distance = Double(hypot(onArc.x - point.x, onArc.y - point.y))
        if distance < best.distance { best = (t, distance) }
    }
    // Two sampling steps of slack: an arc several hundred points long is only being
    // measured this finely, so anything closer than this is on it.
    guard best.distance <= Double(beamLength(beam)) * samplingStep * 2 else { return nil }
    return best.t
}

/// Roughly how long the arc is, as the straight distance between its ends.
private func beamLength(_ beam: ChiaroscuroBeam) -> Double {
    Double(hypot(beam.destination.x - beam.source.x, beam.destination.y - beam.source.y))
}