import Foundation
import Vision

/// An approximate shortest open path through the shown photos, using the same
/// Vision distance as grouping. No time window applies to this global order.
enum MinimalVisualDistanceOrder {
    static func sorted(_ frames: [AnalyzedFrame], reversed: Bool = false) throws -> [AnalyzedFrame] {
        var observations: [FeaturePrintObservation] = []
        var eligible: [AnalyzedFrame] = []
        var missing: [AnalyzedFrame] = []
        for frame in frames {
            try Task.checkCancellation()
            if let payload = frame.featurePrint, let observation = FeaturePrintCodec.decode(payload) {
                observations.append(observation)
                eligible.append(frame)
            } else {
                missing.append(frame)
            }
        }
        let indices: [Int]
        if eligible.count > 1, eligible.count <= CuOptMetalTSP.vectorLimit,
           let vectors = VisualFeatureVectors(observations: observations) {
            let seed = try path(count: eligible.count) { Double(vectors.distance($0, $1)) }
            let seeds = CuOptTSP.seeds(from: seed)
            let proposed: [Int]
            do {
                // Optimized CPU/Metal measurements on Apple M3 put the crossover
                // between 256 and 1,024 photos; avoid GPU dispatches for small sets.
                if eligible.count >= 512, let metal = CuOptMetalTSP.shared {
                    proposed = try metal.solve(vectors, seeds: seeds)
                } else if eligible.count < 512 {
                    proposed = try CuOptTSP.solveCPU(vectors.matrix(), seeds: seeds)
                } else {
                    proposed = seed
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // GPU allocation/compilation failures preserve a valid CPU route.
                proposed = seed
            }
            // Check the full proposed objective with Vision itself. Float kernels
            // and metric sampling cannot cause a worse native-distance sequence.
            var best = seed
            var bestCost = try nativeCost(seed, observations: observations)
            for route in [Array(0..<eligible.count), proposed] {
                let cost = try nativeCost(route, observations: observations)
                if cost < bestCost { best = route; bestCost = cost }
            }
            if best[0] > best[best.count - 1] { best.reverse() }
            indices = best
        } else {
            indices = try path(count: eligible.count) { left, right in
                try? observations[left].distance(to: observations[right])
            }
        }
        // Missing or corrupt prints remain visible in their existing score order.
        let ordered = indices.map { eligible[$0] } + missing
        return reversed ? Array(ordered.reversed()) : ordered
    }

    private static func nativeCost(_ route: [Int], observations: [FeaturePrintObservation]) throws -> Double {
        var cost = 0.0
        for (left, right) in zip(route, route.dropFirst()) {
            try Task.checkCancellation()
            guard let distance = try? observations[left].distance(to: observations[right]), distance.isFinite else {
                return .infinity
            }
            cost += distance
        }
        return cost
    }

    /// A metric seam lets route quality and scaling be checked without Vision.
    /// Small sets use nearest-neighbour construction and bounded 2-opt refinement.
    /// Large sets first divide around distant pivots, keeping both the number of
    /// distance calls and the cache bounded rather than building an n² matrix.
    static func path(count: Int, distance: (Int, Int) -> Double?) throws -> [Int] {
        try Task.checkCancellation()
        guard count > 1 else { return Array(0..<count) }
        var cache: [UInt64: Double] = [:]
        let cacheLimit = 131_072
        var calls = 0

        func measure(_ left: Int, _ right: Int) throws -> Double {
            if left == right { return 0 }
            calls += 1
            if calls % 128 == 0 { try Task.checkCancellation() }
            let key = UInt64(min(left, right)) * UInt64(count) + UInt64(max(left, right))
            if let value = cache[key] { return value }
            let measured = distance(left, right)
            // Incompatible observations must not discard frames or introduce NaN
            // into the route comparisons. Treat that edge as very expensive.
            let value = measured.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil } ?? 1e12
            // Refresh the bounded cache when full so later partitions still get
            // reuse during local refinement instead of repeatedly calling Vision.
            if cache.count == cacheLimit { cache.removeAll(keepingCapacity: true) }
            cache[key] = value
            return value
        }

        func cost(_ route: [Int]) throws -> Double {
            var total = 0.0
            for index in 1..<route.count { total += try measure(route[index - 1], route[index]) }
            return total
        }

        func farthest(from pivot: Int, in indices: [Int]) throws -> Int {
            var best = pivot
            var greatest = -1.0
            for index in indices {
                let value = try measure(pivot, index)
                if value > greatest { greatest = value; best = index }
            }
            return best
        }

        func refine(_ input: [Int], passes: Int, window: Int) throws -> [Int] {
            guard input.count > 2 else { return input }
            var route = input
            for _ in 0..<passes {
                var changed = false
                for start in 0..<(route.count - 1) {
                    try Task.checkCancellation()
                    for end in (start + 1)..<min(route.count, start + window + 1) {
                        if start == 0 && end == route.count - 1 { continue }
                        var before = 0.0
                        var after = 0.0
                        if start > 0 {
                            before += try measure(route[start - 1], route[start])
                            after += try measure(route[start - 1], route[end])
                        }
                        if end + 1 < route.count {
                            before += try measure(route[end], route[end + 1])
                            after += try measure(route[start], route[end + 1])
                        }
                        if after + 1e-9 < before {
                            route[start...end].reverse()
                            changed = true
                        }
                    }
                }
                if !changed { break }
            }
            return route
        }

        func build(_ indices: [Int]) throws -> [Int] {
            try Task.checkCancellation()
            guard indices.count > 1 else { return indices }
            let pivot = try farthest(from: indices[0], in: indices)
            if indices.count <= 64 {
                var remaining = indices.filter { $0 != pivot }
                var route = [pivot]
                while !remaining.isEmpty {
                    let last = route[route.count - 1]
                    var bestPosition = 0
                    var bestDistance = Double.infinity
                    for position in remaining.indices {
                        let candidate = try measure(last, remaining[position])
                        if candidate < bestDistance {
                            bestDistance = candidate
                            bestPosition = position
                        }
                    }
                    route.append(remaining.remove(at: bestPosition))
                }
                return try refine(route, passes: 4, window: route.count)
            }

            let otherPivot = try farthest(from: pivot, in: indices)
            var projections: [(index: Int, position: Double)] = []
            projections.reserveCapacity(indices.count)
            for index in indices {
                let fromFirst = try measure(pivot, index)
                let fromSecond = try measure(otherPivot, index)
                projections.append((index, fromFirst - fromSecond))
            }
            projections.sort {
                $0.position == $1.position ? $0.index < $1.index : $0.position < $1.position
            }
            let projected = projections.map(\.index)
            let midpoint = projected.count / 2
            let left = try build(Array(projected[..<midpoint]))
            let right = try build(Array(projected[midpoint...]))
            // Both halves are open paths, so orient their ends to minimize the
            // join without adding a closing edge between the final endpoints.
            var bestLeft = left
            var bestRight = right
            var bestJoin = Double.infinity
            for reverseLeft in [false, true] {
                for reverseRight in [false, true] {
                    let end = reverseLeft ? left[0] : left[left.count - 1]
                    let start = reverseRight ? right[right.count - 1] : right[0]
                    let join = try measure(end, start)
                    if join < bestJoin {
                        bestJoin = join
                        bestLeft = reverseLeft ? Array(left.reversed()) : left
                        bestRight = reverseRight ? Array(right.reversed()) : right
                    }
                }
            }
            return bestLeft + bestRight
        }

        let baseline = Array(0..<count)
        var route = try refine(build(baseline), passes: 2, window: min(count, 32))
        // Never worsen a sequence already supplied by the other sort signals.
        if try cost(baseline) <= cost(route) + 1e-9 { route = baseline }
        // Pick a deterministic forward orientation; reverse uses this exact path.
        if route[0] > route[route.count - 1] { route.reverse() }
        try Task.checkCancellation()
        return route
    }
}
