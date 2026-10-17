import Foundation
import Metal
import Vision

private func testedMetalSolver() throws -> CuOptMetalTSP {
    guard MTLCreateSystemDefaultDevice()?.hasUnifiedMemory == true else {
        throw SkipTest(reason: "Apple Silicon Metal device unavailable")
    }
    // A shader compilation error on supported hardware is a failure, not a skip.
    if let solver = CuOptMetalTSP.shared { return solver }
    return try CuOptMetalTSP()
}

private func bruteForceOpenCost(_ matrix: TSPDistanceMatrix) -> Double {
    var path = Array(0..<matrix.count)
    var best = Double.infinity
    func search(_ position: Int) {
        if position == path.count { best = min(best, matrix.cost(path)); return }
        for next in position..<path.count {
            path.swapAt(position, next)
            search(position + 1)
            path.swapAt(position, next)
        }
    }
    search(0)
    return best
}

func registerCuOptMetalTSPTests() {
    let suite = "frame arrangement: cuOpt Metal"
    Registry.shared.add(suite: suite, TestCase(name: "small CPU and Metal paths agree with brute-force open optima", knownBug: nil) {
        let solver = try testedMetalSolver()
        for points: [Float] in [[0, 10, 1, 9, 2, 8], [0, 0, 2, 0, 3, 1, 2, 3, 0, 2]] {
            let dimensions = points.count == 6 ? 1 : 2
            let vectors = VisualFeatureVectors(count: points.count / dimensions, dimensions: dimensions, values: points)!
            let matrix = try vectors.matrix()
            let seeds = CuOptTSP.seeds(from: Array(0..<matrix.count))
            let optimum = bruteForceOpenCost(matrix)
            let cpu = try CuOptTSP.solveCPU(matrix, seeds: seeds)
            let gpu = try solver.solve(matrix, seeds: seeds)
            checkClose(matrix.cost(cpu), optimum, tolerance: 1e-5, "CPU reaches the known open optimum")
            checkClose(matrix.cost(gpu), optimum, tolerance: 1e-5, "GPU reaches the known open optimum")
            checkEqual(Set(gpu), Set(0..<matrix.count), "GPU result is a permutation")
            checkEqual(gpu.count, matrix.count, "every node occurs once")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "GPU vector distances match CPU including SIMD tails and duplicates", knownBug: nil) {
        let solver = try testedMetalSolver()
        let count = 67, dimensions = 37
        let values = (0..<(count * dimensions)).map { index in
            Float(((index / dimensions) % 13 * 17 + index % dimensions * 3) % 101) / 101
        }
        for metric in [VisualFeatureVectors.Metric.euclidean, .squaredEuclidean] {
            let vectors = VisualFeatureVectors(count: count, dimensions: dimensions, values: values, metric: metric)!
            let cpu = try vectors.matrix()
            let gpu = try solver.distances(vectors)
            for index in cpu.values.indices {
                checkClose(Double(gpu.values[index]), Double(cpu.values[index]), tolerance: 2e-5, "GPU distance at \(index)")
            }
            checkEqual(gpu[0, 13], 0, "identical vectors have exactly zero distance")
            checkEqual(gpu[66, 66], 0, "last partial dispatch writes the diagonal")
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "raw Vision embedding and GPU distance preserve the grouping metric", knownBug: nil) {
        let observations = try await [makeFeaturePrintObservation(0), makeFeaturePrintObservation(2), makeFeaturePrintObservation(4)]
        guard let vectors = checkNotNil(VisualFeatureVectors(observations: observations), "current Vision float embeddings validate") else { return }
        let solver = try testedMetalSolver()
        let matrix = try solver.distances(vectors)
        for left in observations.indices {
            for right in observations.indices {
                let native = try observations[left].distance(to: observations[right])
                check(VisualFeatureVectors.matches(native, Double(matrix[left, right])), "Metal matches native Vision for \(left),\(right)")
            }
        }
    })

    Registry.shared.add(suite: suite, TestCase(name: "snapshot reductions and multi-route moves match the CPU reference", knownBug: nil) {
        let count = 33
        var values: [Float] = []
        for index in 0..<count {
            let x = Float((index * 19) % 37) / Float(37)
            let y = Float((index * 11 + 7) % 41) / Float(41)
            values.append(contentsOf: [x, y])
        }
        let matrix = try VisualFeatureVectors(count: count, dimensions: 2, values: values)!.matrix()
        let seeds = CuOptTSP.seeds(from: Array(0..<count))
        let solver = try testedMetalSolver()
        let cpu = try CuOptTSP.solveCPU(matrix, seeds: seeds, iterations: 40)
        let gpu = try solver.solve(matrix, seeds: seeds, iterations: 40)
        checkClose(matrix.cost(gpu), matrix.cost(cpu), tolerance: 1e-5, "same improving moves and objective")
        check(matrix.cost(gpu) < matrix.cost(seeds[0]), "a no-op implementation cannot pass")
        checkEqual(gpu.count, count, "move execution preserves length")
        checkEqual(Set(gpu), Set(0..<count), "no overlap duplicates or loses nodes")
    })

    Registry.shared.add(suite: suite, TestCase(name: "bounded vector candidate mode avoids the dense matrix", knownBug: nil) {
        let count = 33
        let values = (0..<count).flatMap { index -> [Float] in
            let position = Float((index * 13) % count)
            return [position, position * 0.5]
        }
        let vectors = VisualFeatureVectors(count: count, dimensions: 2, values: values)!
        let matrix = try vectors.matrix()
        let seeds = Array(CuOptTSP.seeds(from: Array(0..<count)).prefix(2))
        let solver = try testedMetalSolver()
        let gpu = try solver.solve(vectors, seeds: seeds, iterations: 32, denseLimit: 0)
        let cpu = try CuOptTSP.solveCPU(matrix, seeds: seeds, iterations: 32)
        checkClose(matrix.cost(gpu), matrix.cost(cpu), tolerance: 1e-4, "bounded GPU edge search matches the reference for this candidate set")
        check(matrix.cost(gpu) < matrix.cost(seeds[0]) * 0.5, "candidate mode actively improves the route")
        checkEqual(Set(gpu), Set(0..<count), "every node remains in the sparse route")
    })

    Registry.shared.add(suite: suite, TestCase(name: "invalid matrices, vectors and tours fail before GPU indexing", knownBug: nil) {
        _ = checkThrows("asymmetric costs rejected", { try TSPDistanceMatrix(count: 2, values: [0, 1, 2, 0]) })
        _ = checkThrows("non-finite costs rejected", { try TSPDistanceMatrix(count: 2, values: [0, .nan, .nan, 0]) })
        _ = checkThrows("matrix shape checked", { try TSPDistanceMatrix(count: 3, values: [0]) })
        checkNil(VisualFeatureVectors(count: 2, dimensions: 2, values: [0]), "vector shape checked")
        checkNil(VisualFeatureVectors(count: 1, dimensions: 1, values: [.infinity]), "non-finite vectors rejected")
        let matrix = try TSPDistanceMatrix(count: 2, values: [0, 1, 1, 0])
        _ = checkThrows("CPU rejects repeated nodes", { try CuOptTSP.solveCPU(matrix, seeds: [[0, 0]]) })
        let solver = try testedMetalSolver()
        _ = checkThrows("GPU rejects out-of-range nodes", { try solver.solve(matrix, seeds: [[0, 2]]) })
        let large = VisualFeatureVectors(count: 4_097, dimensions: 1, values: Array(repeating: 0, count: 4_097))!
        _ = checkThrows("dense GPU allocation has a fixed bound", { try solver.distances(large) })
        _ = checkThrows("dense CPU allocation has the same bound", { try large.matrix() })
    })

    Registry.shared.add(suite: suite, TestCase(name: "sampled candidate mode preserves a squared-distance route", knownBug: nil) {
        let count = 129
        var values: [Float] = []
        for index in 0..<count {
            let x = Float((index * 17) % count) / Float(count)
            values.append(contentsOf: [x, x * x, sin(x)])
        }
        let vectors = VisualFeatureVectors(count: count, dimensions: 3, values: values, metric: .squaredEuclidean)!
        let matrix = try vectors.matrix()
        let seeds = CuOptTSP.seeds(from: Array(0..<count))
        let solver = try testedMetalSolver()
        let route = try solver.solve(vectors, seeds: seeds, iterations: 16, denseLimit: 0)
        checkEqual(Set(route), Set(0..<count), "sampled nonlocal moves retain all IDs")
        checkEqual(route.count, count, "sampled moves never duplicate an ID")
        check(matrix.cost(route) < matrix.cost(seeds[0]), "sampled GPU search reduces the squared objective")
    })

    Registry.shared.add(suite: suite, TestCase(name: "the image sorter orders a GPU-sized set using the native Vision objective", knownBug: nil) {
        _ = try testedMetalSolver()
        let observations = try await [makeFeaturePrintObservation(0), makeFeaturePrintObservation(2), makeFeaturePrintObservation(4)]
        let payloads = try observations.map { try JSONEncoder().encode($0) }
        let frames = (0..<513).map { index in
            FrameFixture.make(id: String(format: "f%06d", index), time: Double(index), featurePrint: payloads[index % 3])
        }
        let sorted = try MinimalVisualDistanceOrder.sorted(frames)
        checkEqual(Set(sorted.map(\.id)), Set(frames.map(\.id)), "actual image integration preserves the whole shown set")
        let transitions = zip(sorted, sorted.dropFirst()).filter {
            $0.0.id.sampleIndex % 3 != $0.1.id.sampleIndex % 3
        }.count
        checkEqual(transitions, 2, "three distinct appearances form a continuous sequence with only two transitions")
        checkEqual(try MinimalVisualDistanceOrder.sorted(frames, reversed: true).map(\.id),
                   Array(sorted.reversed()).map(\.id), "GPU image order is deterministic and exactly reversible")
    })

    Registry.shared.add(suite: suite, TestCase(name: "canceled GPU jobs cannot publish a route", knownBug: nil) {
        let solver = try testedMetalSolver()
        let matrix = try TSPDistanceMatrix(count: 2, values: [0, 1, 1, 0])
        let task = Task.detached { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try solver.solve(matrix, seeds: [[0, 1]])
                return false
            } catch is CancellationError { return true }
            catch { return false }
        }
        check(await task.value, "GPU entry respects task cancellation")
    })
}
