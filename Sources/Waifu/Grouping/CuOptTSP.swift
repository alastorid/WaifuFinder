// SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified for WaifuFinder: native CPU reference for a symmetric open-path subset
// of cuOpt's find_two_opt_moves / execute_two_opt_moves. See ThirdParty/cuopt/NOTICE.

import Foundation

enum TSPError: Error {
    case invalidMatrix, invalidTour, metalUnavailable, allocationFailed, commandFailed
}

struct TSPDistanceMatrix: Sendable {
    let count: Int
    let values: [Float]

    init(count: Int, values: [Float]) throws {
        guard count > 0, count <= Int.max / count, values.count == count * count else { throw TSPError.invalidMatrix }
        for row in 0..<count {
            try Task.checkCancellation()
            for column in row..<count {
                let value = values[row * count + column]
                guard value.isFinite, value >= 0,
                      values[column * count + row] == value,
                      row != column || value == 0 else { throw TSPError.invalidMatrix }
            }
        }
        self.count = count
        self.values = values
    }

    subscript(_ left: Int, _ right: Int) -> Float { values[left * count + right] }

    func cost(_ route: [Int]) -> Double {
        zip(route, route.dropFirst()).reduce(0) { $0 + Double(self[$1.0, $1.1]) }
    }
}

enum CuOptTSP {
    static let epsilon: Float = 1e-6

    static func validate(_ seeds: [[Int]], count: Int) throws {
        guard !seeds.isEmpty else { throw TSPError.invalidTour }
        for seed in seeds {
            guard seed.count == count, Set(seed) == Set(0..<count) else { throw TSPError.invalidTour }
        }
    }

    static func seeds(from route: [Int]) -> [[Int]] {
        guard route.count > 3 else { return [route] }
        let split = route.count / 3
        let rotated = Array(route[split...]) + route[..<split]
        var perturbed = route
        perturbed[(route.count / 4)..<(route.count * 3 / 4)].reverse()
        // Different open endpoints and a perturbed interior explore different
        // local minima; a whole-route reversal would be the same search seed.
        return [route, Array(0..<route.count), rotated, perturbed]
    }

    /// Best improving 2-opt move per immutable route snapshot, then execute it.
    /// Both endpoint moves are allowed; there is no depot or closing edge.
    static func solveCPU(_ matrix: TSPDistanceMatrix, seeds: [[Int]], iterations: Int = 128) throws -> [Int] {
        try validate(seeds, count: matrix.count)
        var winner = seeds[0]
        var winnerCost = matrix.cost(winner)
        for seed in seeds {
            var route = seed
            for _ in 0..<max(0, iterations) {
                try Task.checkCancellation()
                var bestDelta: Float = -epsilon
                var best: (Int, Int)?
                for first in 0..<max(0, route.count - 1) {
                    for second in (first + 1)..<route.count {
                        if first == 0 && second == route.count - 1 { continue }
                        var before: Float = 0
                        var after: Float = 0
                        if first > 0 {
                            before += matrix[route[first - 1], route[first]]
                            after += matrix[route[first - 1], route[second]]
                        }
                        if second + 1 < route.count {
                            before += matrix[route[second], route[second + 1]]
                            after += matrix[route[first], route[second + 1]]
                        }
                        let delta = after - before
                        if delta < bestDelta { bestDelta = delta; best = (first, second) }
                    }
                }
                guard let (first, second) = best else { break }
                route[first...second].reverse()
            }
            let cost = matrix.cost(route)
            if cost < winnerCost { winner = route; winnerCost = cost }
        }
        return winner
    }
}
