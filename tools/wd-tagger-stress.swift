import AppKit
import Foundation

@main struct TaggerStress {
    static func main() async throws {
        setbuf(stdout, nil)
        let args = CommandLine.arguments
        func option(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), index + 1 < args.count else { return nil }
            return args[index + 1]
        }
        guard let directory = option("--model") else { throw TaggerError.invalidModel("Supply --model DIRECTORY") }
        let count = Int(option("--instances") ?? "2") ?? 2
        let rounds = Int(option("--rounds") ?? "12") ?? 12
        let location = URL(fileURLWithPath: directory)
        var backends: [WDTaggerBackend] = []
        for _ in 0..<count { backends.append(try WDTaggerBackend(directory: location)) }
        let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: "Tests/Fixtures/WDTagger/rgb-gradient.png") as CFURL, nil)!
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
        let context = CGContext(data: nil, width: 448, height: 448, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(gray: 0, alpha: 1); context.fill(CGRect(x: 0, y: 0, width: 448, height: 448))
        let dark = context.makeImage()!
        let darkReference = try await backends[0].tag(batch: [dark])[0].probabilities
        let reference = try await backends[0].tag(batch: [image])[0].probabilities
        print("Starting \(count) independent models × \(rounds) mixed batches")
        try await withThrowingTaskGroup(of: Void.self) { group in
            for (index, backend) in backends.enumerated() {
                group.addTask {
                    for round in 0..<rounds {
                        let batchCount = round.isMultiple(of: 3) ? 1 : 8
                        let batch = (0..<batchCount).map { ($0 + round).isMultiple(of: 2) ? image : dark }
                        let results = try await backend.tag(batch: batch)
                        guard results.count == batchCount else { throw TaggerError.invalidProbabilities }
                        for (row, result) in results.enumerated() {
                            let expected = (row + round).isMultiple(of: 2) ? reference : darkReference
                            let error = zip(result.probabilities, expected).map { abs($0 - $1) }.max()!
                            guard error < 0.0001 else { throw TaggerError.invalidModel("Cross-call corruption in model \(index), batch \(round), row \(row): \(error)") }
                        }
                        print("model \(index) batch \(round) passed")
                    }
                }
            }
            try await group.waitForAll()
        }
        print("All concurrent mixed batches passed")
    }
}
