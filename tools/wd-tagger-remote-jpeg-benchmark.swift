// Actual native Metal/MLX inference over authenticated loopback TCP. No UI.
import CoreGraphics
import CryptoKit
import Foundation

private actor JPEGBenchmarkSource {
    private var position = 0
    func next() -> [Int] {
        guard position < 64 else { return [] }
        let end = min(position + 8, 64)
        defer { position = end }
        return Array(position..<end)
    }
}

private struct JPEGBenchmarkRows: Sendable {
    let indices: [Int]
    let rows: [Data]
}

private actor JPEGBenchmarkOutput {
    private var rows: [Int: Data] = [:]
    func commit(_ output: JPEGBenchmarkRows) throws {
        guard output.indices.count == output.rows.count else { throw TaggerError.invalidBatch }
        for (index, data) in zip(output.indices, output.rows) {
            guard rows.updateValue(data, forKey: index) == nil else { throw TaggerError.invalidBatch }
        }
    }
    func digest() throws -> String {
        guard rows.count == 64 else { throw TaggerError.invalidBatch }
        var data = Data()
        for index in 0..<64 { guard let row = rows[index] else { throw TaggerError.invalidBatch }; data.append(row) }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private func benchmarkJPEG(_ seed: Int) throws -> Data {
    let width = seed.isMultiple(of: 2) ? 512 : 384
    let height = seed.isMultiple(of: 2) ? 384 : 512
    var pixels = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let offset = (y * width + x) * 4
            pixels[offset] = UInt8((x + seed * 7) % 256)
            pixels[offset + 1] = UInt8((y + seed * 11) % 256)
            pixels[offset + 2] = UInt8((x / 3 + y / 7 + seed * 13) % 256)
        }
    }
    guard let provider = CGDataProvider(data: Data(pixels) as CFData),
          let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
        throw TaggerError.preprocessing
    }
    return try ThumbnailCache.encodeJPEG(image)
}

@main struct RemoteJPEGBenchmark {
    @MainActor static func main() async throws {
        let arguments = CommandLine.arguments
        func option(_ key: String) -> String? {
            arguments.firstIndex(of: key).flatMap { $0 + 1 < arguments.count ? arguments[$0 + 1] : nil }
        }
        guard let model = option("--model"), let worker = option("--worker"), let report = option("--report") else {
            throw ModelInstaller.fail("Use --model DIRECTORY --worker EXECUTABLE --report JSON [--rounds 3] [--depth 8]")
        }
        let rounds = Int(option("--rounds") ?? "3") ?? 3
        let depth = Int(option("--depth") ?? "8") ?? 8
        guard rounds > 0, (1...RemoteLimits.taggingBatchWindow).contains(depth) else { throw TaggerError.invalidBatch }
        let directory = URL(fileURLWithPath: model, isDirectory: true)
        let vocabulary = try TagVocabulary.load(from: directory.appendingPathComponent("wd-tagger-labels.csv"))
        let csv = try Data(contentsOf: directory.appendingPathComponent("wd-tagger-labels.csv"))
        let native = try WDTaggerBackend(directory: directory, executableURL: URL(fileURLWithPath: worker))
        let catalog = RemoteModelCatalog(tagger: { native }, backend: { $0.makeBackend() }, vocabularyCSV: { csv },
            vocabularyFingerprint: { vocabulary.cacheFingerprint }, capabilities: { version, name, session in
                RemoteCapabilities(protocolVersion: Int(RemoteLimits.protocolVersion), appVersion: version,
                    serverName: name, sessionID: session,
                    tagger: .init(available: true, imageSize: 448, batchSize: 8, headCount: TagVocabulary.headCount,
                                  vocabularyFingerprint: vocabulary.cacheFingerprint,
                                  inputCodecs: RemoteTagImageCodec.allCases.map(\.rawValue)),
                    uncover: [], vision: .init(available: true))
            })
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("waifu-native-jpeg-\(UUID().uuidString)")
        let defaultsName = "waifu.native-jpeg-benchmark.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: defaultsName)!
        let keys = RemoteFileKeyStore(url: temporary.appendingPathComponent("key"))
        try keys.saveKey(RemoteCrypto.generateKey())
        let server = RemoteServer(keyStore: keys, catalog: catalog, port: 0, advertises: false,
                                  appVersion: RemoteAppInfo.version, serverName: "Native JPEG benchmark")
        let client = RemoteClient(defaults: defaults, keyStore: keys, discoveryEnabled: false,
                                  vocabularyCacheDirectory: temporary.appendingPathComponent("vocabulary"))
        defer {
            client.setEnabled(false); server.stop()
            defaults.removePersistentDomain(forName: defaultsName)
            try? FileManager.default.removeItem(at: temporary)
        }
        server.start()
        let deadline = Date().addingTimeInterval(20)
        while true {
            if case .listening(let port) = server.state { client.manualAddress = "127.0.0.1:\(port)"; break }
            guard Date() < deadline else { throw ModelInstaller.fail("Benchmark listener did not start") }
            try await Task.sleep(for: .milliseconds(10))
        }
        client.setEnabled(true)
        while !client.status.isConnected {
            guard Date() < deadline else { throw ModelInstaller.fail("Benchmark authentication did not complete") }
            try await Task.sleep(for: .milliseconds(10))
        }
        await client.loadVocabularyIfNeeded()
        let remote = RemoteTaggerBackend(client: client, threshold: 0.35, vocabularyFingerprint: vocabulary.cacheFingerprint)
        let cached = try (0..<64).map(benchmarkJPEG)
        let firstSources = cached.prefix(8).map { TaggingImageSource.thumbnail($0) }
        _ = try await remote.tag(sources: firstSources) // Native model compilation/warmup is excluded.
        let firstImages = try await TaggingImageLoader.load(firstSources)
        _ = try await remote.tag(batch: firstImages)
        let jpegPayloadBytes = try await RemoteTagBatchPayload.encode(sources: firstSources).count
        let rgbaPayloadBytes = try await RemoteTagBatchPayload.encode(images: firstImages, codec: .rgba8LZ4).count

        var measurements: [[String: Any]] = []
        var hashes = Set<String>()
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        for round in 0..<rounds {
            for direct in round.isMultiple(of: 2) ? [false, true] : [true, false] {
                let source = JPEGBenchmarkSource(), output = JPEGBenchmarkOutput()
                let started = ProcessInfo.processInfo.systemUptime
                try await TaggingBatchPipeline.runPipelined(maximumInFlight: depth, commitInOrder: false,
                    next: { await source.next() }, whenEmpty: { false }, process: { indices in
                        let sources = indices.map { TaggingImageSource.thumbnail(cached[$0]) }
                        let results: [TagResult]
                        if direct { results = try await remote.tag(sources: sources) }
                        else {
                            let images = try await TaggingImageLoader.load(sources)
                            results = try await remote.tag(batch: images)
                        }
                        return try JPEGBenchmarkRows(indices: indices, rows: results.map {
                            guard let bytes = $0.encodedProbabilities else { throw TaggerError.invalidProbabilities }
                            return bytes
                        })
                    }, commit: { try await output.commit($0) })
                let seconds = ProcessInfo.processInfo.systemUptime - started
                hashes.insert(try await output.digest())
                let mode = direct ? "cached-jpeg" : "client-rgba-lz4"
                measurements.append(["mode": mode, "round": round, "seconds": seconds, "fps": 64 / seconds,
                                     "thermal": ProcessInfo.processInfo.thermalState.rawValue])
                print(String(format: "%@: %.3f seconds, %.2f fps", mode, seconds, 64 / seconds))
                fflush(stdout)
            }
        }
        guard hashes.count == 1 else { throw ModelInstaller.fail("JPEG and RGBA paths changed quantized model results") }
        var medians: [String: Double] = [:]
        for mode in ["cached-jpeg", "client-rgba-lz4"] {
            let seconds = measurements.filter { $0["mode"] as? String == mode }.compactMap { $0["seconds"] as? Double }.sorted()
            medians[mode] = 64 / seconds[seconds.count / 2]
        }
        let result: [String: Any] = ["frames": 64, "rounds": rounds, "requestDepth": depth,
            "firstBatchPayloadBytes": ["cached-jpeg": jpegPayloadBytes, "client-rgba-lz4": rgbaPayloadBytes],
            "measurements": measurements, "medianFPS": medians, "outputSHA256": hashes.first!,
            "quantizedResultsIdentical": true, "thermalBefore": thermalBefore,
            "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue,
            "scope": "Actual production native Metal/MLX worker, preprocessing, encrypted loopback TCP and result derivation on one Mac. Generated cached 512px portrait/landscape JPEGs. Alternating mode order; model warmup excluded. No PhotoKit fetch, physical LAN, second host, or interactive UI. Both modes start from identical JPEG pixels; the RGBA control includes client decode, preprocessing and negotiated LZ4." ]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: report))
    }
}
