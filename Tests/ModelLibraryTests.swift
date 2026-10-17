import Foundation

func registerModelLibraryTests() {
    let suite = "Models"
    Registry.shared.add(suite: suite, TestCase(name: "published native WD Tagger installs beside Core ML and activates", knownBug: nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_TEST_TAGGER_INSTALL"] == "1" else {
            throw SkipTest(reason: "opt in with WAIFU_TEST_TAGGER_INSTALL=1")
        }
        let configURL = WDTaggerConfiguration.configurationURL
        let previous = try? Data(contentsOf: configURL)
        defer {
            if let previous { try? previous.write(to: configURL, options: .atomic) }
            else { try? FileManager.default.removeItem(at: configURL) }
        }
        let root = try Temporary.directory("wd-native-install")
        let legacy = root.appendingPathComponent("wd-tagger")
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
        let marker = legacy.appendingPathComponent("previous-model")
        try Data("preserve Core ML installation".utf8).write(to: marker)
        let installer = ModelInstaller()
        let installed = try await installer.install(.tagger, root: root, progress: { _, _ in })
        checkEqual(installed.lastPathComponent, "wd-tagger-native-v2", "native release uses a separate folder")
        checkEqual(try String(contentsOf: marker, encoding: .utf8), "preserve Core ML installation", "old model survives")
        checkEqual(WDTaggerConfiguration.load()?.directory, installed, "verified native release activates")
        check(WDTaggerBackend.isAvailable(directory: installed), "native model and labels are available")
        let backend = try WDTaggerBackend(directory: installed)
        let image = try ImageFixture.cgImage(width: 448, height: 448, gray: 0)
        checkEqual(try await backend.tag(batch: [image]).count, 1, "downloaded release runs through the app adapter")
        let reused = try await installer.install(.tagger, root: root, progress: { _, _ in })
        checkEqual(reused, installed, "repeated setup verifies and reuses native release")
    })
    Registry.shared.add(suite: suite, TestCase(name: "published fused PGTFormer installs alongside an older model and activates", knownBug: nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_TEST_PGT_INSTALL"] == "1" else {
            throw SkipTest(reason: "opt in to the published model download with WAIFU_TEST_PGT_INSTALL=1")
        }
        let configURL = PGTFormerConfiguration.configurationURL
        let previous = try? Data(contentsOf: configURL)
        defer {
            if let previous { try? previous.write(to: configURL, options: .atomic) }
            else { try? FileManager.default.removeItem(at: configURL) }
        }
        let root = try Temporary.directory("pgt-published-install")
        let legacy = root.appendingPathComponent("pgtformer")
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
        let marker = legacy.appendingPathComponent("previous-model")
        try Data("preserve the old installation".utf8).write(to: marker)
        let installer = ModelInstaller()
        let installed = try await installer.install(.pgt, root: root, progress: { _, _ in })
        check(installed != legacy, "upgrade uses a separate installation directory")
        checkEqual(try String(contentsOf: marker, encoding: .utf8), "preserve the old installation", "setup preserves the old model")
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: installed.appendingPathComponent("manifest.json"))) as! [String:Any]
        checkEqual((manifest["optimizations"] as? [String:Any])?["fusedCodeAttention"] as? Bool, true, "the downloaded release contains fused attention")
        checkEqual(PGTFormerConfiguration.load()?.model, installed.path, "setup activates the downloaded package")
        check(await PGTFormerBackend().capabilities().available, "the app accepts the published fused package")
        let reused = try await installer.install(.pgt, root: root, progress: { _, _ in })
        checkEqual(reused, installed, "repeated setup verifies and reuses the current release")
    })
    Registry.shared.add(suite: suite, TestCase(name: "model verification rejects traversal and escaping symlinks", knownBug: nil) {
        let root = try Temporary.directory("model-paths")
        let hash = String(repeating: "a", count: 64)
        for name in ["../model", "/model", "a/../../model", "a\\model", "a//model", "a/./model"] {
            check(!ModelInstaller.safeFile(name, hash: hash), "reject unsafe path \(name)")
        }
        check(ModelInstaller.safeFile("transformer/block-001.safetensors", hash: hash), "nested model files supported")
        let external = try Temporary.directory("model-external")
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("transformer"), withDestinationURL: external)
        _ = checkThrows("reject symlink escaping the install folder") { try ModelInstaller.checkedFile(root, "transformer/model", hash) }
    })
    Registry.shared.add(suite: suite, TestCase(name: "verified downloads reuse files and preserve an existing model on checksum failure", knownBug: nil) {
        let root = try Temporary.directory("model-download")
        let server = try ModelDownloadFixture()
        defer { server.stop() }
        let target = root.appendingPathComponent("model.bin")
        let existing = Data("old model".utf8)
        try existing.write(to: target)
        let hash = try ModelInstaller.digest(target)
        let installer = ModelInstaller()
        try await installer.fetch(server.base.appendingPathComponent("missing"), to: target, hash: hash, progress: { _, _ in })
        checkEqual(try Data(contentsOf: target), existing, "verified file reused without network request")
        _ = await checkThrowsAsync("checksum failure is surfaced") {
            try await installer.fetch(server.base.appendingPathComponent("model"), to: target, hash: String(repeating: "0", count: 64), progress: { _, _ in })
        }
        checkEqual(try Data(contentsOf: target), existing, "old model preserved on failure")
        let reference = root.appendingPathComponent("reference.bin")
        try Data("new model".utf8).write(to: reference)
        try await installer.fetch(server.base.appendingPathComponent("model"), to: target, hash: ModelInstaller.digest(reference), progress: { _, _ in })
        checkEqual(try Data(contentsOf: target), Data("new model".utf8), "verified replacement installed")
    })
    Registry.shared.add(suite: suite, TestCase(name: "cancelling a model download preserves installed files", knownBug: nil) {
        let root = try Temporary.directory("model-cancel")
        let server = try ModelDownloadFixture()
        defer { server.stop() }
        let target = root.appendingPathComponent("model.bin")
        let existing = Data("installed model".utf8)
        try existing.write(to: target)
        let installer = ModelInstaller()
        let slowURL = server.base.appendingPathComponent("slow")
        let task = Task {
            try await installer.fetch(slowURL, to: target, hash: String(repeating: "0", count: 64), progress: { _, _ in })
        }
        try await Task.sleep(for: .milliseconds(200))
        task.cancel()
        _ = await checkThrowsAsync("cancelled transfer throws") { try await task.value }
        checkEqual(try Data(contentsOf: target), existing, "cancellation leaves the installed model intact")
    })
    Registry.shared.add(suite: suite, TestCase(name: "every managed model has a pinned hosted release", knownBug: nil) {
        // PGTFormer and RealESRGAN used to arrive from a GitHub release or from a
        // hand-picked folder. A model with no descriptor is one the app cannot
        // download, verify or offer, so this fails loudly rather than at runtime.
        let missing = ManagedModel.allCases.filter { $0.releaseName == nil }
        checkEqual(missing.map(\.rawValue), [], "every managed model is downloadable")
    })
    Registry.shared.add(suite: suite, TestCase(name: "a release manifest must pin its model files, and Core ML packages are found by manifest", knownBug: nil) {
        let root = try Temporary.directory("model-manifest")
        let hash = String(repeating: "b", count: 64)
        let empty: [String: Any] = ["algorithm": "realesrgan-x2plus", "validated": true]
        try JSONSerialization.data(withJSONObject: empty).write(to: root.appendingPathComponent("manifest.json"))
        _ = checkThrows("a manifest without fileSHA256 is rejected") { try ModelInstaller.hashes(root) }
        let populated: [String: Any] = [
            "fileSHA256": [
                "RealESRGANValidated.mlpackage/Manifest.json": hash,
                "RealESRGANValidated.mlpackage/Data/com.apple.CoreML/weights/weight.bin": hash,
            ],
        ]
        try JSONSerialization.data(withJSONObject: populated).write(to: root.appendingPathComponent("manifest.json"))
        checkEqual(try ModelInstaller.hashes(root).count, 2, "both pinned members are read")
        checkEqual(try ModelInstaller.package(root, try ModelInstaller.hashes(root), "RealESRGANValidated.mlpackage")?.path,
                   root.appendingPathComponent("RealESRGANValidated.mlpackage").path,
                   "the package directory is located from the manifest, not assumed")
        checkNil(try ModelInstaller.package(root, try ModelInstaller.hashes(root), "PGTEncoder.mlpackage"),
                 "a package the release does not contain is absent, not invented")
        _ = checkThrows("a package path escaping the release folder is refused") {
            try ModelInstaller.package(root, ["../escape/Manifest.json": hash], "../escape")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "a commercial build never offers a non-commercial model", knownBug: nil) {
        // The default build is non-commercial and offers everything. A commercial
        // build must drop PGTFormer everywhere, not merely refuse to download it.
        checkEqual(ManagedModel.pgt.isNonCommercialRestricted, true, "PGTFormer is licence-restricted")
        for model in ManagedModel.allCases where model != .pgt {
            checkEqual(model.isNonCommercialRestricted, false, "\(model.rawValue) is permissively licensed")
        }
        let offered = Set(ManagedModel.offered)
        if WaifuDistribution.isNonCommercialOnly {
            checkEqual(ManagedModel.offered.count, ManagedModel.allCases.count,
                       "the default build offers every model")
        } else {
            check(!offered.contains(.pgt), "a commercial build does not offer PGTFormer")
            checkEqual(ManagedModel.offered.count, ManagedModel.allCases.count - 1,
                       "only the restricted model is dropped")
        }
        // A session saved while PGTFormer was offered must still decode and report
        // itself unavailable, rather than failing to load.
        checkNotNil(UncoverAlgorithm(rawValue: "pgtFormer"), "PGTFormer still parses in a commercial build")
        if WaifuDistribution.isNonCommercialOnly {
            checkEqual(UncoverAlgorithm.offered.count, UncoverAlgorithm.allCases.count,
                       "the default build can select every algorithm")
        } else {
            check(!UncoverAlgorithm.offered.contains(.pgtFormer), "PGTFormer is not selectable in one")
            checkEqual(UncoverAlgorithm.offered.count, UncoverAlgorithm.allCases.count - 1,
                       "every other algorithm is still offered")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "explicitly disconnected SeedVR2 variants cannot reactivate from sibling discovery", knownBug: nil) {
        let root = try Temporary.directory("model-disabled")
        let model = root.appendingPathComponent("model-optimized")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let manifest: [String: Any] = ["algorithm": "seedvr2-3b", "fusion": "packed-projections-v1", "vaeFusion": "qkv-packed-v1", "precision": "mlx-4bit", "groupSize": 64]
        try JSONSerialization.data(withJSONObject: manifest).write(to: model.appendingPathComponent("manifest.json"))
        var config = SeedVR2Configuration(python: "", model: model.path, helper: "", allowUnsafeMemory: false, nativeExecutable: "/worker")
        checkNotNil(config.optimized(variant: "seedvr2-3b"), "configured bundle discovered")
        config.disconnectedVariants = ["seedvr2-3b"]
        checkNil(config.optimized(variant: "seedvr2-3b"), "explicit disconnect defeats fallback discovery")
        let decoded = try JSONDecoder().decode(SeedVR2Configuration.self, from: JSONEncoder().encode(config))
        checkNil(decoded.optimized(variant: "seedvr2-3b"), "disconnect survives relaunch")
    })
}

private final class ModelDownloadFixture {
    let process = Process()
    let base: URL
    init() throws {
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-u", "-c", """
import http.server, time
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == '/missing':
            self.send_error(404); return
        data = b'new model' if self.path == '/model' else b'x' * 1024 * 1024
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            if self.path == '/slow':
                for i in range(0, len(data), 4096):
                    self.wfile.write(data[i:i+4096]); self.wfile.flush(); time.sleep(.02)
            else: self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError): pass
server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""]
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        let output = pipe.fileHandleForReading.availableData
        guard let port = Int(String(decoding: output, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)),
              let url = URL(string: "http://127.0.0.1:\(port)") else {
            process.terminate()
            throw ModelInstaller.fail("Could not start local download fixture.")
        }
        base = url
    }
    func stop() { if process.isRunning { process.terminate() }; process.waitUntilExit() }
}
