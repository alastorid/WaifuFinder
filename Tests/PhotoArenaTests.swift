import Foundation

func registerPhotoArenaTests() {
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "sparse schedule preserves cycle and decision history", knownBug: nil) {
        let sources = (0..<4).map { ArenaSource(id: "source-\($0)", url: URL(fileURLWithPath: "/fixture"), seconds: Double($0)) }
        var document = ArenaDocument(sources: sources)
        try document.validate()
        checkEqual(document.pairs.count, 4, "linear schedule")
        checkEqual(document.pairs.last?.1.id, sources[0].id, "closing edge permits cycles")
        for outcome in ArenaOutcome.allCases {
            document.votes.append(ArenaVote(id: UUID(), date: Date(), candidateA: sources[0].id,
                candidateB: sources[1].id, category: .authenticity, outcome: outcome,
                author: "human", region: "whole-image", evidence: "observable texture"))
        }
        let decoded = try JSONDecoder().decode(ArenaDocument.self, from: JSONEncoder().encode(document))
        try decoded.validate()
        checkEqual(decoded.votes.map(\.outcome), ArenaOutcome.allCases, "ties and abstentions survive export")
        checkEqual(decoded.votes.count, 4, "history retains disagreements")
        checkEqual(ArenaDocument(sources: Array(sources.prefix(2))).pairs.count, 1, "two sources have one match")
    })
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "invalid identities and schema rejected", knownBug: nil) {
        let source = ArenaSource(id: "same", url: URL(fileURLWithPath: "/fixture"), seconds: 0)
        do {
            try ArenaDocument(sources: [source, source]).validate()
            check(false, "duplicate identities must fail")
        } catch { }
        var document = ArenaDocument(sources: [source, ArenaSource(id: "other", url: source.url, seconds: 1)])
        document.schemaVersion = 2
        do { try document.validate(); check(false, "unknown schema must fail") } catch { }
    })
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "reversed preferences abstain on position bias", knownBug: nil) {
        func result(_ outcome: ArenaOutcome) -> ArenaJudgeResult {
            ArenaJudgeResult(judgment: ArenaJudgment(watermark: outcome, authenticity: outcome,
                detail: outcome, quality: outcome, consistency: outcome, recommended_source: outcome,
                uncertain: false, evidence: ["Visible detail"]), tokens: 10, prefillSeconds: 1,
                decodeSeconds: 1, physicalFootprintBytes: 100, compressedBytes: 0)
        }
        let biased = ArenaJudge.reconcile(result(.a), result(.a))
        checkEqual(biased.judgment.recommended_source, .unknown, "same position preference must abstain")
        check(biased.judgment.uncertain, "order disagreement is uncertain")
        let agrees = ArenaJudge.reconcile(result(.a), result(.b))
        checkEqual(agrees.judgment.recommended_source, .a, "reverse labels map to original source")
        checkEqual(ArenaJudge.reconcile(result(.tie), result(.tie)).judgment.detail, .tie, "ties survive reversal")
        do {
            try RemoteArenaRequest(revision: "unsupported", images: [Data([1]), Data([2])]).validate()
            check(false, "incompatible remote model must be rejected")
        } catch { }
    })

    Registry.shared.add(suite: "PhotoArena setup", TestCase(name: "Hugging Face release installs validates and reuses without changing user settings", knownBug: nil) {
        guard ProcessInfo.processInfo.environment["WAIFU_TEST_ARENA_INSTALL"] == "1" else {
            throw SkipTest(reason: "Opt in with WAIFU_TEST_ARENA_INSTALL=1 for the real Hugging Face download and worker validation")
        }
        let start = Date()
        let previous = try? Data(contentsOf: ArenaJudgeConfiguration.configurationURL)
        let root = try Temporary.directory("arena-huggingface-install")
        let configRoot = root.appendingPathComponent("configuration")
        let installer = ModelInstaller(configurationRoot: configRoot)
        let installed = try await installer.install(.arena, root: root.appendingPathComponent("models"), progress: { _, _ in })
        let reused = try await installer.install(.arena, root: root.appendingPathComponent("models"), progress: { _, _ in })
        checkEqual(installed, reused, "valid existing installation is reused")
        checkEqual(try String(contentsOf: configRoot.appendingPathComponent("photoarena-model.txt"), encoding: .utf8), installed.path, "validated worker activates isolated configuration")
        checkEqual(try? Data(contentsOf: ArenaJudgeConfiguration.configurationURL), previous, "user model configuration unchanged")
        let release = try ModelInstaller.release(.arena)
        if let report = ProcessInfo.processInfo.environment["WAIFU_TEST_ARENA_INSTALL_REPORT"] {
            let result: [String: Any] = ["status": "passed", "repository": release.repository,
                "revision": release.revision, "manifestSHA256": release.manifestSHA256,
                "verifiedFiles": try ModelInstaller.hashes(installed).keys.sorted(),
                "seconds": Date().timeIntervalSince(start), "nativeWorkerValidation": true,
                "reusedInstallation": installed == reused, "userConfigurationUnchanged": (try? Data(contentsOf: ArenaJudgeConfiguration.configurationURL)) == previous,
                "scope": "Real ModelInstaller Hugging Face download, every file checksum, native model load validation, isolated activation and reuse; no interactive UI."]
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: report))
        }
    })

    Registry.shared.add(suite: "PhotoArena", TestCase(name: "buffered worker framing preserves coalesced replies and refuses truncation", knownBug: nil) {
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(contentsOf: Data("ready\nfirst\nsecond\n".utf8))
        try pipe.fileHandleForWriting.close()
        let reader = ArenaPipeReader(pipe.fileHandleForReading)
        checkEqual(try reader.line(), Data("ready".utf8), "ready frame")
        checkEqual(try reader.line(), Data("first".utf8), "first coalesced frame")
        checkEqual(try reader.line(), Data("second".utf8), "second coalesced frame")
        do { _ = try reader.line(); check(false, "EOF rejected") } catch { }
        let truncated = Pipe()
        try truncated.fileHandleForWriting.write(contentsOf: Data("unfinished".utf8))
        try truncated.fileHandleForWriting.close()
        do { _ = try ArenaPipeReader(truncated.fileHandleForReading).line(); check(false, "partial frame rejected") } catch { }
    })
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "one action reuses one worker for paired and legacy replies", knownBug: nil) {
        let root = try Temporary.directory("arena-pair-protocol")
        let a = Data([1, 2, 3]), b = Data([4, 5])
        for paired in [false, true] {
            let directory = root.appendingPathComponent(paired ? "paired" : "legacy")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let worker = directory.appendingPathComponent("worker.sh")
            let first = ArenaJudgeResult(judgment: ArenaJudgment(watermark: .a, authenticity: .unknown, detail: .a,
                quality: .tie, consistency: .unknown, recommended_source: .a, uncertain: true, evidence: ["forward"]),
                tokens: 3, prefillSeconds: 0, decodeSeconds: 0, physicalFootprintBytes: 0, compressedBytes: 0)
            let reverse = ArenaJudgeResult(judgment: ArenaJudgment(watermark: .b, authenticity: .unknown, detail: .b,
                quality: .tie, consistency: .unknown, recommended_source: .b, uncertain: true, evidence: ["reverse"]),
                tokens: 4, prefillSeconds: 0, decodeSeconds: 0, physicalFootprintBytes: 0, compressedBytes: 0)
            let encoded = try [first, reverse].map { String(decoding: try JSONEncoder().encode($0), as: UTF8.self) }
            let header = paired ? "{\"ready\":true,\"pairedComparison\":true}" : "{\"ready\":true}"
            let read = "IFS= read -r request\nprintf '%s\\n' \"$request\" >> \"${1%/*}/requests.jsonl\"\n"
            let script = "#!/bin/sh\nprintf '%s\\n' '" + header + "'\n" + read +
                "printf '%s\\n' '" + encoded[0] + "'\n" + (paired ? "" : read) +
                "printf '%s\\n' '" + encoded[1] + "'\n"
            try script.write(to: worker, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: worker.path)
            let result = try await ArenaJudge.compare(images: [a, b], directory: directory, executable: worker)
            checkEqual(result.tokens, 7, "both outputs preserved")
            checkEqual(result.judgment.watermark, .a, "reverse order reconciled")
            let lines = try String(contentsOf: directory.appendingPathComponent("requests.jsonl"), encoding: .utf8).split(separator: "\n")
            checkEqual(lines.count, paired ? 1 : 2, "paired images uploaded once; legacy requests share one worker")
            let values = try lines.map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
            checkEqual(values[0]["images"] as? [String], [a.base64EncodedString(), b.base64EncodedString()], "streamed payload decodes exactly")
            if paired { checkEqual(values[0]["compareBoth"] as? Bool, true, "native paired flag") }
            else { checkEqual(values[1]["images"] as? [String], [b.base64EncodedString(), a.base64EncodedString()], "legacy reverse payload") }
        }
    })
    Registry.shared.add(suite: "PhotoArena", TestCase(name: "cancelling a paired local action reaps its worker", knownBug: nil) {
        let root = try Temporary.directory("arena-pair-cancel")
        let worker = root.appendingPathComponent("worker.sh")
        try "#!/bin/sh\nprintf '%s\\n' '{\"ready\":true,\"pairedComparison\":true}'\nwhile IFS= read -r request; do :; done\n".write(to: worker, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: worker.path)
        let task = Task { try await ArenaJudge.compare(images: [Data([1]), Data([2])], directory: root, executable: worker) }
        try await Task.sleep(for: .milliseconds(80))
        task.cancel()
        do { _ = try await task.value; check(false, "cancelled action returned a result") } catch { }
    })

}
