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

}
