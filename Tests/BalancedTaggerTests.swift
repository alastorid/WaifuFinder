import CoreGraphics
import Foundation

private actor BalancedLaneProbe: TaggerBackend {
    private var waiting: [CheckedContinuation<Void, Never>] = []
    private(set) var arrivals = 0
    private var open = false
    var fails = false

    func tag(batch: [CGImage]) async throws -> [TagResult] {
        arrivals += 1
        if !open { await withCheckedContinuation { waiting.append($0) } }
        try Task.checkCancellation()
        if fails { throw TaggerError.preprocessing }
        return [TagResult(probabilities: [], rating: .general, topTags: [], generalScore: 0)]
    }

    func release() {
        open = true
        let pending = waiting
        waiting = []
        for continuation in pending { continuation.resume() }
    }

    func setFails(_ value: Bool) { fails = value }
}

private func waitForBalancedArrivals(_ probes: [BalancedLaneProbe], total: Int) async throws {
    let deadline = Date().addingTimeInterval(5)
    while Date() < deadline {
        var count = 0
        for probe in probes { count += await probe.arrivals }
        if count >= total { return }
        try await Task.sleep(for: .milliseconds(2))
    }
    throw TaggerError.invalidBatch
}

private actor UnorderedTaggingProbe {
    var fetched = 0
    var committed: [Int] = []
    func next() -> [Int] {
        defer { fetched += 1 }
        return fetched < 10 ? [fetched] : []
    }
    func process(_ batch: [Int]) async throws -> Int {
        let value = batch[0]
        if value == 0 {
            let deadline = Date().addingTimeInterval(5)
            while committed.count < 5 {
                guard Date() < deadline else { throw TaggerError.preprocessing }
                try await Task.sleep(for: .milliseconds(2))
            }
        }
        return value
    }
    func commit(_ value: Int) { committed.append(value) }
}

func registerBalancedTaggerTests() {
    let suite = "Remote balancing"
    Registry.shared.add(suite: suite, TestCase(name: "queue depth spreads work and sends new batches to a freed server", knownBug: nil) {
        let first = BalancedLaneProbe()
        let second = BalancedLaneProbe()
        let local = BalancedLaneProbe()
        let backend = BalancedTaggerBackend(lanes: [
            .init(backend: first, capacity: 2), .init(backend: second, capacity: 2),
            .init(backend: local, capacity: 1)
        ])
        checkEqual(backend.maximumInFlight, 5, "window covers all servers and one local batch")
        let tasks = (0..<5).map { _ in Task { try await backend.tag(batch: []) } }
        do {
            try await waitForBalancedArrivals([first, second, local], total: 5)
            checkEqual(await first.arrivals, 2, "first remote has two pending batches")
            checkEqual(await second.arrivals, 2, "second remote has two pending batches")
            checkEqual(await local.arrivals, 1, "local model has no inference backlog")
            await first.release()
            // Wait for the scheduler to observe the completed remote requests.
            var accepted = false
            for _ in 0..<100 {
                do { _ = try await backend.tag(batch: []); accepted = true; break }
                catch { try await Task.sleep(for: .milliseconds(2)) }
            }
            check(accepted, "a free server takes work while other lanes are occupied")
            checkEqual(await first.arrivals, 3, "batch goes to the least-loaded server")
        } catch {
            await first.release(); await second.release(); await local.release()
            for task in tasks { _ = await task.result }
            throw error
        }
        await second.release(); await local.release()
        for task in tasks { _ = try await task.value }
    })

    Registry.shared.add(suite: suite, TestCase(name: "errors and cancellation release lane reservations", knownBug: nil) {
        let probe = BalancedLaneProbe()
        await probe.release()
        await probe.setFails(true)
        let backend = BalancedTaggerBackend(lanes: [.init(backend: probe, capacity: 1)])
        _ = await checkThrowsAsync("backend error") { _ = try await backend.tag(batch: []) }
        await probe.setFails(false)
        checkEqual(try await backend.tag(batch: []).count, 1, "error does not consume capacity")
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await backend.tag(batch: [])
        }
        _ = await task.result
        checkEqual(try await backend.tag(batch: []).count, 1, "cancelled submission leaves capacity available")
    })

    Registry.shared.add(suite: suite, TestCase(name: "fast workers refill while a slow batch remains in flight", knownBug: nil) {
        let probe = UnorderedTaggingProbe()
        try await TaggingBatchPipeline.runPipelined(maximumInFlight: 3, commitInOrder: false,
            next: { await probe.next() }, whenEmpty: { false },
            process: { try await probe.process($0) }, commit: { await probe.commit($0) })
        let committed = await probe.committed
        checkEqual(committed.sorted(), Array(0..<10), "all frames commit exactly once")
        check(committed.first != 0, "faster batches commit ahead of the slow batch")
    })

    Registry.shared.add(suite: suite, TestCase(name: "server settings persist independently and removal leaves primary intact", knownBug: nil) {
        try await Task { @MainActor in
            let suiteName = "waifu.pool.tests.\(UUID().uuidString)"
            guard let defaults = UserDefaults(suiteName: suiteName) else { throw TaggerError.invalidBatch }
            defer { defaults.removePersistentDomain(forName: suiteName) }
            let primary = RemoteClient(defaults: defaults, discoveryEnabled: false)
            primary.manualAddress = "primary:8354"
            let pool = RemoteInferencePool(defaults: defaults, primary: primary, discoveryEnabled: false)
            pool.add()
            let entry = pool.entries[1]
            entry.client.manualAddress = "second:8354"
            let restored = RemoteInferencePool(defaults: defaults, primary: primary, discoveryEnabled: false)
            checkEqual(restored.entries.count, 2, "additional server restores")
            checkEqual(restored.entries[1].client.manualAddress, "second:8354", "second address restores")
            checkEqual(primary.manualAddress, "primary:8354", "primary address is independent")
            pool.remove(entry.id)
            let afterRemoval = RemoteInferencePool(defaults: defaults, primary: primary, discoveryEnabled: false)
            checkEqual(afterRemoval.entries.count, 1, "removed server stays removed")
            pool.remove("primary")
            checkEqual(pool.entries.count, 1, "shared primary slot remains available")
            checkEqual(primary.manualAddress, "", "removing primary clears its server card")
            check(!primary.isEnabled, "removed primary is switched off")
            checkEqual(pool.beginAdding().id, "primary", "next added server reuses the primary slot")
        }.value
    })
}
