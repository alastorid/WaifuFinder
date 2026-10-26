import Foundation

private func waitForCloudTest(_ predicate: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(3)
    while !predicate(), Date() < deadline { try await Task.sleep(for: .milliseconds(5)) }
    guard predicate() else { throw PhotosMedia.failure("Cloud queue did not reach the expected state") }
}

func registerPhotosCloudQueueTests() {
    let suite = "Photos cloud queue"
    Registry.shared.add(suite: suite, TestCase(name: "cloud-enabled tagging retains four concurrent local PhotoKit reads", knownBug: nil) {
        let image = try ImageFixture.cgImage(width:448,height:448)
        let events = TickLog()
        let sources = (0..<8).map { TaggingImageSource.photo(URL(string:"waifu-photos://local/\($0)")!) }
        let images = try await TaggingImageLoader.load(sources, photoRendition: { _, _ in
            try await PhotosRenditionLoader.load(allowNetwork:true,requestedSize:CGSize(width:448,height:448),
                local:{
                    events.record(1)
                    try await Task.sleep(for:.milliseconds(20))
                    events.record(2)
                    return image
                },cloud:{
                    check(false,"available local pixels must never queue a cloud request")
                    return image
                })
        })
        checkEqual(images.count,8,"all local photos reach the batch")
        checkEqual(Array(events.all.prefix(4)),[1,1,1,1],"local reads overlap even when cloud downloads are enabled")
    })
    Registry.shared.add(suite: suite, TestCase(name: "only missing or undersized local renditions request a cloud upgrade", knownBug: nil) {
        let sharp = try ImageFixture.cgImage(width:448,height:448)
        let tiny = try ImageFixture.cgImage(width:64,height:64)
        let events = TickLog()
        for missing in [false,true] {
            let result = try await PhotosRenditionLoader.load(allowNetwork:true,requestedSize:CGSize(width:448,height:448),
                local:{ if missing { throw PhotosMedia.localUnavailable() };return tiny },
                cloud:{ events.record(1);return sharp })
            check(result === sharp,"cloud upgrade supplies the requested pixels")
        }
        let offline = try await PhotosRenditionLoader.load(allowNetwork:false,requestedSize:CGSize(width:448,height:448),
            local:{tiny},cloud:{check(false,"disabled cloud policy must never queue an upgrade");return sharp})
        check(offline === tiny,"offline mode retains its available pixels")
        do {
            _ = try await PhotosRenditionLoader.load(allowNetwork:true,requestedSize:CGSize(width:448,height:448),
                local:{throw CancellationError()},cloud:{check(false,"cancellation must not start a cloud upgrade");return sharp})
            check(false,"canceled read must stop")
        } catch is CancellationError { }
        checkEqual(events.all,[1,1],"exactly the two upgrades were requested")
    })
    Registry.shared.add(suite: suite, TestCase(name: "viewing preempts background work and resumes it before idle preload", knownBug: nil) {
        let queue = PhotosCloudQueue(), events = TickLog()
        let background = Task {
            try await queue.perform(priority: .background) {
                events.record(1)
                if events.all.filter({ $0 == 1 }).count == 1 {
                    do { try await Task.sleep(for: .seconds(30)) }
                    catch { events.record(-1); throw error }
                }
                events.record(2)
                return 42
            }
        }
        defer { background.cancel() }
        try await waitForCloudTest { events.all.contains(1) }
        let preload = Task {
            try await queue.perform(priority: .preload) { events.record(4); return 7 }
        }
        defer { preload.cancel() }
        let foreground = try await queue.perform(priority: .viewing) { events.record(3); return 99 }
        checkEqual(foreground,99,"viewed photo completes first")
        checkEqual(try await background.value,42,"interrupted album request is resumed, not failed")
        checkEqual(try await preload.value,7,"speculation runs once the queue is idle")
        checkEqual(events.all, [1,-1,3,1,2,4], "only the viewed photo preempts background; preload waits")
    })
    Registry.shared.add(suite: suite, TestCase(name: "navigation cancels the old preload and gives the new photo priority", knownBug: nil) {
        let queue = PhotosCloudQueue(), events = TickLog()
        let preload = Task {
            try await queue.perform(priority: .preload) {
                events.record(1)
                do { try await Task.sleep(for: .seconds(30)) }
                catch { events.record(-1); throw error }
                return 1
            }
        }
        defer { preload.cancel() }
        try await waitForCloudTest { events.all.contains(1) }
        preload.cancel()
        let viewed = try await queue.perform(priority: .viewing) { events.record(2); return 2 }
        if case .success = await preload.result { check(false,"old preload must cancel") }
        checkEqual(viewed,2,"new photo obtains the cloud lane")
        checkEqual(events.all,[1,-1,2],"no old preload restarts after cancellation")
    })
    Registry.shared.add(suite: suite, TestCase(name: "a viewed photo directly interrupts an in-flight speculative preload", knownBug: nil) {
        let queue = PhotosCloudQueue(), events = TickLog()
        let preload = Task {
            try await queue.perform(priority: .preload) {
                events.record(1)
                if events.all.filter({ $0 == 1 }).count == 1 {
                    do { try await Task.sleep(for: .seconds(30)) }
                    catch { events.record(-1); throw error }
                }
                return 1
            }
        }
        defer { preload.cancel() }
        try await waitForCloudTest { events.all.contains(1) }
        _ = try await queue.perform(priority: .viewing) { events.record(2); return 2 }
        checkEqual(try await preload.value,1,"preempted preload can finish later if its owner still wants it")
        checkEqual(events.all,[1,-1,2,1],"foreground runs ahead of speculative work")
    })
    Registry.shared.add(suite: suite, TestCase(name: "preload order is next, previous, then forward through seven without wrapping", knownBug: nil) {
        checkEqual(PhotosPreviewNeighbours.indices(current:3,count:20),[4,2,5,6,7,8,9,10],"next and previous lead the bounded forward window")
        checkEqual(PhotosPreviewNeighbours.indices(current:0,count:3),[1,2],"start does not wrap backwards")
        checkEqual(PhotosPreviewNeighbours.indices(current:2,count:3),[1],"end does not wrap forwards")
        checkEqual(PhotosPreviewNeighbours.indices(current:0,count:1),[],"single photo has no speculative downloads")
        checkEqual(PhotosPreviewNeighbours.indices(current:0,count:0),[],"empty album has no speculative downloads")
    })
}
