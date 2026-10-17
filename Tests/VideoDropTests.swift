import Foundation

private func makeDropFixture(_ name: String) throws -> URL {
    let directory = try Temporary.directory("drop")
    let url = directory.appendingPathComponent(name)
    guard FileManager.default.createFile(atPath: url.path, contents: Data()) else {
        throw SkipTest(reason: "could not create a fixture at \(url.path)")
    }
    return url
}

func registerVideoDropTests() {
    let suite = "video drop"

    Registry.shared.add(suite: suite, TestCase(
        name: "a video file with a video extension is accepted", knownBug: nil) {
        for name in ["clip.mp4", "clip.mov", "clip.M4V", "clip.mpeg"] {
            let url = try makeDropFixture(name)
            check(VideoDrop.accepts(url), "\(name) should be accepted")
        }
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "regular files are candidates regardless of extension", knownBug: nil) {
        for name in ["notes.txt", "photo.png", "song.mp3", "archive.zip"] {
            let url = try makeDropFixture(name)
            check(VideoDrop.accepts(url), "\(name) should be inspected")
        }
        let bare = try makeDropFixture("README")
        check(VideoDrop.accepts(bare), "extensionless media should be inspected")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "an empty extension is still a candidate", knownBug: nil) {
        let url = try makeDropFixture("clip.")
        check(VideoDrop.accepts(url), "contents determine the format")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a directory is refused even when it is named like a video", knownBug: nil) {
        let directory = try Temporary.directory("drop")
        let folder = directory.appendingPathComponent("holiday.mp4", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        check(!VideoDrop.accepts(folder), "a directory named like a clip must be refused")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "a path that does not exist is refused", knownBug: nil) {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("waifu-does-not-exist-\(UUID().uuidString).mp4")
        check(!VideoDrop.accepts(missing), "a vanished source is refused deterministically")
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "web URLs are candidates and unsupported schemes are refused", knownBug: nil) {
        check(VideoDrop.accepts(URL(string: "https://example.com/download")!), "HTTP links are candidates")
        for string in ["ftp://example.com/clip.mov"] {
            guard let url = URL(string: string) else {
                check(false, "\(string) should parse as a URL")
                continue
            }
            check(!VideoDrop.accepts(url), "\(string) is not a local file and should be refused")
        }
    })

    Registry.shared.add(suite: suite, TestCase(
        name: "firstAcceptableVideo takes the first acceptable URL, ignoring the rest", knownBug: nil) {
        let good = try makeDropFixture("good.mp4")
        let bad = URL(string: "ftp://example.com/notes.txt")!

        checkEqual(VideoDrop.firstAcceptableVideo(in: [good, bad]), good,
                   "the first acceptable URL wins")
        checkEqual(VideoDrop.firstAcceptableVideo(in: [bad, good]), good,
                   "a leading refusal does not stop the search")
        checkNil(VideoDrop.firstAcceptableVideo(in: [bad]),
                 "no acceptable URL yields nil, not a rejection of the last candidate")
        checkNil(VideoDrop.firstAcceptableVideo(in: []), "an empty drop yields nil")
    })
}
