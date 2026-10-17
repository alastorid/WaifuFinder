import Foundation
import AVFoundation

/// Real local HTTP server: exercises URLSession, redirects, MIME-independent probing,
/// playlist resolution and native AVFoundation media inspection.
private final class ImportHTTPServer {
    let process = Process()
    let base: URL
    init(directory: URL) throws {
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-u", "-c", """
        import http.server, functools, time
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/slow':
                    time.sleep(10)
                if self.path == '/redirect':
                    self.send_response(302)
                    self.send_header('Location', '/master')
                    self.end_headers()
                else: super().do_GET()
            def log_message(self, *args): pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=\(String(reflecting: directory.path))))
        print(server.server_port, flush=True)
        server.serve_forever()
        """]
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        var line = Data()
        while let byte = try pipe.fileHandleForReading.read(upToCount: 1), !byte.isEmpty, byte != Data([10]) { line.append(byte) }
        guard let port = Int(String(decoding: line, as: UTF8.self)) else {
            process.terminate()
            throw SkipTest(reason: "local HTTP fixture could not start")
        }
        base = URL(string: "http://127.0.0.1:\(port)")!
    }
    deinit { if process.isRunning { process.terminate() } }
}

func registerMediaImportTests() {
    let suite = "media import"
    Registry.shared.add(suite: suite, TestCase(name: "HTTP URL validation and protocol URL round trip", knownBug: nil) {
        for value in ["https://example.com/download?id=7", "http://localhost/clip.dms"] {
            let url = MediaImport.webURL(value)!
            checkEqual(FirefoxIntegration.incomingURL(FirefoxIntegration.wrappedURL(url)), url, "URL survives scheme handoff")
        }
        for value in ["ftp://example.com/x", "file:///tmp/x", "javascript:alert(1)", "https://", "https://user:password@example.com/x"] {
            checkNil(MediaImport.webURL(value), "unsafe or incomplete URL is rejected")
        }
        checkNil(FirefoxIntegration.incomingURL(URL(string: "waifufinder://open?url=file:///etc/passwd")!), "bridge cannot load local paths")
        checkEqual(FirefoxIntegration.suffixes(".MP4, ts .dms,mp4, ../bad"), ["dms", "mp4", "ts"], "normalizes suffixes")
    })
    Registry.shared.add(suite: suite, TestCase(name: "add-on loader keeps its URL fragment by running Firefox's own launcher", knownBug: nil) {
        // `/usr/bin/open` percent-encodes the `#` of a scheme-only URL, which used to
        // leave Firefox on `about:debugging%23/runtime/this-firefox`.
        checkEqual(FirefoxIntegration.addOnLoaderURL, "about:debugging#/runtime/this-firefox", "loader URL keeps a literal fragment")
        check(!FirefoxIntegration.addOnLoaderURL.contains("%23"), "loader URL is never percent-encoded")
        checkEqual(FirefoxIntegration.addOnLoaderArguments, ["-new-tab", "about:debugging#/runtime/this-firefox"],
                   "Firefox receives the URL on its own command line, not through `open`")

        let bundle = try Temporary.directory("firefox-bundle").appendingPathComponent("Firefox.app")
        let binDirectory = bundle.appendingPathComponent("Contents/MacOS")
        try FileManager.default.createDirectory(at: binDirectory, withIntermediateDirectories: true)
        checkNil(FirefoxIntegration.launcher(in: bundle), "a bundle with no launcher reports none")

        try FileManager.default.createDirectory(at: binDirectory.appendingPathComponent("firefox-shim"), withIntermediateDirectories: true)
        checkNil(FirefoxIntegration.launcher(in: bundle), "a directory named like the launcher is not runnable")

        let launcher = binDirectory.appendingPathComponent("firefox")
        try "#!/bin/sh\nexit 0\n".write(to: launcher, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: launcher.path)
        checkEqual(FirefoxIntegration.launcher(in: bundle), launcher, "launcher inside the bundle is found")
    })
    Registry.shared.add(suite: suite, TestCase(name: "extensionless and misleading HTTP media imports use native decoding", knownBug: nil) {
        let dir = try Temporary.directory("http-import")
        let clip = try VideoFixture.makeMP4(in: dir, durationSeconds: 1, width: 64, height: 48, frameRate: 5)
        for name in ["download", "clip.dms"] { try FileManager.default.copyItem(at: clip, to: dir.appendingPathComponent(name)) }
        let server = try ImportHTTPServer(directory: dir)
        for name in ["download?token=abc", "clip.dms"] {
            var imported: PreparedMedia? = try await MediaImport.prepare(URL(string: name, relativeTo: server.base)!.absoluteURL)
            let path = imported!.url
            let info = try await VideoProbe.probe(path)
            checkEqual(info.width, 64, "downloaded content is a video regardless of name or MIME")
            imported = nil
            check(!FileManager.default.fileExists(atPath: path.path), "release deletes imported data")
        }
        _ = await checkThrowsAsync("HTTP errors fail visibly") {
            try await MediaImport.prepare(server.base.appendingPathComponent("missing"))
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "redirected extensionless HLS master is materialized for native analysis", knownBug: nil) {
        let dir = try Temporary.directory("hls-import")
        let clip = try VideoFixture.makeMP4(in: dir, durationSeconds: 1, width: 64, height: 48, frameRate: 5)
        try await VideoFixture.makeHLS(from: clip, in: dir)
        try Data("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=150000\nplaylist\n".utf8)
            .write(to: dir.appendingPathComponent("master"))
        let server = try ImportHTTPServer(directory: dir)
        let imported = try await MediaImport.prepare(server.base.appendingPathComponent("redirect"))
        let info = try await VideoProbe.probe(imported.url)
        checkEqual(info.width, 64, "native AVFoundation reads the downloaded HLS transport stream")
        check(info.durationSeconds > 0, "the VOD stream is finite")
        var samples = AVFoundationFrameSampler().sample(imported.url, analysisWidth: 64,
                                                         mode: .quick, onProgress: { _ in }).makeAsyncIterator()
        let first = try await samples.next()
        check(first != nil, "native frame sampling processes the HLS transport stream")
    })
    Registry.shared.add(suite: suite, TestCase(name: "live HLS playlists fail with a clear limitation", knownBug: nil) {
        let dir = try Temporary.directory("live-hls")
        try Data("#EXTM3U\n#EXTINF:4.0,\nsegment.ts\n".utf8)
            .write(to: dir.appendingPathComponent("live"))
        let server = try ImportHTTPServer(directory: dir)
        let error = await checkThrowsAsync("live streams are rejected before analysis") {
            try await MediaImport.prepare(server.base.appendingPathComponent("live"))
        }
        check(error?.localizedDescription.contains("Live HLS streams") == true,
              "the unsupported live-stream case is explained")
    })
    Registry.shared.add(suite: suite, TestCase(name: "invalid media and cancelled import fail", knownBug: nil) {
        let dir = try Temporary.directory("bad-import")
        try Data("not video".utf8).write(to: dir.appendingPathComponent("bad.mp4"))
        let server = try ImportHTTPServer(directory: dir)
        _ = await checkThrowsAsync("non-media is not accepted") { try await MediaImport.prepare(server.base.appendingPathComponent("bad.mp4")) }
        let task = Task { try await MediaImport.prepare(server.base.appendingPathComponent("bad.mp4")) }
        task.cancel()
        _ = await checkThrowsAsync("cancelled import throws") { try await task.value }
    })
    Registry.shared.add(suite: suite, TestCase(name: "standalone transport stream imports without a suffix", knownBug: nil) {
        let dir = try Temporary.directory("library-codec")
        let clip = try VideoFixture.makeMP4(in: dir, durationSeconds: 1, width: 64, height: 48, frameRate: 5)
        let input = dir.appendingPathComponent("source.dms")
        try await VideoFixture.makeTransportStream(from: clip, to: input)
        let imported = try await MediaImport.prepare(input)
        let info = try await VideoProbe.probe(imported.url)
        checkEqual(info.width, 64, "native AVFoundation reads transport stream bytes regardless of suffix")
        var frames = AVFoundationFrameSampler().sample(imported.url, analysisWidth: 64,
                                                       mode: .quick, onProgress: { _ in }).makeAsyncIterator()
        check(try await frames.next() != nil, "native sampler decodes a transport stream")
    })
    Registry.shared.add(suite: suite, TestCase(name: "cancelled remote download stops promptly", knownBug: nil) {
        let dir = try Temporary.directory("library-cancel")
        let server = try ImportHTTPServer(directory: dir)
        let url = server.base.appendingPathComponent("slow")
        let task = Task {
            try await MediaImport.prepare(url)
        }
        try await Task.sleep(for: .milliseconds(150))
        let start = Date()
        task.cancel()
        _ = await checkThrowsAsync("blocked import cancels") { try await task.value }
        check(Date().timeIntervalSince(start) < 3, "cancellation stops the download without awaiting the timeout")
    })

}
