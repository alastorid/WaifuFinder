import Foundation

func registerDefaultExportFolderTests() {
    let suite = "default export folder"
    Registry.shared.add(suite: suite, TestCase(name: "selected folder persists, reloads and clears", knownBug: nil) {
        let directory = try Temporary.directory("default-export")
        try await Task { @MainActor in
            let name = "WaifuFinder-Export-Tests-\(UUID().uuidString)"
            let defaults = UserDefaults(suiteName: name)!
            defer { defaults.removePersistentDomain(forName: name) }
            let folder = DefaultExportFolder(defaults: defaults)
            checkNil(folder.url, "no default until a folder is selected")
            try folder.set(directory)
            checkEqual(folder.url?.path, directory.path, "selected folder is published")
            check(defaults.data(forKey: "defaultExportFolderBookmark") != nil, "ordinary bookmark is saved")
            let reloaded = DefaultExportFolder(defaults: defaults)
            checkEqual(reloaded.url?.resolvingSymlinksInPath(), directory.resolvingSymlinksInPath(), "folder survives a restart")
            reloaded.clear()
            checkNil(DefaultExportFolder(defaults: defaults).url, "clear removes bookmark and path")
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "bookmark failure still saves the selected folder and replaces the old destination", knownBug: nil) {
        let oldDirectory = try Temporary.directory("old-export")
        let directory = try Temporary.directory("Downloads")
        try await Task { @MainActor in
            let name = "WaifuFinder-Export-Tests-\(UUID().uuidString)"
            let defaults = UserDefaults(suiteName: name)!
            defer { defaults.removePersistentDomain(forName: name) }
            try DefaultExportFolder(defaults: defaults).set(oldDirectory)
            let folder = DefaultExportFolder(defaults: defaults, makeBookmark: { _ in
                throw CocoaError(.fileReadUnknown)
            })
            try folder.set(directory)
            checkEqual(folder.url?.path, directory.path, "bookmark errors do not reject the selection")
            checkNil(defaults.data(forKey: "defaultExportFolderBookmark"), "old bookmark cannot override the new path")
            let reloaded = DefaultExportFolder(defaults: defaults)
            checkEqual(reloaded.url?.path, directory.path, "path fallback survives a restart")
            guard let destination = reloaded.url else {
                check(false, "saved destination is available")
                return
            }
            let output = destination.appendingPathComponent("export.txt")
            try Data("export".utf8).write(to: output)
            checkEqual(try Data(contentsOf: output), Data("export".utf8), "saved folder remains usable for output")
        }.value
    })
    Registry.shared.add(suite: suite, TestCase(name: "invalid destinations preserve the previous folder", knownBug: nil) {
        let directory = try Temporary.directory("default-export-invalid")
        let file = directory.appendingPathComponent("file.txt")
        try Data().write(to: file)
        try await Task { @MainActor in
            let name = "WaifuFinder-Export-Tests-\(UUID().uuidString)"
            let defaults = UserDefaults(suiteName: name)!
            defer { defaults.removePersistentDomain(forName: name) }
            let folder = DefaultExportFolder(defaults: defaults)
            try folder.set(directory)
            for invalid in [file, directory.appendingPathComponent("missing")] {
                do {
                    try folder.set(invalid)
                    check(false, "files and missing directories are rejected")
                } catch { }
            }
            checkEqual(folder.url?.path, directory.path, "failed changes preserve the current destination")
            checkEqual(DefaultExportFolder(defaults: defaults).url?.resolvingSymlinksInPath(),
                       directory.resolvingSymlinksInPath(), "failed changes preserve the saved destination")
        }.value
    })
}
