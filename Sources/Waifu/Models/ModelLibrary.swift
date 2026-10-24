import AppKit
import Combine
import Foundation

struct ModelLibraryEntry: Identifiable {
    let model: ManagedModel
    var id: String { model.id }
    var location: URL?
    var bytes: Int64 = 0
    var ready = false
    var updateAvailable = false
    var pending: URL?
}

@MainActor
final class ModelLibrary: ObservableObject {
    static let shared = ModelLibrary()
    @Published private(set) var entries = ManagedModel.offered.map { ModelLibraryEntry(model: $0) }
    @Published private(set) var active: ManagedModel?
    @Published private(set) var status = ""
    @Published private(set) var batchStatus = ""
    @Published private(set) var isDownloadingAll = false
    @Published private(set) var fraction: Double?
    @Published var error: String?
    @Published private(set) var inUse: [ManagedModel: Int] = [:]
    @Published private(set) var storage: URL
    private var operation: Task<Void, Never>?
    private var refreshTask: Task<Void, Never>?
    private var generation = UUID()
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        storage = defaults.string(forKey: "modelStoragePath").map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Waifu/Models", isDirectory: true)
    }
    func acquire(_ algorithm: UncoverAlgorithm) {
        if let model = ManagedModel.offered.first(where: { $0.algorithm == algorithm }) { inUse[model, default: 0] += 1 }
    }
    func release(_ algorithm: UncoverAlgorithm) {
        if let model = ManagedModel.offered.first(where: { $0.algorithm == algorithm }) { inUse[model] = max(0, (inUse[model] ?? 0) - 1) }
    }
    func acquireTagger() { inUse[.tagger, default: 0] += 1 }
    func releaseTagger() { inUse[.tagger] = max(0, (inUse[.tagger] ?? 0) - 1) }
    func acquireSafire() { inUse[.safire, default:0] += 1 }
    func releaseSafire() { inUse[.safire] = max(0,(inUse[.safire] ?? 0)-1) }
    func acquireTruFor() { inUse[.trufor,default:0]+=1 }
    func releaseTruFor() { inUse[.trufor]=max(0,(inUse[.trufor] ?? 0)-1) }
    var totalBytes: Int64 { entries.reduce(0) { $0 + $1.bytes } }
    var canDownloadAll: Bool { entries.contains { $0.model.supportsDownload && (!$0.ready || $0.updateAvailable) } }
    func isManaged(_ entry: ModelLibraryEntry) -> Bool {
        guard let location = entry.location,
              let owned = defaults.string(forKey: "ownedModel.\(entry.id)") else { return false }
        let root = URL(fileURLWithPath: owned).resolvingSymlinksInPath()
        let actual = location.resolvingSymlinksInPath()
        return actual == root || (entry.model == .real && actual == root.appendingPathComponent("model.mlmodelc"))
    }
    static func configuredLocation(_ model: ManagedModel) -> URL? {
        let root = SeedVR2Configuration.configurationURL.deletingLastPathComponent()
        switch model {
        case .seed3, .seed7:
            return SeedVR2Configuration.load()?.optimized(variant: model.rawValue).map { URL(fileURLWithPath: $0.model) }
        case .flash: return FlashVSRConfiguration.load().map { URL(fileURLWithPath: $0.model) }
        case .pgt: return PGTFormerConfiguration.load().map { URL(fileURLWithPath: $0.model) }
        case .tagger: return WDTaggerConfiguration.availableDirectory
        case .safire: return SafireConfiguration.availableDirectory
        case .trufor: return TruForConfiguration.availableDirectory
        case .real:
            if let data = try? Data(contentsOf: root.appendingPathComponent("realesrgan-model.txt")),
               let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
                return URL(fileURLWithPath: text)
            }
            if let data = try? Data(contentsOf: SeedVR2Configuration.configurationURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let path = json["spatialUpscaler"] as? String {
                return URL(fileURLWithPath: path)
            }
            return nil
        }
    }
    func refresh() {
        refreshTask?.cancel()
        refreshTask = Task {
            var updated: [ModelLibraryEntry] = []
            for model in ManagedModel.offered {
                let location = Self.configuredLocation(model)
                let ready: Bool
                if model == .tagger {
                    ready = location.map { WDTaggerBackend.isAvailable(directory: $0) } ?? false
                } else if model == .safire {
                    ready = SafireConfiguration.load() != nil
                } else if model == .trufor {
                    ready=TruForConfiguration.load() != nil
                } else if let algorithm = model.algorithm {
                    ready = await algorithm.makeBackend().capabilities().available
                } else { ready = false }
                let stage = storage.appendingPathComponent(".\(model.rawValue)-download")
                let pending = FileManager.default.fileExists(atPath: stage.path) ? stage : nil
                var bytes = if let location { await ModelInstaller.shared.diskBytes(location) } else { Int64(0) }
                if let pending { bytes += await ModelInstaller.shared.diskBytes(pending) }
                guard !Task.isCancelled else { return }
                let updateAvailable: Bool
                if model == .trufor, let location, ready {
                    // A locally converted folder never matches the published
                    // manifest byte-for-byte, so TruFor compares what inference
                    // actually loads: the converted weights hash.
                    updateAvailable = (try? ModelInstaller.hashes(location)["model.safetensors"]) != TruForConfiguration.weightsSHA
                } else if model == .safire, let location, let release=try? ModelInstaller.release(model),
                   let installed=try? ModelInstaller.digest(location.appendingPathComponent("manifest.json")) {
                    updateAvailable = installed != release.manifestSHA256
                } else { updateAvailable = false }
                updated.append(ModelLibraryEntry(model: model, location: location, bytes: bytes, ready: ready, updateAvailable:updateAvailable,pending: pending))
            }
            entries = updated
        }
    }
    func chooseStorage() {
        let panel = NSOpenPanel()
        panel.title = "Choose where new models will be downloaded"
        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.canCreateDirectories = true
        panel.directoryURL = storage
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        storage = folder
        defaults.set(folder.path, forKey: "modelStoragePath")
    }
    func chooseExisting(_ model: ManagedModel) {
        let panel = NSOpenPanel()
        panel.title = "Choose \(model.name) model folder"
        panel.canChooseFiles = false; panel.canChooseDirectories = true
        panel.treatsFilePackagesAsDirectories = true
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        start(model, download: false, directory: folder)
    }
    func download(_ model: ManagedModel) { start(model, download: true, directory: storage) }
    func downloadAll() {
        guard active == nil, !isDownloadingAll else { return }
        error = nil
        batchStatus = "Preparing all models…"
        isDownloadingAll = true
        let token = UUID()
        generation = token
        operation = Task {
            var completed = 0
            var total = 0
            do {
                await refreshTask?.value
                try Task.checkCancellation()
                let pending = entries.filter { $0.model.supportsDownload && (!$0.ready || $0.updateAvailable) }
                total = pending.count
                guard total > 0 else {
                    batchStatus = "All models are already ready"
                    status = "All models are ready"
                    finishOperation(token: token)
                    return
                }
                for (index, entry) in pending.enumerated() {
                    try Task.checkCancellation()
                    let model = entry.model
                    guard (inUse[model] ?? 0) == 0 else {
                        throw ModelInstaller.fail("\(model.name) is currently in use and cannot be set up.")
                    }
                    batchStatus = "Downloading and configuring all models · \(index + 1) of \(total) · \(model.name)"
                    let directory = entry.updateAvailable ? storage : entry.location ?? storage
                    try await performSetup(model, download: entry.location == nil || entry.updateAvailable, directory: directory, token: token)
                    completed += 1
                    active = nil
                    fraction = nil
                }
                batchStatus = "All models are downloaded and configured"
                status = "All models are ready"
            } catch {
                if Task.isCancelled {
                    batchStatus = completed == 0 ? "Stopped" : "Stopped · \(completed) of \(total) models completed"
                    status = "Cancelled. Downloaded files are kept for retry."
                } else {
                    self.error = error.localizedDescription
                    batchStatus = "Setup stopped · \(completed) of \(total) models completed"
                    status = "Setup failed"
                }
            }
            finishOperation(token: token)
        }
    }
    private func start(_ model: ManagedModel, download: Bool, directory: URL) {
        guard active == nil, !isDownloadingAll, (inUse[model] ?? 0) == 0 else { return }
        active = model; error = nil; batchStatus = ""; isDownloadingAll = false; status = "Preparing \(model.name)"; fraction = nil
        let token = UUID(); generation = token
        operation = Task {
            do {
                try await performSetup(model, download: download, directory: directory, token: token)
            } catch {
                if Task.isCancelled { status = "Cancelled. Downloaded files are kept for retry." }
                else { self.error = error.localizedDescription; status = "Setup failed" }
            }
            finishOperation(token: token)
        }
    }
    private func performSetup(_ model: ManagedModel, download: Bool, directory: URL, token: UUID) async throws {
        active = model
        error = nil
        status = "Preparing \(model.name)"
        fraction = nil
        let callback: ModelInstaller.Progress = { [weak self] message, value in
            Task { @MainActor in
                guard let self, self.generation == token, self.active == model else { return }
                self.status = message; self.fraction = value
            }
        }
        if download {
            let installed = try await ModelInstaller.shared.install(model, root: directory, progress: callback)
            defaults.set(installed.path, forKey: "ownedModel.\(model.id)")
        } else {
            try await ModelInstaller.shared.configure(model, directory: directory, progress: callback)
        }
        try Task.checkCancellation()
        status = "\(model.name) is ready"
        NotificationCenter.default.post(name: .waifuModelsChanged, object: nil)
    }
    private func finishOperation(token: UUID) {
        guard generation == token else { return }
        active = nil
        fraction = nil
        operation = nil
        isDownloadingAll = false
        refresh()
    }
    func cancel() {
        if isDownloadingAll {
            batchStatus = "Stopping…"
            refreshTask?.cancel()
        }
        operation?.cancel()
    }
    func remove(_ entry: ModelLibraryEntry) {
        guard active == nil, (inUse[entry.model] ?? 0) == 0 else { return }
        let owned = isManaged(entry)
        let alert = NSAlert()
        alert.messageText = owned ? "Remove \(entry.model.name)?" : "Disconnect \(entry.model.name)?"
        alert.informativeText = owned ? "The model folder will move to Trash. You can download it again. Saved exports are kept." : "The app will stop using this model. Its files will stay at their current location."
        alert.addButton(withTitle: owned ? "Move to Trash" : "Disconnect")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        do {
            if owned, let path = defaults.string(forKey: "ownedModel.\(entry.id)"), FileManager.default.fileExists(atPath: path) {
                try FileManager.default.trashItem(at: URL(fileURLWithPath: path), resultingItemURL: nil)
            }
            try disconnect(entry.model)
            defaults.removeObject(forKey: "ownedModel.\(entry.id)")
            NotificationCenter.default.post(name: .waifuModelsChanged, object: nil)
            refresh()
        } catch { self.error = error.localizedDescription }
    }
    func discardDownload(_ entry: ModelLibraryEntry) {
        guard active == nil, let pending = entry.pending else { return }
        do {
            try FileManager.default.trashItem(at: pending, resultingItemURL: nil)
            refresh()
        } catch { self.error = error.localizedDescription }
    }
    private func disconnect(_ model: ManagedModel) throws {
        let root = SeedVR2Configuration.configurationURL.deletingLastPathComponent()
        func removeFile(_ name: String) throws {
            let url = root.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
        }
        switch model {
        case .tagger: try removeFile("wd-tagger-model.txt")
        case .safire: try removeFile("safire-model.txt")
        case .trufor: try removeFile("trufor-model.txt")
        case .flash: try removeFile("flashvsr.json")
        case .pgt: try removeFile("pgtformer.json")
        case .real:
            try removeFile("realesrgan-model.txt")
            // Also clear the legacy fallback so disconnect cannot silently reactivate it.
            try clearLegacySpatial(configURL: SeedVR2Configuration.configurationURL)
        case .seed3, .seed7:
            guard var config = SeedVR2Configuration.load() else { return }
            config.optimizedModels?.removeValue(forKey: model.rawValue)
            if config.modelVariant == model.rawValue || config.optimized(variant: model.rawValue)?.model == config.model {
                let other = model == .seed3 ? "seedvr2-7b" : "seedvr2-3b"
                config.model = config.optimizedModels?[other] ?? ""
                config.modelVariant = config.model.isEmpty ? nil : other
                config.nativeEmbedding = config.model.isEmpty ? nil : config.model + "/pos_emb.safetensors"
            }
            // Block sibling auto-discovery for an explicitly disconnected variant.
            var disabled = config.disconnectedVariants ?? []
            if !disabled.contains(model.rawValue) { disabled.append(model.rawValue) }
            config.disconnectedVariants = disabled
            try JSONEncoder().encode(config).write(to: SeedVR2Configuration.configurationURL, options: .atomic)
        }
    }
    private func clearLegacySpatial(configURL: URL) throws {
        guard let data = try? Data(contentsOf: configURL), var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        json.removeValue(forKey: "spatialUpscaler")
        try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]).write(to: configURL, options: .atomic)
    }
}
