import Foundation

func registerPanelPresetTests() {
    let suite = "Panel presets"
    Registry.shared.add(suite: suite, TestCase(name: "all sidebar preferences survive save, restart and load", knownBug: nil) {
        let directory = try Temporary.directory("panel-presets-roundtrip")
        await MainActor.run {
            let url = directory.appendingPathComponent("settings.json")
            let session = WaifuSession(settingsURL: url)
            session.frameOrder = FrameOrder(minimalVisualDistance: .off(ascending: false),
                                           time: .chronological, art: .best, ratingExplicit: .best)
            session.frameLimits.time = FrameLimit(low: 12, high: 84)
            session.frameLimits.art = FrameLimit(low: 0.3, high: nil)
            session.frameLimits.ratingExplicit = FrameLimit(low: nil, high: 0.7)
            session.tagFilter = TagFilter(requiredTags: ["smile", "blue_hair"], excludedTags: ["hat"],
                                          threshold: 0.62, selectionOrder: ["smile", "blue_hair"],
                                          scoreSortTag: "smile", scoreSortDescending: false)
            session.faceFilter = .withFaces
            session.favoriteFilter = .notFavorite
            session.groupingEnabled = true
            session.groupingThreshold = 0.58
            session.groupRows = false
            session.settings.perGroupFrameCount = 13
            session.settings.revealAllGroups = true
            session.settings.tagAdvancedMode = true
            session.settings.tagTopCount = 19
            session.settings.taggerThreshold = 0.65
            session.settings.exportJPEGQuality = 0.8
            guard session.savePanelPreset(at: 0) else {
                check(false, "save succeeds"); return
            }
            let expected = session.panelPresets[0]
            let id = expected.id
            let reopened = WaifuSession(settingsURL: url)
            checkEqual(reopened.panelPresets, session.panelPresets, "all three slots round trip on disk")
            checkEqual(reopened.settings.revealAllGroups, true, "reveal mode persists across restart")
            checkEqual(reopened.settings.tagAdvancedMode, true, "advanced mode persists across restart")
            checkEqual(reopened.settings.tagTopCount, 19, "tag top count persists")
            checkEqual(reopened.settings.taggerThreshold, 0.65, "dynamic threshold persists")
            reopened.settings.tagAdvancedMode = false
            reopened.settings.tagTopCount = 99
            reopened.settings.taggerThreshold = 0.35
            reopened.settings.revealAllGroups = false
            reopened.settings.exportJPEGQuality = 0.5
            reopened.loadPanelPreset(id)
            checkEqual(reopened.frameOrder, expected.order, "sort signals and directions restored")
            checkEqual(reopened.frameLimits, expected.limits, "numeric ranges restored")
            checkEqual(reopened.tagFilter, expected.tags, "tag sets, order, threshold and score sorting restored")
            checkEqual(reopened.faceFilter, expected.faces, "face filter restored")
            checkEqual(reopened.favoriteFilter, .notFavorite, "favorite filter restored")
            checkEqual(reopened.groupingEnabled, true, "grouping restored")
            checkEqual(reopened.groupingThreshold, 0.58, "grouping threshold restored")
            checkEqual(reopened.settings.similarityThreshold, 0.58, "threshold persists with no media loaded")
            checkEqual(reopened.groupRows, false, "row preference restored")
            checkEqual(reopened.settings.perGroupFrameCount, 13, "cap restored")
            checkEqual(reopened.settings.revealAllGroups, true, "reveal mode restored from preset")
            checkEqual(reopened.settings.tagAdvancedMode, true, "preset restores advanced mode")
            checkEqual(reopened.settings.tagTopCount, 19, "preset restores tag top count")
            checkEqual(reopened.settings.taggerThreshold, 0.65, "preset restores dynamic threshold")
            checkEqual(reopened.settings.exportJPEGQuality, 0.5, "unrelated preferences retained")
            checkEqual(reopened.regroupPending, false, "no stale pending indicator")
            reopened.clearPanelSettings()
            checkEqual(reopened.settings.revealAllGroups, false, "clear returns to limited mode")
            checkEqual(reopened.favoriteFilter, .all, "clear resets favorite filter")
            checkEqual(reopened.settings.tagAdvancedMode, true, "clear returns to Advanced")
            checkEqual(reopened.settings.tagTopCount, 99, "clear resets Top to all")
            checkEqual(reopened.settings.taggerThreshold, 0.35, "clear resets threshold")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "three slots start from app defaults and memory replaces only the chosen slot", knownBug: nil) {
        let directory = try Temporary.directory("panel-presets-slots")
        try await MainActor.run {
            let url = directory.appendingPathComponent("settings.json")
            var custom = WaifuSettings.default
            custom.perGroupFrameCount = 99
            custom.similarityThreshold = 0.9
            try JSONEncoder().encode(custom).write(to: url)
            let session = WaifuSession(settingsURL: url)
            checkEqual(session.panelPresets.count, 3, "exactly three slots always exist")
            let initial = session.panelPresets
            for preset in initial {
                checkEqual(preset.order, .defaultOrder, "default ordering")
                checkEqual(preset.limits, .none, "no ranges")
                checkEqual(preset.tags, TagFilter(), "default tags")
                checkEqual(preset.faces, .all, "all faces")
                checkEqual(preset.groupingEnabled, false, "grouping off")
                checkEqual(preset.groupingThreshold, WaifuSettings.default.similarityThreshold, "app default threshold")
                checkEqual(preset.groupRows, true, "default rows")
                checkEqual(preset.perGroupCount, WaifuSettings.default.perGroupFrameCount, "app default cap, not saved preferences")
            }
            session.settings.perGroupFrameCount = 23
            check(session.savePanelPreset(at: 1), "memory saves slot 2")
            checkEqual(session.panelPresets.count, 3, "memory never appends")
            checkEqual(session.panelPresets[0], initial[0], "slot 1 retained")
            checkEqual(session.panelPresets[2], initial[2], "slot 3 retained")
            checkEqual(session.panelPresets[1].id, initial[1].id, "slot identity retained")
            session.settings.perGroupFrameCount = 27
            check(session.savePanelPreset(at: 1), "memory overwrites slot 2 again")
            let reopened = WaifuSession(settingsURL: url)
            reopened.loadPanelPreset(initial[1].id)
            checkEqual(reopened.settings.perGroupFrameCount, 27, "latest overwrite survives restart")
            reopened.settings.perGroupFrameCount = 25
            reopened.loadPanelPreset(initial[1].id)
            checkEqual(reopened.settings.perGroupFrameCount, 27, "reselecting the same number recalls it again")
            reopened.loadPanelPreset(initial[0].id)
            checkEqual(reopened.settings.perGroupFrameCount, WaifuSettings.default.perGroupFrameCount, "untouched slots load defaults")
            check(!session.savePanelPreset(at: 3), "out-of-range slot rejected")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "clear preserves recalled memory until Memory is pressed", knownBug: nil) {
        let directory = try Temporary.directory("panel-presets-clear")
        try await MainActor.run {
            let url = directory.appendingPathComponent("settings.json")
            let session = WaifuSession(settingsURL: url)
            session.settings.perGroupFrameCount = 23
            session.tagFilter = TagFilter(requiredTags: ["smile"])
            check(session.savePanelPreset(at: 1), "save customized memory")
            let saved = session.panelPresets
            let file = directory.appendingPathComponent("settings.presets.json")
            let savedData = try Data(contentsOf: file)
            session.loadPanelPreset(saved[1].id)
            session.clearPanelSettings()
            checkEqual(session.settings.perGroupFrameCount, WaifuSettings.default.perGroupFrameCount, "clear resets live cap")
            checkEqual(session.tagFilter, TagFilter(), "clear resets live tags")
            checkEqual(session.panelPresets, saved, "clear preserves all memories")
            checkEqual(try Data(contentsOf: file), savedData, "clear leaves saved memory file unchanged")
            let reopened = WaifuSession(settingsURL: url)
            reopened.loadPanelPreset(saved[1].id)
            checkEqual(reopened.settings.perGroupFrameCount, 23, "memory survives clear and restart")
            checkEqual(reopened.tagFilter, saved[1].tags, "saved tags can be recalled")
            check(session.savePanelPreset(at: 1), "Memory explicitly saves cleared preferences")
            let updated = WaifuSession(settingsURL: url)
            updated.loadPanelPreset(saved[1].id)
            checkEqual(updated.tagFilter, TagFilter(), "Memory persists cleared tags")
            checkEqual(updated.panelPresets[0], saved[0], "other slot retained")
            checkEqual(updated.panelPresets[2], saved[2], "other slot retained")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "favorite choices round trip and older presets default to all", knownBug: nil) {
        let directory = try Temporary.directory("panel-presets-favorites")
        try await MainActor.run {
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            for choice in FrameFavoriteFilter.allCases {
                session.favoriteFilter = choice
                check(session.savePanelPreset(at: 0), "save favorite choice")
                let data = try JSONEncoder().encode(session.panelPresets[0])
                let preset = try JSONDecoder().decode(PanelPreset.self, from: data)
                checkEqual(preset.favorites, choice, "choice survives encoding")
                session.favoriteFilter = .all
                session.loadPanelPreset(preset.id)
                checkEqual(session.favoriteFilter, choice, "choice recalled")
            }
            var legacy = try JSONSerialization.jsonObject(with: JSONEncoder().encode(session.panelPresets)) as! [[String: Any]]
            legacy[0].removeValue(forKey: "favorites")
            try JSONSerialization.data(withJSONObject: legacy).write(to: directory.appendingPathComponent("settings.presets.json"))
            let reopened = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            reopened.favoriteFilter = .favorite
            reopened.loadPanelPreset(session.panelPresets[0].id)
            checkEqual(reopened.favoriteFilter, .all, "older preset defaults to all favorites")
        }
    })
    Registry.shared.add(suite: suite, TestCase(name: "failed overwrite preserves the slot and reports an error", knownBug: nil) {
        let directory = try Temporary.directory("panel-presets-write-error")
        try await MainActor.run {
            let session = WaifuSession(settingsURL: directory.appendingPathComponent("settings.json"))
            check(session.savePanelPreset(at: 0), "initial save succeeds")
            let before = session.panelPresets
            let file = directory.appendingPathComponent("settings.presets.json")
            try FileManager.default.removeItem(at: file)
            try FileManager.default.createDirectory(at: file, withIntermediateDirectories: false)
            session.settings.perGroupFrameCount = 44
            check(!session.savePanelPreset(at: 0), "failed memory does not claim success")
            checkEqual(session.panelPresets, before, "all saved slots retained")
            checkNotNil(session.errorMessage, "storage error surfaced")
        }
    })
}
