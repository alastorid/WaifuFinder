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
            session.groupingEnabled = true
            session.groupingThreshold = 0.58
            session.groupRows = false
            session.settings.perGroupFrameCount = 13
            session.settings.exportJPEGQuality = 0.8
            guard session.savePanelPreset(at: 0) else {
                check(false, "save succeeds"); return
            }
            let expected = session.panelPresets[0]
            let id = expected.id
            let reopened = WaifuSession(settingsURL: url)
            checkEqual(reopened.panelPresets, session.panelPresets, "all three slots round trip on disk")
            reopened.settings.exportJPEGQuality = 0.5
            reopened.loadPanelPreset(id)
            checkEqual(reopened.frameOrder, expected.order, "sort signals and directions restored")
            checkEqual(reopened.frameLimits, expected.limits, "numeric ranges restored")
            checkEqual(reopened.tagFilter, expected.tags, "tag sets, order, threshold and score sorting restored")
            checkEqual(reopened.faceFilter, expected.faces, "face filter restored")
            checkEqual(reopened.groupingEnabled, true, "grouping restored")
            checkEqual(reopened.groupingThreshold, 0.58, "grouping threshold restored")
            checkEqual(reopened.settings.similarityThreshold, 0.58, "threshold persists with no media loaded")
            checkEqual(reopened.groupRows, false, "row preference restored")
            checkEqual(reopened.settings.perGroupFrameCount, 13, "cap restored")
            checkEqual(reopened.settings.exportJPEGQuality, 0.5, "unrelated preferences retained")
            checkEqual(reopened.regroupPending, false, "no stale pending indicator")
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
            session.settings.perGroupFrameCount = 37
            check(session.savePanelPreset(at: 1), "memory overwrites slot 2 again")
            let reopened = WaifuSession(settingsURL: url)
            reopened.loadPanelPreset(initial[1].id)
            checkEqual(reopened.settings.perGroupFrameCount, 37, "latest overwrite survives restart")
            reopened.settings.perGroupFrameCount = 55
            reopened.loadPanelPreset(initial[1].id)
            checkEqual(reopened.settings.perGroupFrameCount, 37, "reselecting the same number recalls it again")
            reopened.loadPanelPreset(initial[0].id)
            checkEqual(reopened.settings.perGroupFrameCount, WaifuSettings.default.perGroupFrameCount, "untouched slots load defaults")
            check(!session.savePanelPreset(at: 3), "out-of-range slot rejected")
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
