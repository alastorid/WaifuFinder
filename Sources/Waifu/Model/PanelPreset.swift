import Foundation

/// Only reusable right-panel preferences; media, selections and analysis settings stay in the workspace.
struct PanelPreset: Codable, Equatable, Identifiable, Sendable {
    var id = UUID()
    let name: String
    let order: FrameOrder
    let limits: FrameLimits
    let tags: TagFilter
    let faces: FrameFaceFilter
    let groupingEnabled: Bool
    let groupingThreshold: Float
    let groupRows: Bool
    let perGroupCount: Int

    static func defaultSlot(_ number: Int) -> PanelPreset {
        PanelPreset(name: "Preset \(number)", order: .defaultOrder, limits: .none,
                    tags: TagFilter(), faces: .all, groupingEnabled: false,
                    groupingThreshold: WaifuSettings.default.similarityThreshold,
                    groupRows: true, perGroupCount: WaifuSettings.default.perGroupFrameCount)
    }
}
