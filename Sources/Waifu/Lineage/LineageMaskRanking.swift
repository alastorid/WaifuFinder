import Foundation

enum LineageMaskRanking {
    static func ordered(_ photos: [LineagePhoto], findings: [String: LineageSafireSummary]) -> [LineagePhoto] {
        photos.enumerated().sorted { a, b in
            let left = findings[a.element.id].flatMap { $0.contentIdentity == a.element.contentIdentity ? $0 : nil }
            let right = findings[b.element.id].flatMap { $0.contentIdentity == b.element.contentIdentity ? $0 : nil }
            switch (left, right) {
            case let (.some(l), .some(r)): return l.ratio == r.ratio ? a.offset < b.offset : l.ratio < r.ratio
            case (.some, .none): return true
            case (.none, .some): return false
            case (.none, .none): return a.offset < b.offset
            }
        }.map(\.element)
    }
}
