import Foundation

struct TagLabel: Sendable, Equatable {
    let name: String
    let category: Int
}

/// The label array is never sorted: each CSV row names the same output head.
struct TagVocabulary: Sendable {
    static let headCount = 10_861
    static let generalCount = 8_106
    static let topCount = 15
    static let kaomoji: Set<String> = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]
    let labels: [TagLabel]
    let generalIndices: [Int]
    let generalTags: [String]
    private let lookup: [String: Int]

    /// Stable identity for the exact model-head index order. Only the numeric
    /// fingerprint is persisted with tag results; the vocabulary text is not.
    var cacheFingerprint: UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for label in labels {
            for byte in label.name.utf8 {
                hash ^= UInt64(byte)
                hash &*= 1_099_511_628_211
            }
            hash ^= 0xff
            hash &*= 1_099_511_628_211
            hash ^= UInt64(label.category)
            hash &*= 1_099_511_628_211
        }
        return hash
    }

    static func normalized(_ value: String) -> String {
        let lowered = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return kaomoji.contains(lowered) ? lowered : lowered.replacingOccurrences(of: "_", with: " ")
    }

    static func load(from url: URL) throws -> Self {
        let text = try String(contentsOf: url, encoding: .utf8)
        let rows = try csvRows(text)
        guard let header = rows.first,
              let nameColumn = header.firstIndex(of: "name"),
              let categoryColumn = header.firstIndex(of: "category") else {
            throw TaggerError.invalidVocabulary("missing name/category columns")
        }
        var labels: [TagLabel] = []
        for row in rows.dropFirst() {
            guard row.count == header.count, let category = Int(row[categoryColumn]),
                  [0, 4, 9].contains(category), !row[nameColumn].isEmpty else {
                throw TaggerError.invalidVocabulary("malformed row \(labels.count + 2)")
            }
            labels.append(TagLabel(name: normalized(row[nameColumn]), category: category))
        }
        guard labels.count == headCount,
              Array(labels.prefix(4).map(\.name)) == TagRating.allCases.map(\.rawValue),
              labels.prefix(4).allSatisfy({ $0.category == 9 }),
              labels.filter({ $0.category == 9 }).count == 4,
              labels.filter({ $0.category == 0 }).count == generalCount,
              labels.filter({ $0.category == 4 }).count == 2_751 else {
            throw TaggerError.invalidVocabulary("expected 4 rating, 8,106 general and 2,751 character heads in model order")
        }
        let indices = labels.indices.filter { labels[$0].category == 0 }
        var lookup: [String: Int] = [:]
        for index in indices {
            guard lookup.updateValue(index, forKey: labels[index].name) == nil else {
                throw TaggerError.invalidVocabulary("duplicate general tag \(labels[index].name)")
            }
        }
        return Self(labels: labels, generalIndices: indices, generalTags: indices.map { labels[$0].name }, lookup: lookup)
    }

    static func loadAvailable() -> Self? {
        guard let directory = WDTaggerConfiguration.availableDirectory else { return nil }
        return try? load(from: directory.appendingPathComponent("wd-tagger-labels.csv"))
    }

    func index(for tag: String) -> Int? { lookup[Self.normalized(tag)] }

    /// Exact matches first, then prefix matches, then substring matches. Ties
    /// follow output-head order, so keyboard selection remains deterministic.
    func autocomplete(query: String, limit: Int = 12) -> [String] {
        let query = Self.normalized(query)
        guard limit > 0, !query.isEmpty else { return [] }
        return generalTags.enumerated().compactMap { index, tag -> (Int, Int, String)? in
            guard tag.contains(query) else { return nil }
            return (tag == query ? 0 : tag.hasPrefix(query) ? 1 : 2, index, tag)
        }.sorted { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }.prefix(limit).map(\.2)
    }

    func result(probabilities: [Float], threshold: Float = 0.35) throws -> TagResult {
        guard probabilities.count == Self.headCount,
              probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              threshold.isFinite, (0...1).contains(threshold) else { throw TaggerError.invalidProbabilities }
        let ratingIndex = (0..<4).max { probabilities[$0] < probabilities[$1] } ?? 0
        let sorted = generalIndices.filter { probabilities[$0] > threshold }.sorted {
            probabilities[$0] == probabilities[$1] ? $0 < $1 : probabilities[$0] > probabilities[$1]
        }
        return TagResult(probabilities: probabilities, rating: TagRating.allCases[ratingIndex],
                         topTags: sorted.prefix(Self.topCount).map { labels[$0].name },
                         generalScore: sorted.prefix(Self.topCount).reduce(0) { $0 + probabilities[$1] })
    }

    /// Small RFC 4180 reader: quoted commas and escaped quotes are meaningful
    /// label characters, and must not shift the model's output indices.
    private static func csvRows(_ text: String) throws -> [[String]] {
        var rows: [[String]] = [], row: [String] = [], field = ""
        var quoted = false, closedQuote = false
        var characters = text.makeIterator()
        var next = characters.next()
        while let character = next {
            next = characters.next()
            if quoted {
                if character == "\"" {
                    if next == "\"" { field.append("\""); next = characters.next() }
                    else { quoted = false; closedQuote = true }
                } else { field.append(character) }
            } else if character == "\"" {
                guard field.isEmpty && !closedQuote else { throw TaggerError.invalidVocabulary("unexpected quote") }
                quoted = true
            } else if character == "," {
                row.append(field); field = ""; closedQuote = false
            } else if character == "\n" || character == "\r\n" || character == "\r" {
                row.append(field)
                if !(row.count == 1 && field.isEmpty) { rows.append(row) }
                row = []; field = ""; closedQuote = false
                if character == "\r" && next == "\n" { next = characters.next() }
            } else {
                guard !closedQuote else { throw TaggerError.invalidVocabulary("text following closing quote") }
                field.append(character)
            }
        }
        guard !quoted else { throw TaggerError.invalidVocabulary("unterminated quote") }
        if !field.isEmpty || !row.isEmpty { row.append(field); rows.append(row) }
        return rows
    }
}
