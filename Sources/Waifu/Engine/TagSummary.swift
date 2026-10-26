import Foundation
import Metal

struct TagSummary: Sendable {
    struct Tag: Sendable {
        let index: Int
        let matchingFrames: Int
        let scoredFrames: Int
        let mean: Double?
    }

    struct Labels: Sendable {
        let common: [String]
        let ranked: [String]
    }

    let frameCount: Int
    let tags: [Tag]
    let gpuChunks: Int

    func labels(using vocabulary: TagVocabulary) -> Labels {
        let common = tags.filter { $0.matchingFrames == frameCount }.map { vocabulary.labels[$0.index].name }
            .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        let ranked = tags.sorted { left, right in
            let lhs = left.mean ?? -1, rhs = right.mean ?? -1
            if lhs != rhs { return lhs > rhs }
            return vocabulary.labels[left.index].name.localizedStandardCompare(vocabulary.labels[right.index].name) == .orderedAscending
        }.map { vocabulary.labels[$0.index].name }
        return Labels(common: common, ranked: ranked)
    }
}

enum TagSummaryError: Error {
    case invalidInput, metalUnavailable, allocationFailed, commandFailed
}

/// Owns UInt8 general-tag vectors independently of display order and threshold.
/// All decoding, matrix packing and reductions execute on this actor, away from UI.
actor TagSummaryComputer {
    enum Backend: Sendable { case automatic, cpu, metal }

    private struct Row {
        let source: Data?
        let savedTags: [String]?
        let hasProbabilities: Bool
        let values: [UInt8]
        let populated: [Int]
    }
    private var rows: [FrameID: Row] = [:]
    private var cachedBytes = 0
    private var generalColumns: [Int: Int] = [:]
    private struct Slot { let chunk: Int; let row: Int }
    private struct Chunk {
        let buffer: any MTLBuffer
        let capacity: Int
        var used: Int = 0
    }
    private var metalOwner: MetalTagSummary?
    private var metalChunks: [Chunk] = []
    private var metalSlots: [FrameID: Slot] = [:]
    private var freeMetalSlots: [Slot] = []
    private var dirtyMetalRows: Set<FrameID> = []
    private var metalBytes = 0
    private var vocabularyFingerprint: UInt64?
    private let cacheByteLimit: Int
    private let chunkByteLimit: Int
    private let metalProvider: @Sendable () -> MetalTagSummary?
    private(set) var decodedVectorCount = 0
    private(set) var uploadedVectorCount = 0

    init(cacheByteLimit: Int = 64 * 1_024 * 1_024, chunkByteLimit: Int = 8 * 1_024 * 1_024,
         metalProvider: @escaping @Sendable () -> MetalTagSummary? = { MetalTagSummary.shared }) {
        self.cacheByteLimit = max(0, cacheByteLimit)
        self.chunkByteLimit = max(1, chunkByteLimit)
        self.metalProvider = metalProvider
    }

    func summarize(_ frames: [AnalyzedFrame], vocabulary: TagVocabulary, threshold: Float,
                   retaining sourceIDs: Set<FrameID>? = nil, backend: Backend = .automatic) throws -> TagSummary {
        try Task.checkCancellation()
        guard threshold.isFinite, (0...1).contains(threshold), Set(frames.map(\.id)).count == frames.count
        else { throw TagSummaryError.invalidInput }
        let fingerprint = vocabulary.cacheFingerprint
        if vocabularyFingerprint != fingerprint {
            rows.removeAll()
            cachedBytes = 0
            clearMetalCache()
            vocabularyFingerprint = fingerprint
            generalColumns = Dictionary(uniqueKeysWithValues: vocabulary.generalIndices.enumerated().map { ($0.element, $0.offset) })
        }
        if let sourceIDs {
            for id in Array(rows.keys) where !sourceIDs.contains(id) {
                let old = rows.removeValue(forKey: id)!
                cachedBytes -= old.values.count + old.populated.count * MemoryLayout<Int>.stride
                if let slot = metalSlots.removeValue(forKey: id) { freeMetalSlots.append(slot) }
                dirtyMetalRows.remove(id)
            }
        }
        let columns = vocabulary.generalIndices.count
        guard !frames.isEmpty, columns > 0 else {
            return TagSummary(frameCount: frames.count, tags: [], gpuChunks: 0)
        }
        var rowsWithTagData = 0
        var retainedScores = 0
        for frame in frames {
            if let data = frame.tagProbabilities {
                rowsWithTagData += 1
                retainedScores += TagProbabilityCodec.retainedCount(in: data) ?? columns
            } else if let tags = frame.topTags, !tags.isEmpty {
                rowsWithTagData += 1
                retainedScores += tags.count
            }
        }
        // Sparse CPU reduction skips absent heads; a GPU matrix still scans them.
        // Small or sparse selections avoid dispatch and selection-mask overhead.
        let preferCPU = rowsWithTagData < 512 || retainedScores <= rowsWithTagData * columns / 8
        let metal = backend == .cpu || (backend == .automatic && preferCPU) ? nil : metalProvider()
        if backend == .metal, metal == nil { throw TagSummaryError.metalUnavailable }
        if let metal {
            do {
                let totals = try reduceOnMetal(frames, vocabulary: vocabulary, threshold: threshold, metal: metal)
                return try summary(totals, vocabulary: vocabulary, frameCount: frames.count)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                clearMetalCache()
                if backend == .metal { throw error }
            }
        }
        var totals = Totals(columns: columns)
        let minimumCode = (89...255).first { TagProbabilityCodec.probability(UInt8($0)) > threshold } ?? 256
        let includeEqual = threshold < TagProbabilityCodec.storageThreshold
        // Legacy rows contribute membership only. Completely untagged photos
        // contribute no scores, but still count in the common-tag denominator.
        for frame in frames where frame.tagProbabilities != nil || !(frame.topTags ?? []).isEmpty {
            try Task.checkCancellation()
            let row = row(for: frame, vocabulary: vocabulary)
            for column in row.populated {
                let code = row.values[column]
                if code == 2 {
                    totals.present[column] += 1 // legacy membership without a score
                } else if Int(code) >= minimumCode || (code == 1 && includeEqual) {
                    totals.sums[column] += UInt64(code)
                    totals.scored[column] += 1
                    totals.present[column] += 1
                    if code == 1 { totals.cutoffEqual[column] += 1 }
                    if code == 89 { totals.cutoffAbove[column] += 1 }
                }
            }
        }
        return try summary(totals, vocabulary: vocabulary, frameCount: frames.count)
    }

    private struct Totals {
        var sums: [UInt64]
        var scored: [Int]
        var present: [Int]
        var cutoffEqual: [Int]
        var cutoffAbove: [Int]
        var gpuChunks = 0

        init(columns: Int) {
            sums = [UInt64](repeating: 0, count: columns)
            scored = [Int](repeating: 0, count: columns)
            present = [Int](repeating: 0, count: columns)
            cutoffEqual = [Int](repeating: 0, count: columns)
            cutoffAbove = [Int](repeating: 0, count: columns)
        }

        mutating func add(_ partial: [MetalTagSummary.Total]) {
            for column in partial.indices {
                sums[column] += UInt64(partial[column].sum)
                scored[column] += Int(partial[column].scored)
                present[column] += Int(partial[column].present)
                cutoffEqual[column] += Int(partial[column].cutoffEqual)
                cutoffAbove[column] += Int(partial[column].cutoffAbove)
            }
            gpuChunks += 1
        }
    }

    private func summary(_ totals: Totals, vocabulary: TagVocabulary, frameCount: Int) throws -> TagSummary {
        try Task.checkCancellation()
        var tags: [TagSummary.Tag] = []
        let equalCorrection: Double = Double(TagProbabilityCodec.storageThreshold) - 1.0 / 255
        let aboveCorrection: Double = Double(TagProbabilityCodec.storageThreshold.nextUp) - 89.0 / 255
        for column in vocabulary.generalIndices.indices where totals.present[column] > 0 {
            let scored: Int = totals.scored[column]
            let mean: Double?
            if scored > 0 {
                let quantizedSum: Double = Double(totals.sums[column]) / 255
                let equalAdjustment: Double = Double(totals.cutoffEqual[column]) * equalCorrection
                let aboveAdjustment: Double = Double(totals.cutoffAbove[column]) * aboveCorrection
                mean = (quantizedSum + equalAdjustment + aboveAdjustment) / Double(scored)
            } else {
                mean = nil
            }
            tags.append(TagSummary.Tag(index: vocabulary.generalIndices[column],
                matchingFrames: totals.present[column], scoredFrames: scored, mean: mean))
        }
        return TagSummary(frameCount: frameCount, tags: tags, gpuChunks: totals.gpuChunks)
    }

    private func clearMetalCache() {
        metalChunks = []
        metalSlots = [:]
        freeMetalSlots = []
        dirtyMetalRows = []
        metalBytes = 0
    }

    private func reduceOnMetal(_ frames: [AnalyzedFrame], vocabulary: TagVocabulary, threshold: Float,
                               metal: MetalTagSummary) throws -> Totals {
        if metalOwner !== metal { clearMetalCache(); metalOwner = metal }
        let columns = vocabulary.generalIndices.count
        let rowsPerChunk = max(1, chunkByteLimit / columns)
        var masks: [Int: [UInt32]] = [:]
        var transient: [UInt8] = []
        var totals = Totals(columns: columns)
        for frame in frames where frame.tagProbabilities != nil || !(frame.topTags ?? []).isEmpty {
            try Task.checkCancellation()
            let values = row(for: frame, vocabulary: vocabulary).values
            if rows[frame.id] != nil {
                let slot: Slot
                if let existing = metalSlots[frame.id] {
                    slot = existing
                } else {
                    if let free = freeMetalSlots.popLast() {
                        slot = free
                    } else {
                        if metalChunks.isEmpty || metalChunks.last!.used == metalChunks.last!.capacity {
                            let capacity = min(rowsPerChunk, (cacheByteLimit - metalBytes) / columns)
                            guard capacity > 0 else { throw TagSummaryError.allocationFailed }
                            let buffer = try metal.matrixBuffer(rows: capacity, columns: columns)
                            metalChunks.append(Chunk(buffer: buffer, capacity: capacity))
                            metalBytes += capacity * columns
                        }
                        let chunk = metalChunks.count - 1
                        slot = Slot(chunk: chunk, row: metalChunks[chunk].used)
                        metalChunks[chunk].used += 1
                    }
                    metalSlots[frame.id] = slot
                    dirtyMetalRows.insert(frame.id)
                }
                if dirtyMetalRows.contains(frame.id) {
                    values.withUnsafeBytes { bytes in
                        metalChunks[slot.chunk].buffer.contents().advanced(by: slot.row * columns)
                            .copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
                    }
                    dirtyMetalRows.remove(frame.id)
                    uploadedVectorCount += 1
                }
                masks[slot.chunk, default: [UInt32](repeating: 0, count: metalChunks[slot.chunk].capacity)][slot.row] = 1
            } else {
                transient.append(contentsOf: values)
                if transient.count / columns == rowsPerChunk {
                    totals.add(try metal.reduce(transient, rows: rowsPerChunk, columns: columns, threshold: threshold))
                    transient.removeAll(keepingCapacity: true)
                }
            }
        }
        if !transient.isEmpty {
            totals.add(try metal.reduce(transient, rows: transient.count / columns, columns: columns, threshold: threshold))
        }
        for chunk in masks.keys.sorted() {
            try Task.checkCancellation()
            totals.add(try metal.reduce(metalChunks[chunk].buffer, rows: metalChunks[chunk].used,
                                       columns: columns, included: masks[chunk]!, threshold: threshold))
        }
        return totals
    }

    private func row(for frame: AnalyzedFrame, vocabulary: TagVocabulary) -> Row {
        if let cached = rows[frame.id], cached.source == frame.tagProbabilities,
           cached.hasProbabilities || cached.savedTags == frame.topTags {
            return cached
        }
        var values = [UInt8](repeating: 0, count: vocabulary.generalIndices.count)
        var populated: [Int] = []
        let hasProbabilities: Bool
        if let data = frame.tagProbabilities, let pairs = TagProbabilityCodec.sparseBytes(in: data) {
            decodedVectorCount += 1
            hasProbabilities = true
            for pair in pairs {
                if let column = generalColumns[pair.index] {
                    values[column] = pair.value
                    populated.append(column)
                }
            }
        } else {
            hasProbabilities = false
            for tag in frame.topTags ?? [] {
                if let index = vocabulary.index(for: tag), let column = generalColumns[index], values[column] == 0 {
                    values[column] = 2 // reserved membership-only sentinel
                    populated.append(column)
                }
            }
        }
        let result = Row(source: frame.tagProbabilities, savedTags: frame.topTags,
                         hasProbabilities: hasProbabilities, values: values, populated: populated)
        if let old = rows.removeValue(forKey: frame.id) { cachedBytes -= old.values.count + old.populated.count * MemoryLayout<Int>.stride }
        let bytes = values.count + populated.count * MemoryLayout<Int>.stride
        // Stable bounded admission avoids cache churn on large albums.
        if cachedBytes + bytes <= cacheByteLimit {
            rows[frame.id] = result
            cachedBytes += bytes
            dirtyMetalRows.insert(frame.id)
        }
        return result
    }
}

/// A separate queue from visual routing. Input buffers belong to the worker actor;
/// each reduction owns its selection mask and output buffer.
final class MetalTagSummary: @unchecked Sendable {
    static let shared: MetalTagSummary? = try? MetalTagSummary()

    struct Total {
        var sum: UInt32
        var scored: UInt32
        var present: UInt32
        var cutoffEqual: UInt32
        var cutoffAbove: UInt32
    }
    private struct Parameters {
        var rows: UInt32
        var columns: UInt32
        var minimumCode: UInt32
        var includeEqual: UInt32
    }
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
              let queue = device.makeCommandQueue() else { throw TagSummaryError.metalUnavailable }
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let function = library.makeFunction(name: "summarize_tags") else { throw TagSummaryError.metalUnavailable }
        self.device = device
        self.queue = queue
        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    func reduce(_ values: [UInt8], rows: Int, columns: Int, threshold: Float) throws -> [Total] {
        try Task.checkCancellation()
        guard rows > 0, columns > 0, rows <= Int(UInt32.max), columns <= Int(UInt32.max),
              rows <= Int.max / columns, values.count == rows * columns,
              values.count <= device.maxBufferLength else { throw TagSummaryError.invalidInput }
        let input = values.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }
        guard let input else { throw TagSummaryError.allocationFailed }
        return try reduce(input, rows: rows, columns: columns, included: [UInt32](repeating: 1, count: rows), threshold: threshold)
    }

    func matrixBuffer(rows: Int, columns: Int) throws -> any MTLBuffer {
        guard rows > 0, columns > 0, rows <= Int.max / columns,
              rows * columns <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: rows * columns, options: .storageModeShared)
        else { throw TagSummaryError.allocationFailed }
        return buffer
    }

    func reduce(_ input: any MTLBuffer, rows: Int, columns: Int, included: [UInt32], threshold: Float) throws -> [Total] {
        try Task.checkCancellation()
        guard rows > 0, columns > 0, rows <= Int(UInt32.max), columns <= Int(UInt32.max),
              rows <= Int(UInt32.max) / columns, rows <= Int(UInt32.max) / 255, rows * columns <= input.length,
              included.count >= rows, threshold.isFinite, (0...1).contains(threshold),
              columns <= device.maxBufferLength / MemoryLayout<Total>.stride
        else { throw TagSummaryError.invalidInput }
        let mask = included.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }
        guard let mask, let output = device.makeBuffer(length: columns * MemoryLayout<Total>.stride, options: .storageModeShared)
        else { throw TagSummaryError.allocationFailed }
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
        else { throw TagSummaryError.commandFailed }
        let minimumCode = (89...255).first { TagProbabilityCodec.probability(UInt8($0)) > threshold } ?? 256
        var parameters = Parameters(rows: UInt32(rows), columns: UInt32(columns), minimumCode: UInt32(minimumCode),
                                    includeEqual: threshold < TagProbabilityCodec.storageThreshold ? 1 : 0)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(mask, offset: 0, index: 2)
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: columns, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(128, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        try Task.checkCancellation()
        command.commit()
        command.waitUntilCompleted()
        try Task.checkCancellation()
        guard command.status == .completed else { throw command.error ?? TagSummaryError.commandFailed }
        return Array(UnsafeBufferPointer(start: output.contents().bindMemory(to: Total.self, capacity: columns), count: columns))
    }

    private static let source = #"""
    #include <metal_stdlib>
    using namespace metal;
    struct Params { uint rows, columns, minimumCode, includeEqual; };
    struct Total { uint sum, scored, present, cutoffEqual, cutoffAbove; };
    kernel void summarize_tags(device const uchar* values [[buffer(0)]], device Total* totals [[buffer(1)]],
                               device const uint* included [[buffer(2)]], constant Params& p [[buffer(3)]],
                               uint tag [[thread_position_in_grid]]) {
        if (tag >= p.columns) return;
        uint sum = 0, scored = 0, present = 0, cutoffEqual = 0, cutoffAbove = 0;
        for (uint row = 0; row < p.rows; ++row) {
            if (!included[row]) continue;
            uint value = values[row * p.columns + tag];
            if (value >= p.minimumCode || (value == 1 && p.includeEqual)) {
                sum += value;
                ++scored; ++present;
                cutoffEqual += value == 1;
                cutoffAbove += value == 89;
            } else if (value == 2) {
                ++present;
            }
        }
        totals[tag] = {sum, scored, present, cutoffEqual, cutoffAbove};
    }
    """#
}
