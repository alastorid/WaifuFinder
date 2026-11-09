import CoreGraphics
import SwiftUI

struct ArenaStudioView: View {
    @Environment(\.displayScale) private var displayScale
    @StateObject private var studio = ArenaStudio()
    var initialSources: [ArenaSource] = []
    @State private var swipe = 0.5
    @State private var blink = false
    @State private var dragOrigin = CGSize.zero
    @State private var zoomOrigin = 1.0
    var body: some View {
        VStack(spacing: 10) {
            toolbar
            if let analysis = studio.analysis {
                summary(analysis)
                gallery(analysis)
                HSplitView {
                    VStack(spacing: 8) {
                        viewControls
                        workspace(analysis).frame(maxWidth: .infinity, maxHeight: .infinity)
                        if let difference = studio.difference {
                            Text("Output → \(sourceName(studio.sourceID, analysis)): \(difference.changedPixels.formatted()) changed · \(difference.selectedFromOtherSourcePixels.formatted()) from source selection · \((difference.changedPixels-difference.selectedFromOtherSourcePixels).formatted()) unexpected · \(difference.uncoveredPixels.formatted()) without overlap")
                                .font(.caption).foregroundStyle(.secondary).lineLimit(2)
                        } else if analysis.sources.first(where: { $0.id == studio.sourceID })?.accepted == false {
                            Text("Selected source has no reliable alignment. Its independent findings are available; output comparison is unavailable.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }.frame(minWidth: 550)
                    inspector(analysis).frame(minWidth: 260, idealWidth: 310, maxWidth: 420)
                }
            } else {
                ContentUnavailableView("Find the best photograph", systemImage: "photo.stack",
                    description: Text("Choose 2–16 versions of the same photograph. PhotoArena reviews every candidate, recommends the best source, and recovers supported clean regions when needed."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            HStack {
                if studio.busy { ProgressView().controlSize(.small) }
                Text(studio.status).font(.caption).textSelection(.enabled)
                Spacer()
                Text(studio.phase.rawValue.capitalized).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(14).frame(minWidth: 1050, minHeight: 720)
        .dropDestination(for: URL.self) { urls, _ in
            guard !studio.busy, urls.allSatisfy(\.isFileURL), (2...16).contains(urls.count) else { return false }
            studio.open(urls.map { ArenaSource(id:$0.absoluteString,url:$0,seconds:0) }); return true
        }
        .task { if !initialSources.isEmpty { studio.open(initialSources) } }
        .onChange(of: studio.sourceID) { _, _ in studio.refreshDisplay() }
        .onChange(of: studio.regionID) { _, _ in studio.loadRegion() }
        .onChange(of: studio.tolerance) { _, _ in studio.refreshDisplay() }
        .onChange(of: studio.opacity) { _, _ in studio.refreshDisplay() }
        .onChange(of: studio.differenceScope) { _, _ in studio.refreshDisplay() }
        .onDisappear { studio.close() }
    }
    private var toolbar: some View {
        HStack {
            Text("PhotoArena").font(.title2.bold())
            Button("Choose Candidates…", action: studio.choose).disabled(studio.busy)
            Spacer()
            if studio.busy { Button("Cancel", action: studio.cancel) }
            else if studio.analysis != nil && studio.plan?.pendingRegionIDs.isEmpty != true {
                Button("Review Next Regions", action: studio.reviewMore)
            }
            Button("Undo", action: studio.undo).disabled(studio.busy)
            Button("Export Result…", action: studio.export).disabled(studio.plan == nil || studio.busy)
        }.buttonStyle(.bordered)
    }
    private func summary(_ analysis: ArenaAnalysis) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(studio.plan?.bestSourceID.map { "Best candidate: " + sourceName($0, analysis) } ?? "Best candidate: evidence insufficient or tied")
                    .font(.headline).lineLimit(1)
                Spacer()
                Text(studio.plan?.kind == .original ? "Original file preserved" : studio.plan?.kind == .composite ? "Recovered composite" : "Recovery unresolved")
                    .font(.subheadline).foregroundStyle(studio.plan?.kind == .original ? .green : .secondary)
            }
            Text("\(analysis.sources.count) candidates · \(studio.changeRegions.count) changed regions · \(studio.plan?.pendingRegionIDs.count ?? 0) review tasks pending · \(studio.plan?.unresolvedRegionIDs.count ?? 0) regions with uncertainty")
                .font(.caption)
            Text((studio.plan?.pendingRegionIDs.isEmpty == false ? "Provisional choice · review is incomplete. " : "") + (studio.plan?.reasons.first ?? "Preparing evidence…")).font(.caption).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private func gallery(_ analysis: ArenaAnalysis) -> some View {
        let counts = contributions(analysis)
        let area = Double(analysis.width)*Double(analysis.height)
        return ScrollView(.horizontal) {
            HStack(spacing: 10) {
                ForEach(candidateCards(analysis), id: \.sourceID) { rank in
                    if rank.sourceID == analysis.sources.first(where: { $0.id != studio.plan?.bestSourceID })?.id {
                        Divider()
                        Text("The rest").font(.headline).padding(.horizontal, 6)
                    }
                    Button {
                        studio.sourceID = rank.sourceID
                        studio.differenceScope = "Detected changes"
                    } label: {
                        HStack(spacing: 7) {
                            if let image = studio.previews[rank.sourceID] {
                                Image(decorative: image, scale: 1).resizable().scaledToFit().frame(width: 70, height: 62)
                            }
                            VStack(alignment: .leading, spacing: 3) {
                                if rank.sourceID == studio.plan?.bestSourceID {
                                    Text("Suggested best").font(.caption.bold()).foregroundStyle(.green)
                                }
                                Text(sourceName(rank.sourceID, analysis)).font(.caption.bold()).lineLimit(1)

                                Text("\(rank.confidence.rawValue) confidence · \(Int(rank.coverage*100))% regional review").font(.caption2)
                                Text(rank.defects.isEmpty ? "No classified defects yet" : rank.defects.map(\.rawValue).joined(separator: ", "))
                                    .font(.caption2).lineLimit(1)
                                Text("\(Double(counts[rank.sourceID,default:0])/area,format:.percent.precision(.fractionLength(0...1))) of output").font(.caption2)
                            }.frame(width: 200, alignment: .leading)
                        }.padding(6).background(studio.sourceID == rank.sourceID ? Color.accentColor.opacity(0.15) : Color.secondary.opacity(0.06))
                            .clipShape(RoundedRectangle(cornerRadius: 7))
                    }.buttonStyle(.plain)
                }
            }
        }.frame(height: 86)
    }
    private func candidateCards(_ analysis: ArenaAnalysis) -> [ArenaCandidateAssessment] {
        let candidates = analysis.sources.compactMap { source in studio.plan?.rankings.first { $0.sourceID == source.id } }
        return candidates.filter { $0.sourceID == studio.plan?.bestSourceID }
            + candidates.filter { $0.sourceID != studio.plan?.bestSourceID }
    }
    private func contributions(_ analysis: ArenaAnalysis) -> [String:Int] {
        guard let plan = studio.plan else { return [:] }
        var counts = [plan.backboneID:analysis.width*analysis.height]
        for selection in plan.selections where selection.regionID != "backbone" {
            let pixels = selection.mask?.pixelCount ?? 0
            counts[plan.backboneID,default:0] -= pixels; counts[selection.sourceID,default:0] += pixels
        }
        return counts
    }
    private var viewControls: some View {
        VStack(spacing: 5) {
            ScrollView(.horizontal) {
                HStack {
                    ForEach(["Result", "Side by side", "Swipe", "Blink", "Overlay", "Provenance", "Alignment"], id: \.self) { mode in
                        Button(mode) { studio.inspectionMode = mode }.tint(studio.inspectionMode == mode ? .accentColor : .secondary)
                    }
                    Button("Fit") { studio.zoom = 1; studio.pan = .zero; studio.nativeDetail = false; dragOrigin = .zero; zoomOrigin = 1 }
                    Button("Native region") { studio.nativeDetail.toggle(); studio.loadRegion() }.disabled(studio.regionID == nil)
                }.buttonStyle(.bordered).controlSize(.small)
            }
            HStack {
                Toggle("Difference annotations", isOn: $studio.showRed).toggleStyle(.checkbox)
                Text("Audit threshold").font(.caption)
                Slider(value: $studio.tolerance, in: 1...64, step: 1).frame(maxWidth: 100)
                    .disabled(studio.differenceScope != "Output differences")
                    .help("Threshold for the output pixel audit. Region boxes show the detected regions used for model review.")

                if studio.inspectionMode == "Swipe" { Slider(value: $swipe, in: 0...1).frame(maxWidth: 100) }
                if studio.inspectionMode == "Blink" { Button(blink ? "Show Result" : "Show Source") { blink.toggle() } }
                Spacer(minLength: 0)
            }.controlSize(.small)
            HStack {
                ForEach(["Detected changes", "Output differences"],id:\.self) { scope in
                    Button(scope) { studio.differenceScope = scope; studio.showRed = true }
                        .tint(studio.differenceScope == scope ? .accentColor : .secondary)
                }
                Spacer()
            }.buttonStyle(.bordered).controlSize(.small)
        }
    }
    private func workspace(_ analysis: ArenaAnalysis) -> some View {
        Group {
            if studio.nativeDetail, let region = studio.region {
                ScrollView([.horizontal,.vertical]) {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 220))]) {
                        if let crop = studio.resultCrop {
                            VStack { Text("Result · 100%").font(.caption.bold()); nativeImage(crop) }
                        }
                        ForEach(analysis.sources) { source in
                            VStack(alignment: .leading) {
                                Text(source.url.lastPathComponent).font(.caption.bold())
                                if let crop = studio.nativeCrops[source.id] {
                                    nativeImage(crop)
                                } else { Text("No reliable corresponding crop").frame(height: 240) }
                            }
                        }
                    }.frame(minWidth: 520)
                    Text("Native region x \(Int(region.rect.x)), y \(Int(region.rect.y)), \(Int(region.rect.width)) × \(Int(region.rect.height)); bottom-left reference coordinates.")
                        .font(.caption)
                }
            } else if studio.inspectionMode == "Side by side" {
                HStack(spacing: 4) { canvas(analysis, source: true); canvas(analysis, source: false) }
            } else { canvas(analysis, source: false) }
        }
    }
    private func nativeImage(_ crop: CGImage) -> some View {
        Image(decorative:crop,scale:displayScale).interpolation(.none)
            .frame(width:CGFloat(crop.width)/displayScale,height:CGFloat(crop.height)/displayScale)
    }
    private func canvas(_ analysis: ArenaAnalysis, source: Bool) -> some View {
        GeometryReader { geometry in
            let fit = min(geometry.size.width/Double(analysis.width), geometry.size.height/Double(analysis.height))
            let scale = fit*studio.zoom
            let size = CGSize(width: Double(analysis.width)*scale, height: Double(analysis.height)*scale)
            ZStack {
                Color.black.opacity(0.94)
                ZStack {
                    let showingSource = source || studio.inspectionMode == "Alignment" || (studio.inspectionMode == "Blink" && blink)
                    if let cg = showingSource ? studio.previews[studio.sourceID] : studio.resultImage {
                        Image(decorative: cg, scale: 1).resizable().interpolation(.none).frame(width: size.width, height: size.height)
                    }
                    if !source, studio.inspectionMode == "Swipe", let cg = studio.previews[studio.sourceID] {
                        Image(decorative: cg, scale: 1).resizable().frame(width: size.width, height: size.height)
                            .mask(alignment: .leading) { Rectangle().frame(width: size.width*swipe) }
                    }
                    if !source, studio.inspectionMode == "Overlay", let cg = studio.previews[studio.sourceID] {
                        Image(decorative: cg, scale: 1).resizable().frame(width: size.width, height: size.height).opacity(0.4)
                    }
                    if studio.showRed {
                        annotations(analysis, scale: scale, size: size, showingSource: showingSource)
                    }
                    if studio.inspectionMode == "Provenance", let plan = studio.plan {
                        Canvas { context, _ in
                            for selection in plan.selections {
                                let i = analysis.sources.firstIndex { $0.id == selection.sourceID } ?? 0
                                let color = Color(hue: Double(i)/Double(analysis.sources.count), saturation: 0.8, brightness: 0.95).opacity(0.45)
                                var path = Path()
                                if let mask = selection.mask {
                                    for run in mask.runs {
                                        path.addRect(CGRect(x: Double(run.x)*scale, y: Double(analysis.height-run.y-1)*scale,
                                            width: Double(run.length)*scale, height: scale))
                                    }
                                } else {
                                    path.addRect(CGRect(x: selection.rect.x*scale, y: (Double(analysis.height)-selection.rect.y-selection.rect.height)*scale,
                                        width: selection.rect.width*scale, height: selection.rect.height*scale))
                                }
                                context.fill(path, with: .color(color))
                            }
                        }.frame(width: size.width, height: size.height)
                    }
                    if studio.inspectionMode == "Alignment", let selected = analysis.sources.first(where: { $0.id == studio.sourceID && $0.accepted }) {
                        Path { path in
                            let points = [CGPoint.zero,CGPoint(x:selected.width,y:0),CGPoint(x:selected.width,y:selected.height),CGPoint(x:0,y:selected.height)]
                                .map(selected.transform.point).map { CGPoint(x:$0.x*scale,y:(Double(analysis.height)-$0.y)*scale) }
                            path.addLines(points); path.closeSubpath()
                        }.stroke(.green,style:StrokeStyle(lineWidth:2,dash:[5,4])).frame(width:size.width,height:size.height)
                    }
                    if let region = studio.region {
                        Rectangle().stroke(.yellow, lineWidth: 1)
                            .frame(width: region.rect.width*scale, height: region.rect.height*scale)
                            .position(x: (region.rect.x+region.rect.width/2)*scale,
                                y: (Double(analysis.height)-region.rect.y-region.rect.height/2)*scale)
                    }
                }.frame(width: size.width, height: size.height).offset(studio.pan)
            }.frame(width: geometry.size.width, height: geometry.size.height).clipped()
                .contentShape(Rectangle())
                .onTapGesture { location in
                    guard scale > 0 && scale.isFinite else { return }
                    let x = (location.x-(geometry.size.width-size.width)/2-studio.pan.width)/scale
                    let y = Double(analysis.height)-(location.y-(geometry.size.height-size.height)/2-studio.pan.height)/scale
                    studio.regionID = studio.changeRegions.first { region in
                        (region.changeMask?.bounds ?? region.rect.cgRect).contains(CGPoint(x: x,y: y))
                    }?.id
                }
                .gesture(DragGesture().onChanged { value in
                    studio.pan = CGSize(width: dragOrigin.width+value.translation.width, height: dragOrigin.height+value.translation.height)
                }.onEnded { _ in dragOrigin = studio.pan })
                .simultaneousGesture(MagnifyGesture().onChanged { value in
                    studio.zoom = max(1, min(32, zoomOrigin*value.magnification))
                }.onEnded { _ in zoomOrigin = studio.zoom })
        }
    }
    private func annotations(_ analysis: ArenaAnalysis, scale: Double, size: CGSize, showingSource: Bool) -> some View {
        Canvas { context, _ in
            let regions = studio.changeRegions
            for region in regions {
                let bounds = region.changeMask?.bounds ?? region.rect.cgRect
                let rect = CGRect(x: bounds.minX*scale, y: (Double(analysis.height)-bounds.maxY)*scale,
                                  width: bounds.width*scale, height: bounds.height*scale)
                context.stroke(Path(rect), with: .color(.red), lineWidth: region.id == studio.regionID ? 2.5 : 1)
            }
            // Only the selected region gets a callout: dense differences must not cover the photo.
            guard let region = studio.region else { return }
            let candidate = showingSource ? studio.sourceID :
                (studio.plan?.selections.last { $0.regionID == region.id }?.sourceID ?? studio.plan?.backboneID ?? studio.sourceID)
            let finding = studio.records.last { $0.id == region.id }?.finding(candidate)
            let positive = finding.map { $0.usable && $0.cleanliness >= 90 && $0.alteration <= 10 && $0.enhancement <= 10 && $0.defects.allSatisfy { [.natural, .color, .geometry, .compression].contains($0) } } ?? false
            let uncertain = finding == nil || finding?.confidence == .low || finding?.defects.contains(.unknown) == true
            let color: Color = uncertain ? .orange : positive ? .green : .red
            let prefix = uncertain ? "?" : positive ? "+" : "−"
            let label = "\(prefix) \(sourceName(candidate, analysis)) · \(finding?.confidence.rawValue ?? "unreviewed")\n\(finding?.evidence ?? "Awaiting regional model review")"
            let bounds = region.changeMask?.bounds ?? region.rect.cgRect
            let anchor = CGPoint(x: bounds.midX*scale, y: (Double(analysis.height)-bounds.maxY)*scale)
            let width = min(300.0, max(80, size.width-16))
            let text = context.resolve(Text(label).font(.caption).foregroundColor(color))
            let measured = text.measure(in: CGSize(width: width-16, height: 130))
            let box = CGRect(x: min(max(8, anchor.x+24), max(8, size.width-width-8)),
                             y: min(max(8, anchor.y-measured.height-40), max(8, size.height-measured.height-24)),
                             width: width, height: measured.height+16)
            var leader = Path(); leader.move(to: anchor)
            leader.addLine(to: CGPoint(x: box.minX, y: box.maxY)); leader.addLine(to: CGPoint(x: box.maxX, y: box.maxY))
            context.stroke(leader, with: .color(color), lineWidth: 1)
            context.fill(Path(box), with: .color(.black.opacity(0.85)))
            context.draw(text, in: box.insetBy(dx: 8, dy: 8))
        }.frame(width: size.width, height: size.height).allowsHitTesting(false)
    }
    private func inspector(_ analysis: ArenaAnalysis) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                Text("Forensic evidence").font(.headline)
                if let rank = studio.plan?.rankings.first(where: { $0.sourceID == studio.sourceID }) {
                    Text(sourceName(rank.sourceID, analysis)).bold()
                    Text("Comparative originality: \(score(rank.originality))/100 · \(rank.confidence.rawValue) confidence").font(.caption)
                    Text("Cleanliness \(score(rank.cleanliness)) · native detail \(score(rank.detail)) · alteration \(score(rank.alteration)) · enhancement \(score(rank.enhancement)) · degradation \(score(rank.degradation)) · consistency \(score(rank.consistency))")
                        .font(.caption)
                    Button("Use as Backbone") { studio.pickSource(rank.sourceID) }
                        .disabled(studio.busy || analysis.sources.first(where: { $0.id == rank.sourceID })?.accepted != true)
                    ForEach(rank.reasons.prefix(3),id:\.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                }
                Divider()
                HStack { Text("Changed regions").bold(); Spacer(); Text("\(studio.changeRegions.count)").font(.caption) }
                ForEach(studio.changeRegions) { region in
                    Button {
                        studio.regionID = region.id
                    } label: {
                        HStack {
                            Circle().fill(studio.records.contains { $0.id == region.id } ? Color.red : .orange).frame(width: 6,height: 6)
                            Text("x \(Int(region.rect.x)), y \(Int(region.rect.y)) · \(region.changeMask?.pixelCount ?? 0) pixels")
                                .font(.caption).lineLimit(1)
                            Spacer()
                        }
                    }.buttonStyle(.plain)
                }
                if let region = studio.region {
                    Divider()
                    Text("Selected region").bold()
                    Text("\(region.id) · \(region.changeMask?.components.count ?? 0) connected components").font(.caption)
                    if let record = studio.selectedReview {
                        ForEach(analysis.sources) { source in
                            if let finding = record.finding(source.id) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(source.url.lastPathComponent).font(.caption.bold())
                                    Text(finding.defects.map(\.rawValue).joined(separator: ", ") + " · " + finding.confidence.rawValue).font(.caption)
                                    Text(finding.evidence).font(.caption).textSelection(.enabled)
                                    Button("Use this Region") { studio.overrideRegion(source.id) }
                                        .disabled(studio.busy || !source.accepted)
                                }
                            }
                        }
                    } else { Text("Awaiting model review. This region has not been classified.").font(.caption).foregroundStyle(.orange) }
                    ForEach(region.evidence, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
                    if let selection = studio.plan?.selections.first(where: { $0.regionID == region.id }) {
                        Text("Recovery: " + sourceName(selection.sourceID, analysis)).font(.caption.bold())
                        ForEach(selection.reasons, id: \.self) { Text($0).font(.caption) }
                    } else { Text("Backbone retained; no supported replacement selected.").font(.caption) }
                    if let check = studio.plan?.verification?.first(where: { $0.regionID == region.id }) {
                        Text(check.reason).font(.caption).foregroundStyle(check.accepted ? Color.secondary : Color.orange)
                    }
                }
                Divider()
                Text("Scores are uncalibrated assessments. Shared watermarks and shared edits may be unobservable. Red boxes locate measured differences. Leader labels show model findings: + green, − red, ? uncertain. Findings describe the named candidate; differences alone are not defects.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(analysis.warnings, id: \.self) { Text($0).font(.caption).foregroundStyle(.secondary) }
            }.padding(.leading, 10).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
    private func sourceName(_ id: String, _ analysis: ArenaAnalysis) -> String {
        analysis.sources.first { $0.id == id }?.url.lastPathComponent ?? id
    }
    private func score(_ value: Double?) -> String { value.map { String(format: "%.0f", $0) } ?? "—" }
}
