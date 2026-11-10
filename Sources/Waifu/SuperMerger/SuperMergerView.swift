import AppKit
import SwiftUI

struct SuperMergerView: View {
    @StateObject private var model = ArenaStudio()
    let sources: [ArenaSource]
    let close: () -> Void
    @State private var zoom: CGFloat = 1
    @State private var center = CGPoint(x: 0.5, y: 0.5)
    @State private var scale: CGFloat = 1
    @State private var annotations = true
    @State private var showingSource = false

    var body: some View {
        HSplitView {
            Group {
                if let image = showingSource ? model.previews[model.sourceID] : model.resultImage, let analysis = model.analysis {
                    ZoomableFrameImage(image: NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height)),
                        frameID: FrameID(sampleIndex: 0), zoomRatio: $zoom, center: $center, displayedScale: $scale,
                        regions: annotations ? model.changeRegions.map {
                            PreviewRegion(id: $0.id, rect: CGRect(x: $0.rect.x / Double(analysis.width),
                                y: $0.rect.y / Double(analysis.height), width: $0.rect.width / Double(analysis.width),
                                height: $0.rect.height / Double(analysis.height)))
                        } : [], selectedRegion: model.regionID,
                        onRegionSelection: { model.regionID = $0 })
                } else {
                    ContentUnavailableView("Super Merger", systemImage: "photo",
                        description: Text(model.admissionReason ?? "Select versions of the same photograph to recover one result."))
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Label("Super Merger", systemImage: "square.stack.3d.up").font(.title3.bold())
                        Spacer()
                        Button("Done", action: close)
                    }
                    Text(model.status).font(.callout).foregroundStyle(.secondary)
                    if let output = model.output {
                        Text(output.plan.kind == .original ? "Original source preserved" :
                             output.plan.kind == .composite ? "Recovered from source pixels" : "Recovery remains uncertain")
                            .font(.headline)
                    }
                    if model.busy { ProgressView(); Button("Cancel", action: model.cancel) }
                    Button("Choose Sources…", action: model.choose).disabled(model.busy)
                    HStack {
                        Button("Fit") { zoom = 1; center = CGPoint(x: 0.5, y: 0.5) }
                        Button("Preview 100%") { zoom /= max(scale, 0.0001) }
                    }
                    .disabled(showingSource ? model.previews[model.sourceID] == nil : model.resultImage == nil)
                    Toggle("Show regions", isOn: $annotations)
                    Text("\(model.changeRegions.count) difference regions").font(.caption)
                    if let analysis = model.analysis {
                        Text("Sources").font(.headline)
                        ForEach(analysis.sources) { source in
                            Button {
                                model.difference = nil
                                model.sourceID = source.id
                                showingSource = true
                            } label: {
                                HStack {
                                    Image(systemName: model.sourceID == source.id && showingSource ? "eye.fill" : "photo")
                                    Text(source.url.lastPathComponent).lineLimit(2)
                                    Spacer()
                                    if source.id == model.output?.plan.backboneID {
                                        Text("Backbone").font(.caption2).foregroundStyle(.secondary)
                                    }
                                }
                            }.buttonStyle(.plain).disabled(model.busy)
                        }
                        HStack {
                            Button("Result") { showingSource = false }
                                .tint(showingSource ? .secondary : .accentColor)
                            Button("Audit differences") {
                                showingSource = false
                                model.differenceScope = "Output differences"
                                model.refreshDisplay()
                            }.disabled(model.busy || model.output == nil)
                        }.buttonStyle(.bordered)
                        if let difference = model.difference {
                            Text("\(difference.changedPixels.formatted()) changed pixels · \(difference.selectedFromOtherSourcePixels.formatted()) expected from source selection")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Divider()
                        Text("Regions").font(.headline)
                        ForEach(Array(model.changeRegions.enumerated()), id: \.element.id) { index, region in
                            Button {
                                model.regionID = region.id
                            } label: {
                                HStack {
                                    Image(systemName: model.regionID == region.id ? "scope" : "rectangle.dashed")
                                    Text("Region \(index + 1)")
                                    Spacer()
                                }
                            }.buttonStyle(.plain)
                        }
                        ForEach(model.excludedSources) { source in
                            Text("Excluded \(source.url.lastPathComponent): \(source.issues.joined(separator: " "))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if let region = model.region {
                            Divider()
                            if let decision = model.selectedReview?.result.review {
                                Text(decision.reason ?? "Review pending").font(.callout)
                                if decision.unresolved == true {
                                    Label("Model abstained", systemImage: "questionmark.circle").foregroundStyle(.orange)
                                }
                            }
                            Text("Selected region").font(.headline)
                            let selection = model.output?.plan.selections.last { $0.regionID == region.id }
                            let used = selection?.sourceID ?? model.output?.plan.backboneID
                            Text(selection?.manual == true ? "Manual selection" : "Automatic selection").font(.caption)
                            if let verification = model.output?.plan.verification?.first(where: { $0.regionID == region.id }) {
                                Text(verification.reason).font(.caption).foregroundStyle(verification.accepted ? Color.secondary : Color.orange)
                            }
                            ForEach(analysis.sources) { source in
                                VStack(alignment: .leading) {
                                    Text(source.url.lastPathComponent + (used == source.id ? " · Used in result" : ""))
                                        .font(.caption.bold())
                                    if let image = model.nativeCrops[source.id] {
                                        Image(decorative: image, scale: 1).resizable().scaledToFit().frame(maxHeight: 160)
                                    }
                                    if let finding = model.selectedReview?.finding(source.id) {
                                        Text(finding.evidence).font(.caption)
                                        Text("\(finding.confidence.rawValue) confidence").font(.caption2)
                                    }
                                    Button("Use this source") { model.overrideRegion(source.id) }
                                        .disabled(model.busy || !source.accepted || model.output == nil ||
                                            !region.measurements.contains { $0.sourceID == source.id && $0.validFraction >= 0.9999 })
                                }
                            }
                            Button("Restore automatic selection", action: model.revertRegion).disabled(model.busy)
                        }
                    }
                    Divider()
                    HStack {
                        Button("Undo", action: model.undo).disabled(model.busy)
                        Button("Export Result…", action: model.export)
                            .disabled(model.busy || model.output == nil)
                    }
                    if model.phase == .cancelled || model.phase == .failed, model.admissionReason == nil {
                        Button("Resume review", action: model.reviewMore).disabled(model.busy)
                    }
                }.padding()
            }.frame(minWidth: 260, idealWidth: 310, maxWidth: 380)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .task { if !sources.isEmpty { model.open(sources) } }
        .onChange(of: model.regionID) { _, _ in model.loadRegion() }
        .onDisappear { model.close() }
    }
}

private struct OpenSuperMergerKey: EnvironmentKey {
    static let defaultValue: @MainActor @Sendable ([ArenaSource]) -> Void = { _ in }
}
extension EnvironmentValues {
    var openSuperMerger: @MainActor @Sendable ([ArenaSource]) -> Void {
        get { self[OpenSuperMergerKey.self] }
        set { self[OpenSuperMergerKey.self] = newValue }
    }
}
struct SuperMergerActionKey: FocusedValueKey {
    typealias Value = () -> Void
}
extension FocusedValues {
    var openSuperMerger: (() -> Void)? {
        get { self[SuperMergerActionKey.self] }
        set { self[SuperMergerActionKey.self] = newValue }
    }
}
