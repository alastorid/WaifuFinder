import AppKit
import SwiftUI

/// The canvas, source browser and review inspector have independent, stable places.
/// Browsing never changes the reconstruction; only an explicit region choice does.
struct SuperMergerView: View {
    @StateObject private var model = ArenaStudio()
    let sources: [ArenaSource]
    let close: () -> Void
    @State private var zoom: CGFloat = 1
    @State private var center = CGPoint(x: 0.5, y: 0.5)
    @State private var scale: CGFloat = 1
    @State private var comparisonScale: CGFloat = 1
    @State private var annotations = true
    @State private var mode: DisplayMode = .result
    @State private var dropTargeted = false

    private enum DisplayMode: String, CaseIterable {
        case result = "Result", source = "Source", compare = "Compare"
    }
    private var selectedSource: ArenaAnalyzedSource? {
        model.analysis?.sources.first { $0.id == model.sourceID }
    }
    private var sourceImage: CGImage? { model.previews[model.sourceID] }
    private var canCompare: Bool { model.resultImage != nil && sourceImage != nil && selectedSource?.accepted == true }
    private var regionIndex: Int? { model.changeRegions.firstIndex { $0.id == model.regionID } }
    private var resultTitle: String {
        guard let plan = model.output?.plan else { return "Preparing result" }
        if model.busy { return "Review in progress" }
        if !plan.pendingRegionIDs.isEmpty { return "Review incomplete" }
        if !plan.unresolvedRegionIDs.isEmpty || plan.kind == .unresolved { return "Needs your review" }
        return plan.kind == .original ? "Original preserved" : "Recovered from sources"
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            HSplitView {
                VStack(spacing: 0) {
                    canvasToolbar
                    Divider()
                    canvas.frame(maxWidth: .infinity, maxHeight: .infinity)
                    if model.analysis != nil {
                        Divider()
                        sourceStrip
                    }
                }.frame(minWidth: 320, maxWidth: .infinity, maxHeight: .infinity)
                inspector.frame(minWidth: 300, idealWidth: 350, maxWidth: 440)
            }
            Divider()
            statusBar
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 12).strokeBorder(Color.accentColor, lineWidth: 3)
                    .padding(4).allowsHitTesting(false)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard !model.busy, !urls.isEmpty else { return false }
            model.open(urls.map { ArenaSource(id: $0.absoluteString, url: $0, seconds: 0) })
            return true
        } isTargeted: { dropTargeted = $0 }
        .task { if !sources.isEmpty { model.open(sources) } }
        .onChange(of: model.regionID) { _, _ in model.loadRegion() }
        .onChange(of: model.analysis?.referenceID) { _, _ in
            fit(); mode = .result
        }
        .onDisappear { model.close() }
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            Image(systemName: "square.stack.3d.up").font(.title2).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text("Super Merger").font(.headline)
                Text("One photograph. The best of every version.").font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button(model.analysis == nil ? "Choose Sources…" : "Replace Sources…", action: model.choose)
                .disabled(model.busy).help("Choose 2–16 versions of the same photograph. You can also drop files here.")
            Button(action: model.undo) { Image(systemName: "arrow.uturn.backward") }
                .disabled(!model.canUndo).help("Undo the last source choice")
                .accessibilityLabel("Undo source choice")
            Button("Export…", action: model.export)
                .buttonStyle(.borderedProminent).disabled(model.busy || model.output == nil)
                .help("Export the full-resolution result, sources and review evidence")
            Button("Done", action: close)
        }.padding(.horizontal, 16).padding(.vertical, 12)
    }

    private var canvasToolbar: some View {
        ViewThatFits(in: .horizontal) {
            HStack {
                displayModes
                Spacer(minLength: 12)
                viewportControls
            }
            VStack(alignment: .leading, spacing: 8) {
                displayModes
                HStack { viewportControls; Spacer() }
            }
        }.padding(10)
    }

    private var displayModes: some View {
        HStack(spacing: 2) {
            ForEach(DisplayMode.allCases, id: \.self) { option in
                Button { mode = option } label: {
                    Text(option.rawValue).font(.callout.weight(mode == option ? .semibold : .regular))
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(mode == option ? Color.accentColor.opacity(0.16) : Color.clear,
                                    in: RoundedRectangle(cornerRadius: 6))
                }.buttonStyle(.plain)
                    .disabled(option == .compare ? !canCompare : option == .source ? sourceImage == nil : model.resultImage == nil)
                    .accessibilityAddTraits(mode == option ? .isSelected : [])
            }
        }
    }

    private var viewportControls: some View {
        HStack(spacing: 8) {
            Button("Fit", action: fit)
            Button("100%") { zoom /= max(scale, 0.0001) }
                .help("Show preview pixels at 100%. Export retains full resolution.")
            Toggle(isOn: $annotations) { Image(systemName: "viewfinder") }
                .toggleStyle(.button).help("Show difference regions")
                .accessibilityLabel("Show difference regions")
        }.controlSize(.small).disabled(model.analysis == nil)
    }

    @ViewBuilder private var canvas: some View {
        if mode == .compare, canCompare, let result = model.resultImage, let source = sourceImage {
            HStack(spacing: 1) {
                imagePane(result, title: "Result", isSource: false, displayedScale: $scale)
                imagePane(source, title: selectedSource?.url.lastPathComponent ?? "Source", isSource: true,
                          displayedScale: $comparisonScale)
            }.background(Color(nsColor: .separatorColor))
        } else if let image = mode == .source ? sourceImage : (model.resultImage ?? sourceImage) {
            imagePane(image, title: mode == .source || model.resultImage == nil
                      ? selectedSource?.url.lastPathComponent ?? "Source" : "Result",
                      isSource: mode == .source || model.resultImage == nil, displayedScale: $scale)
        } else {
            VStack(spacing: 16) {
                Image(systemName: "photo.on.rectangle.angled").font(.system(size: 48, weight: .light))
                    .foregroundStyle(.secondary)
                Text(model.busy ? "Preparing your photographs" : "Bring your versions together")
                    .font(.title2.weight(.semibold))
                Text(model.busy ? "Sources appear here as soon as alignment is complete."
                     : "Drop 2–16 versions of the same photograph.\nSuper Merger aligns them, reviews differences and recovers a result.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 390)
                if model.busy { ProgressView().controlSize(.small) }
                else { Button("Choose Sources…", action: model.choose).buttonStyle(.borderedProminent) }
            }.padding(24).frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func imagePane(_ image: CGImage, title: String, isSource: Bool,
                           displayedScale: Binding<CGFloat>) -> some View {
        VStack(spacing: 0) {
            HStack {
                Text(title).lineLimit(1).truncationMode(.middle)
                Spacer()
                if isSource && selectedSource?.accepted == false { Text("Excluded").foregroundStyle(.orange) }
            }.font(.caption).foregroundStyle(.secondary).padding(.horizontal, 12).padding(.vertical, 7)
            ZoomableFrameImage(image: NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height)),
                frameID: FrameID(sampleIndex: isSource ? 1 : 0), zoomRatio: $zoom, center: $center,
                displayedScale: displayedScale,
                regions: isSource && selectedSource?.accepted == false ? [] : previewRegions,
                selectedRegion: model.regionID, onRegionSelection: { model.regionID = $0 })
        }.background(Color(nsColor: .underPageBackgroundColor))
    }

    private var previewRegions: [PreviewRegion] {
        guard annotations, let analysis = model.analysis else { return [] }
        return model.changeRegions.map {
            PreviewRegion(id: $0.id, rect: CGRect(x: $0.rect.x / Double(analysis.width),
                y: $0.rect.y / Double(analysis.height), width: $0.rect.width / Double(analysis.width),
                height: $0.rect.height / Double(analysis.height)))
        }
    }

    private var sourceStrip: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Sources").font(.caption.weight(.semibold))
                Text("\(model.analysis?.sources.count ?? 0)").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("Select to inspect · Compare to see both").font(.caption).foregroundStyle(.secondary)
            }
            ScrollView(.horizontal) {
                LazyHStack(spacing: 10) {
                    ForEach(model.analysis?.sources ?? []) { source in
                        Button {
                            model.clearAudit()
                            model.sourceID = source.id
                            if mode != .compare || !source.accepted { mode = .source }
                        } label: {
                            VStack(alignment: .leading, spacing: 5) {
                                ZStack(alignment: .topTrailing) {
                                    thumbnail(model.previews[source.id], height: 64)
                                    if source.id == model.output?.plan.backboneID {
                                        Image(systemName: "checkmark.seal.fill").foregroundStyle(.white, Color.accentColor)
                                            .padding(4).help("Base source used in the result")
                                    }
                                }
                                Text(source.url.lastPathComponent).font(.caption).lineLimit(1).truncationMode(.middle)
                                Text(source.accepted ? (source.id == model.output?.plan.backboneID ? "Base source" : "Aligned") : "Excluded")
                                    .font(.caption2).foregroundStyle(source.accepted ? Color.secondary : .orange)
                            }.padding(6).frame(width: 120)
                                .background(model.sourceID == source.id ? Color.accentColor.opacity(0.12) : Color.clear,
                                            in: RoundedRectangle(cornerRadius: 8))
                                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(
                                    model.sourceID == source.id ? Color.accentColor : Color.clear, lineWidth: 2))
                                .contentShape(Rectangle())
                        }.buttonStyle(.plain).help(([source.url.lastPathComponent] + source.issues).joined(separator: "\n"))
                            .accessibilityAddTraits(model.sourceID == source.id ? .isSelected : [])
                    }
                }
            }.scrollIndicators(.hidden)
        }.padding(12)
    }

    private var inspector: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Text(resultTitle).font(.headline)
                if let analysis = model.analysis {
                    Text("\(analysis.width) × \(analysis.height) · \(model.changeRegions.count) difference regions")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Review differences, then export at full resolution.").font(.callout).foregroundStyle(.secondary)
                }
                if let reason = model.admissionReason {
                    Label(reason, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
                }
            }.padding(16)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if !model.changeRegions.isEmpty {
                        regionNavigation
                        if let region = model.region { regionReview(region) }
                        else {
                            Text("Select a region on the photograph or above to compare its sources.")
                                .font(.callout).foregroundStyle(.secondary)
                        }
                    } else {
                        Label(model.analysis == nil ? "Your review will appear here" : "No difference regions detected",
                              systemImage: model.analysis == nil ? "viewfinder" : "checkmark.circle")
                            .font(.callout).foregroundStyle(.secondary)
                        if let reason = model.records.last(where: { $0.id == "overview" })?.result.review?.reason {
                            Text(reason).font(.callout)
                        }
                    }
                    if !model.excludedSources.isEmpty {
                        DisclosureGroup("Excluded sources (\(model.excludedSources.count))") {
                            ForEach(model.excludedSources) { source in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(source.url.lastPathComponent).font(.caption.bold())
                                    Text(source.issues.joined(separator: " ")).font(.caption).foregroundStyle(.secondary)
                                }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 4)
                            }
                        }
                    }
                    if model.output != nil { auditSection }
                }.padding(16)
            }
        }
    }

    private var regionNavigation: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Difference regions").font(.subheadline.weight(.semibold))
                Spacer()
                Button { moveRegion(-1) } label: { Image(systemName: "chevron.left") }
                    .disabled(regionIndex == nil || regionIndex == 0).accessibilityLabel("Previous region")
                Button { moveRegion(1) } label: { Image(systemName: "chevron.right") }
                    .disabled(regionIndex == model.changeRegions.count - 1).accessibilityLabel("Next region")
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 65))], spacing: 6) {
                ForEach(Array(model.changeRegions.enumerated()), id: \.element.id) { index, region in
                    Button { model.regionID = region.id } label: {
                        HStack(spacing: 5) {
                            Image(systemName: regionSymbol(region.id))
                            Text("\(index + 1)")
                        }.font(.callout).frame(maxWidth: .infinity).padding(.vertical, 7)
                            .background(model.regionID == region.id ? Color.accentColor.opacity(0.16) : Color.secondary.opacity(0.08),
                                        in: RoundedRectangle(cornerRadius: 6))
                    }.buttonStyle(.plain).accessibilityLabel("Region \(index + 1), \(regionStatus(region.id))")
                        .help(regionStatus(region.id)).accessibilityAddTraits(model.regionID == region.id ? .isSelected : [])
                }
            }
        }
    }

    private func regionReview(_ region: ArenaRegion) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Region \((regionIndex ?? 0) + 1)").font(.headline)
                Spacer()
                Button("Locate") { focus(region) }.controlSize(.small)
            }
            if let review = model.selectedReview?.result.review {
                if review.unresolved == true {
                    Label("No confident automatic choice", systemImage: "questionmark.circle").font(.callout).foregroundStyle(.orange)
                }
                Text(review.reason ?? "No explanation provided.").font(.callout).textSelection(.enabled)
            } else {
                Text("Waiting for review. You can inspect the aligned sources now.").font(.callout).foregroundStyle(.secondary)
            }
            if let verification = model.output?.plan.verification?.first(where: { $0.regionID == region.id }), !verification.accepted {
                Label(verification.reason, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
            }
            HStack {
                Text("Choose pixels for this region").font(.caption.weight(.semibold))
                Spacer()
                Button("Reset", action: model.revertRegion).controlSize(.small)
                    .disabled(model.busy || !model.hasRegionOverride).help("Restore automatic selection for this region")
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 135), alignment: .top)], alignment: .leading, spacing: 12) {
                ForEach(model.analysis?.sources.filter(\.accepted) ?? []) { source in
                    regionSource(source, region: region)
                }
            }
        }
    }

    private func regionSource(_ source: ArenaAnalyzedSource, region: ArenaRegion) -> some View {
        let selection = model.output?.plan.selections.last { $0.regionID == region.id }
        let used = (selection?.sourceID ?? model.output?.plan.backboneID) == source.id
        let covered = region.measurements.contains { $0.sourceID == source.id && $0.validFraction >= 0.9999 }
        return VStack(alignment: .leading, spacing: 7) {
            thumbnail(model.nativeCrops[source.id], height: 112)
            Text(source.url.lastPathComponent).font(.caption.weight(.semibold)).lineLimit(2).truncationMode(.middle)
                .help(source.url.lastPathComponent)
            if let finding = model.selectedReview?.finding(source.id) {
                Text(finding.evidence).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                Text("\(finding.confidence.rawValue.capitalized) confidence").font(.caption2).foregroundStyle(.secondary)
            }
            Button { model.overrideRegion(source.id) } label: {
                Text(used ? (selection?.manual == true ? "Used · Manual" : "Used · Auto") : "Use Source")
                    .frame(maxWidth: .infinity)
            }.controlSize(.small).tint(used ? .accentColor : .secondary)
                .disabled(model.busy || model.output == nil || !covered || used)
            if !covered { Text("Incomplete coverage").font(.caption2).foregroundStyle(.orange) }
        }.padding(8).frame(maxWidth: .infinity, alignment: .topLeading)
            .background(used ? Color.accentColor.opacity(0.07) : Color.secondary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(used ? Color.accentColor.opacity(0.6) : Color.clear))
    }

    private var auditSection: some View {
        DisclosureGroup("Output verification") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Compare the rendered result with the selected source.").font(.caption).foregroundStyle(.secondary)
                Text(selectedSource?.url.lastPathComponent ?? "Select a source").font(.caption.weight(.semibold)).lineLimit(2)
                Button(model.auditing ? "Checking…" : "Check Differences", action: model.auditOutput)
                    .disabled(model.busy || model.auditing || selectedSource?.accepted != true)
                if let difference = model.difference {
                    Text("\(difference.changedPixels.formatted()) changed pixels · \(difference.selectedFromOtherSourcePixels.formatted()) expected from source selection")
                        .font(.caption).textSelection(.enabled)
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
        }.font(.callout)
    }

    private var statusBar: some View {
        HStack(spacing: 10) {
            if model.busy { ProgressView().controlSize(.small) }
            else { Image(systemName: model.phase == .failed ? "exclamationmark.triangle" : "info.circle").foregroundStyle(.secondary) }
            Text(model.status).font(.caption).lineLimit(2).textSelection(.enabled).help(model.status)
            Spacer(minLength: 8)
            if model.busy { Button("Cancel", action: model.cancel).controlSize(.small) }
            else if (model.phase == .cancelled || model.phase == .failed), model.admissionReason == nil, model.analysis != nil {
                Button("Resume Review", action: model.reviewMore).controlSize(.small)
            }
        }.padding(.horizontal, 16).padding(.vertical, 9)
    }

    private func thumbnail(_ image: CGImage?, height: CGFloat) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 5).fill(Color(nsColor: .underPageBackgroundColor))
            if let image { Image(decorative: image, scale: 1).resizable().scaledToFit() }
            else { Image(systemName: "photo").foregroundStyle(.tertiary) }
        }.frame(height: height).clipped()
    }
    private func fit() { zoom = 1; center = CGPoint(x: 0.5, y: 0.5) }
    private func moveRegion(_ offset: Int) {
        let index = (regionIndex ?? -1) + offset
        guard model.changeRegions.indices.contains(index) else { return }
        model.regionID = model.changeRegions[index].id
    }
    private func focus(_ region: ArenaRegion) {
        guard let analysis = model.analysis else { return }
        if selectedSource?.accepted == false { mode = .result }
        center = CGPoint(x: (region.rect.x + region.rect.width / 2) / Double(analysis.width),
                         y: (region.rect.y + region.rect.height / 2) / Double(analysis.height))
        zoom = CGFloat(min(12, max(1, min(Double(analysis.width) / max(region.rect.width, 1),
                                          Double(analysis.height) / max(region.rect.height, 1)) * 0.65)))
    }
    private func regionStatus(_ id: String) -> String {
        guard let plan = model.output?.plan else { return "Waiting for review" }
        if plan.pendingRegionIDs.contains(id) { return "Waiting for review" }
        if plan.unresolvedRegionIDs.contains(id) { return "Needs review" }
        if plan.selections.contains(where: { $0.regionID == id && $0.manual == true }) { return "Manual choice" }
        return "Automatic choice"
    }
    private func regionSymbol(_ id: String) -> String {
        switch regionStatus(id) {
        case "Waiting for review": return "clock"
        case "Needs review": return "questionmark.circle"
        case "Manual choice": return "hand.point.up.left"
        default: return "checkmark.circle"
        }
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
