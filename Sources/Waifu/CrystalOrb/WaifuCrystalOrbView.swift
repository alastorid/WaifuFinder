import SwiftUI

struct CrystalOrbToolbarButton: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    let action: () -> Void
    @State private var hovering = false
    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.16))
                    .overlay(Circle().stroke(.secondary.opacity(0.85), lineWidth: 1))
                    .frame(width: 16, height: 16)
                Circle()
                    .fill(.secondary.opacity(0.55))
                    .frame(width: 2, height: 2)
                    .offset(x: -3, y: -3)
            }
            .frame(width: 24, height: 24)
            .background((model.isOpen || hovering ? Color.accentColor.opacity(0.2) : .clear), in: RoundedRectangle(cornerRadius: 5))
        }
        .frame(width: 30).onHover { hovering = $0 }
        .help("WaifuCrystalOrb — Find Visually Similar Images")
        .accessibilityLabel("WaifuCrystalOrb — Find Visually Similar Images")
        .accessibilityValue(model.isOpen ? "Open" : "Closed")
    }
}

struct CrystalOrbOpacitySlider: View {
    @Binding var value: Double
    var focusChanged: (Bool) -> Void = { _ in }
    @FocusState private var focused: Bool
    var body: some View {
        HStack(spacing: 7) {
            Text("Opacity").foregroundStyle(.secondary)
            GeometryReader { geometry in
                let travel = max(1, geometry.size.width - 14)
                ZStack(alignment: .leading) {
                    Capsule().fill(.secondary.opacity(0.25)).frame(height: 3)
                    Capsule().fill(Color.accentColor.opacity(0.6)).frame(width: travel * value + 7, height: 3)
                    Circle().fill(Color(nsColor: .controlBackgroundColor)).overlay(Circle().stroke(Color.accentColor, lineWidth: 1.5))
                        .frame(width: 14, height: 14).offset(x: travel * value)
                }.frame(height: 18).contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { value = min(1, max(0, ($0.location.x - 7) / travel)) })
                    .simultaneousGesture(TapGesture(count: 2).onEnded { value = 0.65 })
            }.frame(width: 95, height: 18)
                .focusable().focused($focused).onChange(of: focused) { _, value in focusChanged(value) }.onKeyPress(.leftArrow) { value = max(0, value - 0.01); return .handled }
                .onKeyPress(.rightArrow) { value = min(1, value + 0.01); return .handled }
                .accessibilityLabel("Red overlay opacity")
                .accessibilityValue("\(Int(value * 100)) percent")
                .accessibilityAdjustableAction { direction in value = min(1, max(0, value + (direction == .increment ? 0.01 : -0.01))) }
            Text("\(Int(value * 100))%").monospacedDigit().frame(width: 32, alignment: .trailing)
        }.font(.caption).help("Red overlay opacity · Double-click to reset to 65%")
    }
}

struct WaifuCrystalOrbView: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    @ObservedObject var session: WaifuSession
    @State private var info = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if model.query == nil {
                VStack(spacing: 14) {
                    Image(systemName: "sparkle.magnifyingglass").font(.system(size: 40)).foregroundStyle(.secondary)
                    Text("Drop an image to find visually similar frames or photos.").font(.title3)
                    Text("Paste an image with ⌘V, or choose an image file.").foregroundStyle(.secondary)
                    Button("Choose image…") { model.chooseFile(session: session) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                controls
                Divider()
                CrystalOrbComparisonView(model: model)
                if let error = model.comparisonError { Text(error).font(.caption).foregroundStyle(.secondary).padding(6) }
                metadata
                Divider()
                candidates
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .transaction { $0.animation = nil }
        .onChange(of: info) { _, shown in if shown { model.loadDetailedMetadata() } }
        .onChange(of: model.selectedID) { _, _ in if info { model.loadDetailedMetadata() } }
        .task(id: model.selectedID) {
            guard let id = model.selectedID, let index = model.matches.firstIndex(where: { $0.id == id }) else { return }
            let neighbors = [index + 1, index - 1].filter { model.matches.indices.contains($0) }.map { model.matches[$0].frame }
            for frame in neighbors {
                guard !Task.isCancelled else { return }
                let identity = session.thumbnailGeneration.uuidString + ":" + frame.thumbnailKey
                if GridThumbnailLoader.shared.cachedImage(identity: identity) == nil,
                   let data = await session.thumbnailData(for: frame, retainRegeneratedThumbnail: false) {
                    _ = await GridThumbnailLoader.shared.image(data: data, identity: identity)
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text("WaifuCrystalOrb").font(.headline)
            ProgressView().controlSize(.small).frame(width: 16, height: 16)
                .opacity(model.searching || model.loading || model.importing ? 1 : 0)
            Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(1).help(model.status)
            Spacer()
            Button { model.chooseFile(session: session) } label: { Image(systemName: "photo.badge.plus") }.help("Replace query image")
            if model.query != nil { Button("Search again") { model.search() }.disabled(model.searching) }
            Button { model.close() } label: { Image(systemName: "xmark") }.help("Close WaifuCrystalOrb (Escape)")
                .accessibilityLabel("Close WaifuCrystalOrb")
        }.padding(.horizontal, 14).padding(.vertical, 10)
    }

    private var controls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 14) { modes; overlayControls; Spacer(); zoomControls }
            VStack(spacing: 7) { HStack { modes; Spacer(); zoomControls }; HStack { overlayControls; Spacer() } }
        }.padding(10).controlSize(.small)
    }
    private var modes: some View {
        Picker("Comparison mode", selection: $model.mode) {
            ForEach(WaifuCrystalOrbModel.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
        }.pickerStyle(.segmented).labelsHidden().frame(width: 320)
    }
    private var overlayControls: some View {
        HStack(spacing: 10) {
            Toggle("Red Δ Overlay", isOn: $model.redOverlay).toggleStyle(.checkbox)
            CrystalOrbOpacitySlider(value: $model.opacity, focusChanged: { model.opacityControlFocused = $0 })
        }.disabled(model.mode == .difference).help("Pixel difference intensity in red; unavailable in Difference mode")
    }
    private var zoomControls: some View {
        HStack(spacing: 6) {
            if model.mode == .toggle {
                Button(model.showCandidate ? "Candidate" : "Target") { model.showCandidate.toggle() }.help("Toggle image · Hold Space to compare")
            }
            Button("−") { model.zoom = max(0.1, model.zoom / 1.25) }.help("Zoom out")
            Button("+") { model.zoom = min(32, model.zoom * 1.25) }.help("Zoom in")
            Button("Fit") { model.resetZoom() }
            Button("1:1") { model.actualPixels = true; model.zoom = 1; model.pan = .zero }.help("One comparison pixel per display pixel · Linked zoom and pan")
        }
    }
    private var metadata: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.query?.metadata.filename ?? "Target").fontWeight(.medium)
                if let metadata = model.query?.metadata { Text("\(metadata.dimensionsText) · \(metadata.fileType)" + (metadata.fileSizeBytes.map { " · " + PhotoMetadataFormatter.fileSize($0) } ?? "")) }
            }.lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .leading, spacing: 3) {
                if let selected = model.selected {
                    Text(model.candidateMetadata?.filename ?? "Frame \(selected.id.sampleIndex)").fontWeight(.medium)
                    Text(candidateDimensions)
                    if let source = session.video?.url, PhotosMedia.shared.photoItemURL(for: source, seconds: selected.frame.timeSeconds) == nil {
                        let resolved = VideoCollection.resolve(source, seconds: selected.frame.timeSeconds)
                        Text("\(resolved.0.lastPathComponent) · \(timecode(resolved.1)) · Video frame")
                    }
                }
            }.lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
            if let frame = model.selected?.frame, let source = session.video?.url,
               let photo = PhotosMedia.shared.photoItemURL(for: source, seconds: frame.timeSeconds) {
                Button("Open in Photos.app") { PhotosAppIntegration.reveal(photo) }
            }
            Button { info.toggle() } label: { Image(systemName: "info.circle") }.help("Inspect source metadata")
                .popover(isPresented: $info) {
                    HStack(alignment: .top, spacing: 16) {
                        VStack(alignment: .leading) {
                            Text("Target").font(.headline)
                            PhotoMetadataInspectorView(metadata: model.query?.metadata, isLoading: false, errorMessage: nil)
                            if let query = model.query {
                                if let url = query.fileURL { Text(url.path).font(.caption2).textSelection(.enabled) }
                                if let date = query.fileCreated { Text("File created: " + PhotoMetadataFormatter.dateTime(date)).font(.caption2) }
                                if let date = query.fileModified { Text("File modified: " + PhotoMetadataFormatter.dateTime(date)).font(.caption2) }
                            }
                        }
                        VStack { Text("Candidate").font(.headline); PhotoMetadataInspectorView(metadata: model.candidateMetadata, isLoading: model.metadataLoading,
                            errorMessage: model.candidateMetadata == nil && !model.metadataLoading ? "Video source and timestamp are shown below the canvas. Camera EXIF is unavailable for decoded video frames." : nil) }
                    }.padding().frame(width: 700, height: 460)
                }
        }.font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            .frame(height: 48, alignment: .top).padding(10)
    }

    private var candidateDimensions: String {
        if let pair = model.pair, model.pairIsCurrent { return "\(pair.candidate.width) × \(pair.candidate.height)" }
        if let metadata = model.candidateMetadata { return metadata.dimensionsText }
        return model.loading ? "Loading original…" : "Original unavailable"
    }

    private var candidates: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Similar candidates").font(.caption.weight(.medium))
                Spacer()
                Text("Closest first · Vision distance, lower is closer").font(.caption2).foregroundStyle(.secondary)
                if model.limit < model.matches.count { Button("Show more") { model.limit += 24 }.controlSize(.mini) }
            }
            ScrollViewReader { proxy in
                ScrollView(.horizontal) {
                    LazyHStack(alignment: .top, spacing: 8) {
                        ForEach(Array(model.visibleMatches.enumerated()), id: \.element.id) { index, match in
                            CrystalOrbCandidateCell(session: session, match: match, rank: index + 1, selected: model.selectedID == match.id) { model.select(match.id) }
                                .id(match.id)
                        }
                    }.padding(3)
                }.frame(height: 108).disabled(model.importing)
                .onChange(of: model.selectedID) { _, id in if let id { proxy.scrollTo(id, anchor: .center) } }
            }
        }.padding(.horizontal, 12).padding(.vertical, 8)
    }
}

private struct CrystalOrbCandidateCell: View {
    let session: WaifuSession
    let match: CrystalOrbMatch
    let rank: Int
    let selected: Bool
    let action: () -> Void
    @StateObject private var presentation: GridThumbnailPresentation
    private var identity: String { session.thumbnailGeneration.uuidString + ":" + match.frame.thumbnailKey }
    init(session: WaifuSession, match: CrystalOrbMatch, rank: Int, selected: Bool, action: @escaping () -> Void) {
        self.session = session; self.match = match; self.rank = rank; self.selected = selected; self.action = action
        _presentation = StateObject(wrappedValue: GridThumbnailPresentation(identity: session.thumbnailGeneration.uuidString + ":" + match.frame.thumbnailKey))
    }
    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 4) {
                ZStack {
                    Color(nsColor: .controlBackgroundColor)
                    if let image = presentation.image(for: identity) {
                        Image(decorative: image, scale: 1).resizable().scaledToFit()
                    }
                }.frame(width: 100, height: 68).allowsHitTesting(false)
                Text("#\(rank) · \(String(format: "%.3f", match.distance))").font(.system(size: 10, design: .monospaced))
                Text(match.id.raw).font(.system(size: 9)).foregroundStyle(.secondary)
            }.padding(3).background(selected ? Color.accentColor.opacity(0.2) : .clear, in: RoundedRectangle(cornerRadius: 5))
                .overlay(RoundedRectangle(cornerRadius: 5).stroke(selected ? Color.accentColor : .clear, lineWidth: 2).allowsHitTesting(false))
                .contentShape(Rectangle())
        }.buttonStyle(.plain).contentShape(Rectangle())
            .accessibilityLabel("Candidate \(rank), \(match.id.raw), visual distance \(match.distance)")
            .accessibilityAddTraits(selected ? .isSelected : [])
            .task(id: identity) {
                await presentation.prepare(identity: identity) {
                    await GridThumbnailWorkPriority.shared.withVisibleWork {
                        guard let data = await session.thumbnailData(for: match.frame) else { return nil }
                        return await GridThumbnailLoader.shared.image(data: data, identity: identity)
                    }
                }
            }.onDisappear { presentation.disappear(identity: identity) }
    }
}
