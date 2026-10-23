import SwiftUI

struct CrystalOrbToolbarButton: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    let action: () -> Void
    @State private var hovering = false
    var body: some View {
        Button(action: action) {
            CrystalOrbGlyph(active: model.isOpen && (model.searching || model.importing || model.loading))
            .frame(width: 24, height: 24)
            .background((model.isOpen || hovering ? Color.accentColor.opacity(0.2) : .clear), in: RoundedRectangle(cornerRadius: 5))
        }
        .frame(width: 30).onHover { hovering = $0 }
        .help("WaifuCrystalOrb — Find Visually Similar Images")
        .accessibilityLabel("WaifuCrystalOrb — Find Visually Similar Images")
        .accessibilityValue(model.isOpen ? "Open" : "Closed")
    }
}

/// The same small orb signals ongoing search/quality refinement without a splash.
struct CrystalOrbGlyph: View {
    let active: Bool

    var body: some View {
        ZStack {
            // A little antique stand makes this read as a crystal ball rather
            // than a generic glowing dot at toolbar size.
            Ellipse().fill(CrystalOrbPalette.bronze)
                .frame(width: 15, height: 3).offset(y: 8.5)
            RoundedRectangle(cornerRadius: 1)
                .fill(CrystalOrbPalette.brass)
                .frame(width: 6, height: 2).offset(y: 7)

            Circle()
                .fill(RadialGradient(stops: [
                    .init(color: CrystalOrbPalette.lilac, location: 0),
                    .init(color: CrystalOrbPalette.amethyst, location: 0.43),
                    .init(color: CrystalOrbPalette.plum, location: 0.78),
                    .init(color: CrystalOrbPalette.deepViolet, location: 1)
                ], center: UnitPoint(x: 0.28, y: 0.22), startRadius: 0, endRadius: 16))
                .overlay(Circle().stroke(CrystalOrbPalette.brass.opacity(0.8), lineWidth: 0.8))
                .overlay {
                    CrystalOrbSwirl()
                        .stroke(CrystalOrbPalette.mist.opacity(0.7),
                                style: StrokeStyle(lineWidth: 0.8, lineCap: .round, lineJoin: .round))
                        .padding(1.5)
                }
                .frame(width: 16, height: 16)
                .offset(y: -1)
                .shadow(color: CrystalOrbPalette.mist.opacity(active ? 0.42 : 0), radius: 3)

            Image(systemName: "sparkle")
                .font(.system(size: 4.5, weight: .medium))
                .foregroundStyle(CrystalOrbPalette.brassHighlight)
                .symbolEffect(.pulse, options: .repeating, isActive: active)
                .offset(x: 4, y: -5)
        }.frame(width: 24, height: 24)
        .accessibilityLabel(active ? "Crystal Orb is working" : "Crystal Orb")
    }
}

private enum CrystalOrbPalette {
    static let lilac = Color(red: 0.70, green: 0.61, blue: 0.78)
    static let amethyst = Color(red: 0.43, green: 0.32, blue: 0.61)
    static let plum = Color(red: 0.25, green: 0.20, blue: 0.40)
    static let deepViolet = Color(red: 0.12, green: 0.13, blue: 0.26)
    static let mist = Color(red: 0.72, green: 0.72, blue: 0.86)
    static let brass = Color(red: 0.56, green: 0.40, blue: 0.22)
    static let brassHighlight = Color(red: 0.78, green: 0.63, blue: 0.37)
    static let bronze = Color(red: 0.33, green: 0.25, blue: 0.18)
}

private struct CrystalOrbSwirl: Shape {
    func path(in rect: CGRect) -> Path {
        Path { path in
            path.move(to: CGPoint(x: rect.width * 0.19, y: rect.height * 0.63))
            path.addCurve(
                to: CGPoint(x: rect.width * 0.71, y: rect.height * 0.22),
                control1: CGPoint(x: rect.width * 0.20, y: rect.height * 0.34),
                control2: CGPoint(x: rect.width * 0.52, y: rect.height * 0.15)
            )
            path.addCurve(
                to: CGPoint(x: rect.width * 0.73, y: rect.height * 0.62),
                control1: CGPoint(x: rect.width * 0.91, y: rect.height * 0.33),
                control2: CGPoint(x: rect.width * 0.91, y: rect.height * 0.55)
            )
            path.addCurve(
                to: CGPoint(x: rect.width * 0.38, y: rect.height * 0.76),
                control1: CGPoint(x: rect.width * 0.57, y: rect.height * 0.76),
                control2: CGPoint(x: rect.width * 0.47, y: rect.height * 0.84)
            )
        }
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
    @AppStorage("crystalOrb.showInspector") private var showInspector = true

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
                HSplitView {
                    CrystalOrbComparisonView(model: model).frame(minWidth: 220, maxWidth: .infinity, maxHeight: .infinity)
                    if showInspector { CrystalOrbInspectorView(model: model).frame(minWidth: 260, idealWidth: 310, maxWidth: 420) }
                }
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
            CrystalOrbGlyph(active: model.searching || model.loading || model.importing)
            Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(1).help(model.status)
            Spacer()
            Button { model.chooseFile(session: session) } label: { Image(systemName: "photo.badge.plus") }.help("Replace query image")
            if model.canLoadOriginal {
                Button("Load original") { if let id = model.selectedID { model.select(id) } }
                    .help("Load the full-resolution candidate")
            }
            if model.query != nil { Button("Search again") { model.search() }.disabled(model.searching) }
            Button { showInspector.toggle() } label: { Image(systemName: "sidebar.right") }.help("Show or hide image inspector")
            Button { model.close() } label: { Image(systemName: "xmark") }.help("Close WaifuCrystalOrb (Escape)")
                .accessibilityLabel("Close WaifuCrystalOrb")
        }.padding(.horizontal, 14).padding(.vertical, 10)
    }

    @ViewBuilder private var controls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 10) { modes; Spacer(); zoomControls }
            VStack(spacing: 7) { HStack { modes; Spacer() }; HStack { zoomControls; Spacer() } }
        }
        HStack(spacing: 10) {
            overlayControls
            Button("Auto Align") { model.autoAlign() }.disabled(model.alignmentLocked || model.registering || !model.pairIsCurrent)
            if model.mode == .inspect { Toggle("Difference pane", isOn: Binding(get: { !model.differenceCollapsed }, set: { model.differenceCollapsed = !$0 })).toggleStyle(.checkbox) }
            Toggle("Overview", isOn: $model.showOverview).toggleStyle(.checkbox)
            Spacer(minLength: 0)
        }.padding(.horizontal, 10).padding(.bottom, 6).controlSize(.small)

    }
    private var modes: some View {
        Picker("Comparison mode", selection: $model.mode) {
            ForEach(WaifuCrystalOrbModel.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
        }.pickerStyle(.segmented).labelsHidden().frame(minWidth: 340, idealWidth: 380)
    }
    private var overlayControls: some View {
        HStack(spacing: 10) {
            Toggle("Red Δ Overlay", isOn: $model.redOverlay).toggleStyle(.checkbox).disabled(model.mode == .difference)
            CrystalOrbOpacitySlider(value: $model.opacity, focusChanged: { model.opacityControlFocused = $0 })
        }.help("Red difference overlay and compositing opacity")
    }
    private var zoomControls: some View {
        HStack(spacing: 6) {
            if model.mode == .toggle {
                Button(model.showCandidate ? "Candidate" : "Target") { model.showCandidate.toggle() }.help("Toggle image · Hold Space to compare")
            }
            Button("−") { model.zoom = max(0.1, model.zoom / 1.25) }.help("Zoom out")
            Button("+") { model.zoom = min(128, model.zoom * 1.25) }.help("Zoom in")
            Button("Fit") { model.resetZoom() }
            Button("Fill") { model.fillViewport = true; model.actualPixels = false; model.zoom = 1 }
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
        if let pair = model.pair, model.pairIsCurrent { return "\(Int(pair.candidateNativeSize.width)) × \(Int(pair.candidateNativeSize.height))" }
        if let metadata = model.candidateMetadata { return metadata.dimensionsText }
        return "Preview"
    }

    private var candidates: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Top 19 candidates").font(.caption.weight(.medium))
                Spacer()
                Text("Closest first · Vision distance, lower is closer").font(.caption2).foregroundStyle(.secondary)
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
