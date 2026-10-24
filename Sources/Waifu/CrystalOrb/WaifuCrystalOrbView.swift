import SwiftUI

struct CrystalOrbToolbarButton: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    let action: () -> Void
    @State private var hovering = false
    var body: some View {
        Button(action: action) {
            CrystalOrbGlyph(active: model.isOpen && (model.searching || model.importing || model.loading))
                .frame(width: 28, height: 20)
                .background((model.isOpen || hovering ? Color.accentColor.opacity(0.2) : .clear), in: RoundedRectangle(cornerRadius: 5))
        }
        .frame(width: 30).onHover { hovering = $0 }
        .accessibilityLabel("WaifuCrystalOrb — Find Visually Similar Images")
        .accessibilityValue(model.isOpen ? "Open" : "Closed")
    }
}

/// A small glass sphere at SF Symbol scale: bright anime-style glass with a
/// crisp specular highlight and a lower rim light, plus a sparkle that pulses
/// while the orb works.
struct CrystalOrbGlyph: View {
    let active: Bool

    var body: some View {
        ZStack {
            Circle()
                .fill(LinearGradient(
                    stops: [
                        .init(color: Color(red: 0.93, green: 0.98, blue: 1.00), location: 0.0),
                        .init(color: Color(red: 0.63, green: 0.87, blue: 0.99), location: 0.40),
                        .init(color: Color(red: 0.30, green: 0.61, blue: 0.96), location: 0.78),
                        .init(color: Color(red: 0.09, green: 0.27, blue: 0.60), location: 1.0)
                    ],
                    startPoint: UnitPoint(x: 0.25, y: 0), endPoint: UnitPoint(x: 0.75, y: 1)))
            // The rim light along the lower curve is what makes a sphere read as glass.
            Ellipse()
                .fill(Color.white.opacity(0.55))
                .blur(radius: 1.2)
                .frame(width: 12, height: 4.5)
                .offset(y: -0.8)
            Ellipse()
                .fill(Color.white)
                .frame(width: 4.6, height: 3.2)
                .rotationEffect(.degrees(15))
                .offset(x: -3.0, y: -4.3)
        }
        .frame(width: 17, height: 17)
        .clipShape(Circle())
        .overlay(
            Circle().stroke(
                LinearGradient(colors: [Color.white.opacity(0.9), Color.white.opacity(0.05)],
                               startPoint: .top, endPoint: .bottom),
                lineWidth: 0.8))
        .shadow(color: Color(red: 0.55, green: 0.80, blue: 1.0).opacity(active ? 0.55 : 0), radius: 2.5)
        .overlay(alignment: .topTrailing) {
            Image(systemName: "sparkle")
                .font(.system(size: 4.5, weight: .medium))
                .foregroundStyle(Color(red: 1.0, green: 0.98, blue: 0.88))
                .symbolEffect(.pulse, options: .repeating, isActive: active)
                .offset(x: 3.4, y: -2.6)
        }
        .accessibilityLabel(active ? "Crystal Orb is working" : "Crystal Orb")
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
        }.font(.caption)
    }
}

struct WaifuCrystalOrbView: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    @ObservedObject var session: WaifuSession
    @AppStorage("rightPanelExpanded") private var showInspector = true
    @State private var candidatesExpanded = false
    @State private var candidateStripWidth: CGFloat = 720
    /// Thirteen tiles plus twelve 1px gaps and 1px content padding.
    private var tileSide: CGFloat {
        min(96, max(20, ((candidateStripWidth - 14) / 13).rounded(.down)))
    }
    @AppStorage("photosDownloadFromICloud") private var downloadFromICloud = false

    var body: some View {
        VStack(spacing: 0) {
            if model.query == nil {
                VStack(spacing: 14) {
                    Image(systemName: "sparkle.magnifyingglass").font(.system(size: 40)).foregroundStyle(.secondary)
                    Text("Drop an image to find visually similar frames or photos.").font(.title3)
                    Text("Drop or paste an image with ⌘V.").foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HSplitView {
                    CrystalOrbComparisonView(model: model, session: session).frame(minWidth: 220, maxWidth: .infinity, maxHeight: .infinity)
                    if showInspector {
                        VStack(spacing: 0) {
                            inspectorHeader
                            Divider()
                            orbControls
                            Divider()
                            CrystalOrbInspectorView(model: model)
                        }
                        .frame(minWidth: 260, idealWidth: 310, maxWidth: 420)
                    }
                }
                if let error = model.comparisonError { Text(error).font(.caption).foregroundStyle(.secondary).padding(6) }
                if candidatesExpanded { Divider(); candidates }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .transaction { $0.animation = nil }
        .onChange(of: downloadFromICloud) { _, enabled in if enabled { model.originalAccessChanged() } }
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

    /// Status at the top of the panel. The title bar's back button owns leaving
    /// the orb, so the panel carries no close control of its own.
    private var inspectorHeader: some View {
        HStack(spacing: 8) {
            if !session.isWritingPhotosTags { CrystalOrbSearchBadge(model: model) }
            else { PhotosKeywordJobView(status: session.photosKeywordJob, cancel: session.cancelPhotosKeywordJob) }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// The comparison controls that used to sit above the image, merged into
    /// the panel: the four modes, then the view actions beneath them.
    private var orbControls: some View {
        VStack(spacing: 8) {
            modes
            compactActions
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .controlSize(.small)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var compactActions: some View {
        HStack(spacing: 6) {
            Button("Fit") { model.resetZoom() }
            Button("1:1") { model.actualPixels = true; model.zoom = 1 }
            if model.mode != .difference { Toggle("Δ Overlay", isOn: $model.redOverlay).toggleStyle(.checkbox) }
            Button { candidatesExpanded.toggle() } label: { Label("Candidates", systemImage: candidatesExpanded ? "rectangle.bottomhalf.inset.filled" : "rectangle.bottomhalf.inset.filled") }
                .labelStyle(.iconOnly)
                .accessibilityLabel(candidatesExpanded ? "Hide candidates" : "Show candidates")
        }
    }
    private var modes: some View {
        LazyVGrid(columns: [GridItem(.flexible(), spacing: 4), GridItem(.flexible(), spacing: 4)], spacing: 4) {
            ForEach(WaifuCrystalOrbModel.Mode.standardModes, id: \.self) { mode in
                Button {
                    if mode == .toggle, model.mode == .toggle { model.showCandidate.toggle() }
                    else { model.mode = mode }
                    model.maximizedPane = nil
                } label: {
                    Text(mode == .toggle && model.mode == .toggle ? "Toggle · " + (model.showCandidate ? "Candidate" : "Target") : mode.rawValue)
                        .lineLimit(1).minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                        .background(model.mode == mode ? Color.accentColor.opacity(0.2) : .clear, in: RoundedRectangle(cornerRadius: 4))
                }.buttonStyle(.plain)
                    .accessibilityAddTraits(model.mode == mode ? .isSelected : [])
            }
        }
    }
    private var candidates: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal) {
                LazyHStack(spacing: 1) {
                    ForEach(Array(model.visibleMatches.enumerated()), id: \.element.id) { index, match in
                        CrystalOrbCandidateCell(model: model, session: session, match: match, rank: index + 1, side: tileSide,
                            selected: model.selectedID == match.id) { model.select(match.id) }
                            .id(match.id)
                    }
                }.padding(1)
            }.disabled(model.importing)
            .onChange(of: model.selectedID) { _, id in if let id { proxy.scrollTo(id, anchor: .center) } }
            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { candidateStripWidth = $0 }
        }
        .frame(height: tileSide + 2)
        .padding(.horizontal, 8).padding(.vertical, 5)
    }
}

private struct CrystalOrbCandidateCell: View {
    let model: WaifuCrystalOrbModel
    let session: WaifuSession
    let match: CrystalOrbMatch
    let rank: Int
    let side: CGFloat
    let selected: Bool
    let action: () -> Void
    @AppStorage(GoogleLensSearch.enabledKey) private var googleLensEnabled = false
    @StateObject private var presentation: GridThumbnailPresentation
    private var identity: String { session.thumbnailGeneration.uuidString + ":" + match.frame.thumbnailKey }
    init(model: WaifuCrystalOrbModel, session: WaifuSession, match: CrystalOrbMatch, rank: Int, side: CGFloat, selected: Bool, action: @escaping () -> Void) {
        self.model = model; self.session = session; self.match = match; self.rank = rank; self.side = side; self.selected = selected; self.action = action
        _presentation = StateObject(wrappedValue: GridThumbnailPresentation(identity: session.thumbnailGeneration.uuidString + ":" + match.frame.thumbnailKey))
    }
    var body: some View {
        Button(action: action) {
            ZStack {
                Color(nsColor: .controlBackgroundColor)
                if let image = presentation.image(for: identity) {
                    Image(decorative: image, scale: 1).resizable().scaledToFit()
                }
            }.frame(width: side, height: side)
                .overlay(Rectangle().inset(by: 1).stroke(selected ? Color.accentColor : .clear, lineWidth: 2))
                .contentShape(Rectangle())
        }.buttonStyle(.plain)
            .contextMenu(isEnabled: googleLensEnabled || model.photoURL(for: match) != nil) {
                if googleLensEnabled {
                    Button("Search with Google Lens") {
                        Task { await GoogleLensSearch.search(frame: match.frame, in: session) }
                    }
                }
                if model.photoURL(for: match) != nil {
                    if googleLensEnabled { Divider() }
                    Button("Open in Photos.app") { model.revealCandidateInPhotos(match) }
                }
            }
            .help(String(format: "#%d · %.3f", rank, match.distance))
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

/// Reuses the title-bar keyword-job slot; no second header takes image space.
/// Occupies the slot only while work is running or a message needs attention.
struct CrystalOrbSearchBadge: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    var body: some View {
        if let text = badgeText {
            Text(text)
                .font(.system(size: 12)).lineLimit(1).truncationMode(.tail)
                .padding(.horizontal, 9).padding(.vertical, 5)
                .background(.quaternary, in: Capsule()).frame(maxWidth: 260)
        }
    }

    private var badgeText: String? {
        if model.searching || model.importing { return "Finding matches…" }
        if model.loading { return "Refining photo…" }
        return model.status
    }
}
