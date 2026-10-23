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
    @AppStorage("rightPanelExpanded") private var showInspector = true
    @State private var candidatesExpanded = false
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
                controls
                Divider()
                HSplitView {
                    CrystalOrbComparisonView(model: model).frame(minWidth: 220, maxWidth: .infinity, maxHeight: .infinity)
                    if showInspector { CrystalOrbInspectorView(model: model).frame(minWidth: 260, idealWidth: 310, maxWidth: 420) }
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

    private var controls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) { modes; Spacer(minLength: 4); compactActions }
            VStack(spacing: 4) { HStack { modes; Spacer() }; HStack { compactActions; Spacer() } }
        }.padding(.horizontal, 8).padding(.vertical, 5).controlSize(.small)
    }
    private var compactActions: some View {
        HStack(spacing: 6) {
            Button("Fit") { model.resetZoom() }
            Button("1:1") { model.actualPixels = true; model.zoom = 1 }
            if model.mode != .difference { Toggle("Δ Overlay", isOn: $model.redOverlay).toggleStyle(.checkbox) }
            Button { candidatesExpanded.toggle() } label: { Label("Candidates", systemImage: candidatesExpanded ? "rectangle.bottomhalf.inset.filled" : "rectangle.bottomhalf.inset.filled") }
                .labelStyle(.iconOnly).help(candidatesExpanded ? "Hide candidates" : "Show top 7 candidates")
                .accessibilityLabel(candidatesExpanded ? "Hide candidates" : "Show top 7 candidates")
        }
    }
    private var modes: some View {
        HStack(spacing: 2) {
            ForEach(WaifuCrystalOrbModel.Mode.allCases, id: \.self) { mode in
                Button {
                    if mode == .toggle, model.mode == .toggle { model.showCandidate.toggle() }
                    else { model.mode = mode }
                    model.maximizedPane = nil
                } label: {
                    Text(mode == .toggle && model.mode == .toggle ? "Toggle · " + (model.showCandidate ? "Candidate" : "Target") : mode.rawValue)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(model.mode == mode ? Color.accentColor.opacity(0.2) : .clear, in: RoundedRectangle(cornerRadius: 4))
                }.buttonStyle(.plain)
                    .accessibilityAddTraits(model.mode == mode ? .isSelected : [])
                    .help(mode == .toggle ? "Click Toggle again to switch images · Hold Space to compare" : mode.rawValue)
            }
        }
    }
    private var candidates: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Top 7 candidates").font(.caption.weight(.medium))
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

/// Reuses the title-bar keyword-job slot; no second header takes image space.
struct CrystalOrbSearchBadge: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    var body: some View {
        Text(model.searching || model.importing ? "Finding matches…" : (model.loading ? "Refining photo…" : model.status))
            .font(.system(size: 12)).lineLimit(1).truncationMode(.tail)
            .padding(.horizontal, 9).padding(.vertical, 5)
            .background(.quaternary, in: Capsule()).frame(maxWidth: 260)
            .help(model.status)
    }
}
