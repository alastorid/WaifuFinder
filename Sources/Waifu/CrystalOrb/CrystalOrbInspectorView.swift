import SwiftUI

struct CrystalOrbInspectorView: View {
    @ObservedObject var model: WaifuCrystalOrbModel
    private enum Control: Hashable { case display, channel, tolerance, gain }
    @FocusState private var focusedControl: Control?
    @State private var advanced = false
    @State private var advancedMetadata = false
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("IMAGE INFORMATION").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                metadata
                if model.candidateIsPhoto { Button("Open in Photos.app") { model.revealCandidateInPhotos() } }
                Divider()
                DisclosureGroup("Comparison controls", isExpanded: $advanced) {
                    VStack(alignment: .leading, spacing: 12) {
                        analysisControls
                        CrystalOrbOpacitySlider(value: $model.opacity, focusChanged: { model.opacityControlFocused = $0 })
                        alignmentControls
                        Toggle("Overview", isOn: $model.showOverview)
                        statistics
                    }.padding(.top, 8)
                }
                DisclosureGroup("Detailed metadata", isExpanded: $advancedMetadata) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Target A").font(.headline)
                        PhotoMetadataInspectorView(metadata: model.query?.metadata, isLoading: false, errorMessage: nil)
                        Text("Candidate B").font(.headline)
                        PhotoMetadataInspectorView(metadata: model.candidateMetadata, isLoading: model.metadataLoading,
                            errorMessage: model.candidateMetadata == nil ? (model.candidateIsPhoto ? "Photo metadata is unavailable." : "Decoded video frame · Camera EXIF and frame file size unavailable") : nil)
                    }
                }.onChange(of: advancedMetadata) { _, expanded in if expanded { model.loadDetailedMetadata() } }
            }.padding(12)
        }.font(.caption).textSelection(.enabled)
            .onChange(of: focusedControl) { _, control in model.inspectionControlFocused = control != nil }
            .onDisappear { model.inspectionControlFocused = false }
            .onChange(of: model.selectedID) { _, _ in if advancedMetadata { model.loadDetailedMetadata() } }
    }
    private var analysisControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Difference").font(.headline)
            Picker("Presentation", selection: $model.differenceSettings.display) {
                ForEach(CrystalOrbDifferenceSettings.Display.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.focused($focusedControl, equals: .display)
            Picker("Measure", selection: $model.differenceSettings.channel) {
                ForEach(CrystalOrbDifferenceSettings.Channel.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.focused($focusedControl, equals: .channel)
            HStack { Text("Tolerance"); Spacer(); Text(String(format: "%.1f / 255", model.differenceSettings.tolerance)).monospacedDigit() }
            Slider(value: $model.differenceSettings.tolerance, in: 0...64, step: 0.5).focused($focusedControl, equals: .tolerance).accessibilityLabel("Difference tolerance")
            HStack { Text("Display gain"); Spacer(); Text(String(format: "%.1f×", model.differenceSettings.gain)).monospacedDigit() }
            Slider(value: $model.differenceSettings.gain, in: 1...32).focused($focusedControl, equals: .gain).accessibilityLabel("Difference display gain")
            Toggle("Extend differences to full brightness", isOn: $model.differenceSettings.autoStretch).toggleStyle(.checkbox)
            Button("Reset sensitivity") { model.differenceSettings = .init() }
            Text("Gamma-encoded sRGB · 8-bit statistics · Absolute Δ 0–1 · RGB magnitude = largest channel. Luminance uses Rec.709 luma Y′. Transparency compares over black. Magenta = unmatched; empty canvas is excluded.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }
    private var alignmentControls: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Registration").font(.headline)
            HStack {
                Button(model.registering ? "Aligning…" : "Auto Align") { model.autoAlign() }.disabled(model.alignmentLocked || model.registering || !model.pairIsCurrent)
                Button("Reset") { model.resetAlignment() }.disabled(model.alignmentLocked)
            }
            Toggle("Lock alignment", isOn: $model.alignmentLocked)
            Toggle("Apply registration", isOn: $model.alignmentEnabled)
            Toggle("Drag to align B", isOn: $model.alignmentEditing).disabled(model.alignmentLocked)
            Grid(alignment: .leading) {
                GridRow { Text("X px"); number($model.registration.x) }
                GridRow { Text("Y px"); number($model.registration.y) }
                GridRow { Text("Scale"); number($model.registration.scale) }
                GridRow { Text("Rotation °"); number($model.registration.degrees) }
            }.disabled(model.alignmentLocked)
            Text(model.alignmentMessage).font(.caption2).foregroundStyle(.secondary)
            Text("Manual rotation/scale; automatic translation only. Positive Y points upward.").font(.caption2).foregroundStyle(.secondary)
        }.toggleStyle(.checkbox)
    }
    private func number(_ binding: Binding<CGFloat>) -> some View {
        TextField("", value: Binding<Double>(get: { Double(binding.wrappedValue) }, set: { if $0.isFinite { binding.wrappedValue = CGFloat($0) } }), format: .number.precision(.fractionLength(2)))
            .textFieldStyle(.roundedBorder).frame(minWidth: 70).accessibilityLabel("Alignment value")
    }
    private var statistics: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Measurements" + (model.pair?.differenceIsOriginal == true ? "" : " · Preview estimate")).font(.headline)
            if let value = model.measurements {
                stats(value.full, label: "Full comparison")
                if let region = value.region { stats(region, label: "Selected region") }
                if !value.hotspots.isEmpty {
                    Text("Major changed regions").fontWeight(.medium)
                    ForEach(value.hotspots) { hotspot in
                        Button("\(hotspot.count.formatted()) changed · \(Int(hotspot.rect.midX)), \(Int(hotspot.rect.midY))") { model.zoomToRegion(hotspot.rect) }
                            .buttonStyle(.link).help("Center and zoom all panes on this region")
                    }
                    Text("Connected cells; conservative bounds, exact changed-pixel counts.").font(.caption2).foregroundStyle(.secondary)
                }
            } else { Text("Measurements refine in the background.").foregroundStyle(.secondary) }
            Toggle("Select region", isOn: $model.selectingRegion).toggleStyle(.checkbox)
            HStack {
                Button("Zoom to region") { if let roi = model.roi { model.zoomToRegion(roi) } }.disabled(model.roi == nil)
                Button("Clear") { model.roi = nil }.disabled(model.roi == nil)
            }
        }
    }
    private func stats(_ value: CrystalOrbStatistics, label: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).fontWeight(.medium)
            Text(String(format: "Changed %.2f%% · %@ pixels", value.percent, value.changed.formatted()))
            Text(String(format: "Raw mean Δ %.5f · Max Δ %.5f", value.mean, value.maximum))
            Text(String(format: "Overlap %.2f%% · %@ valid pixels", value.coverage, value.overlap.formatted()))
            Text("Unmatched: \((value.covered-value.overlap).formatted()) pixels").foregroundStyle(.secondary)
        }.monospacedDigit()
    }
    private var metadata: some View {
        let a = model.query?.metadata, b = model.candidateMetadata
        let rows: [(String, String?, String?)] = [
            ("Dimensions", a?.dimensionsText, (model.pairIsCurrent ? model.pair : nil).map { "\(Int($0.candidateNativeSize.width)) × \(Int($0.candidateNativeSize.height))" }),
            ("File", a?.filename, b?.filename ?? (model.candidateIsPhoto ? "Photo" : "Video frame")),
            ("Size", a?.fileSizeBytes.map(PhotoMetadataFormatter.fileSize), b?.fileSizeBytes.map(PhotoMetadataFormatter.fileSize)),
            ("Captured", a?.captureDate.map(PhotoMetadataFormatter.dateTime), b?.captureDate.map(PhotoMetadataFormatter.dateTime)),
            ("Photos date", a?.photosCreationDate.map(PhotoMetadataFormatter.dateTime), b?.photosCreationDate.map(PhotoMetadataFormatter.dateTime)),
            ("Camera", a?.cameraModel, b?.cameraModel), ("Lens", a?.lensModel, b?.lensModel),
            ("Profile", a?.colorProfile, b?.colorProfile), ("Orientation", a?.orientation, b?.orientation),
            ("Aperture", a?.aperture.map(PhotoMetadataFormatter.aperture), b?.aperture.map(PhotoMetadataFormatter.aperture)),
            ("Exposure", a?.exposureTime.map(PhotoMetadataFormatter.shutterSpeed), b?.exposureTime.map(PhotoMetadataFormatter.shutterSpeed)),
            ("ISO", a?.iso.map(String.init), b?.iso.map(String.init))
        ]
        let differences = rows.filter { $0.1 != $0.2 }
        return VStack(alignment: .leading, spacing: 8) {
            Text("Metadata differences").font(.headline)
            if differences.isEmpty { Text("No metadata differences").foregroundStyle(.secondary) }
            Grid(alignment: .leading, horizontalSpacing: 8, verticalSpacing: 7) {
                GridRow { Text(""); Text("TARGET").fontWeight(.semibold); Text("CANDIDATE").fontWeight(.semibold) }
                ForEach(Array(differences.enumerated()), id: \.offset) { _, row in
                    GridRow(alignment: .top) {
                        Text(row.0).foregroundStyle(.secondary)
                        Text(row.1 ?? "Unavailable").frame(maxWidth: .infinity, alignment: .leading)
                        Text(row.2 ?? "Unavailable").frame(maxWidth: .infinity, alignment: .leading)
                    }.padding(.vertical, 2)
                }
            }
        }
    }
}
