import AppKit
import SwiftUI

/// A compact, text-selectable inspector for the currently selected Photos still.
struct PhotoMetadataInspectorView: View {
    let metadata: PhotoMetadata?
    let isLoading: Bool
    let errorMessage: String?

    @State private var generalExpanded = true
    @State private var cameraExpanded = true
    @State private var exposureExpanded = true
    @State private var descriptionExpanded = true

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Photo Info")
                .font(.headline)
            if let metadata {
                summary(metadata)
                if isLoading {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.mini)
                        Text("Reading original details…")
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                    }
                }
                Divider()
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        DisclosureGroup("General", isExpanded: $generalExpanded) {
                            VStack(alignment: .leading, spacing: 7) {
                                row("Original filename", metadata.filename)
                                row("File type", metadata.fileType)
                                row("Native resolution", metadata.dimensionsText)
                                if !metadata.megapixelsText.isEmpty { row("Megapixels", metadata.megapixelsText) }
                                row("Original file size", metadata.fileSizeBytes.map(PhotoMetadataFormatter.fileSize) ?? unavailableSizeText(metadata))
                                if let date = metadata.captureDate {
                                    row(metadata.captureDateSource?.rawValue ?? "Capture date", PhotoMetadataFormatter.dateTime(date))
                                }
                                if let date = metadata.photosCreationDate {
                                    row("Photos asset date", PhotoMetadataFormatter.dateTime(date), tooltip: "The creation date Photos assigns to this library asset; it may differ from the embedded capture date.")
                                }
                                if let date = metadata.photosModificationDate {
                                    row("Photos asset modified", PhotoMetadataFormatter.dateTime(date), tooltip: "Photos asset modification date, not a filesystem timestamp.")
                                }
                                if let value = metadata.colorProfile { row("Color profile", value) }
                                if let value = metadata.orientation { row("Embedded orientation", value) }
                            }
                            .padding(.top, 6)
                        }

                        if metadata.hasCameraDetails {
                            DisclosureGroup("Camera & Lens", isExpanded: $cameraExpanded) {
                                VStack(alignment: .leading, spacing: 7) {
                                    if let value = metadata.cameraMake { row("Camera manufacturer", value) }
                                    if let value = metadata.cameraModel { row("Camera model", value) }
                                    if let value = metadata.lensMake { row("Lens manufacturer", value) }
                                    if let value = metadata.lensModel { row("Lens model", value) }
                                    if let value = metadata.focalLengthMM { row("Focal length", PhotoMetadataFormatter.millimeters(value)) }
                                    if let value = metadata.focalLength35MM { row("35mm equivalent", "\(value) mm") }
                                }
                                .padding(.top, 6)
                            }
                        }

                        if metadata.hasExposureDetails {
                            DisclosureGroup("Exposure", isExpanded: $exposureExpanded) {
                                VStack(alignment: .leading, spacing: 7) {
                                    if let value = metadata.aperture { row("Aperture", PhotoMetadataFormatter.aperture(value)) }
                                    if let value = metadata.exposureTime { row("Shutter speed", PhotoMetadataFormatter.shutterSpeed(value)) }
                                    if let value = metadata.iso { row("ISO sensitivity", "ISO \(value)") }
                                    if let value = metadata.exposureCompensationEV { row("Exposure compensation", PhotoMetadataFormatter.ev(value)) }
                                    if let value = metadata.exposureProgram { row("Exposure program", value) }
                                    if let value = metadata.exposureMode { row("Exposure mode", value) }
                                    if let value = metadata.meteringMode { row("Metering mode", value) }
                                    if let value = metadata.flash { row("Flash", value) }
                                    if let value = metadata.whiteBalance { row("White balance", value) }
                                }
                                .padding(.top, 6)
                            }
                        }

                        DisclosureGroup("Keywords & Descriptions", isExpanded: $descriptionExpanded) {
                            VStack(alignment: .leading, spacing: 7) {
                                row("Photos.app keywords", "Unavailable through public PhotoKit APIs",
                                    tooltip: "PhotoKit does not expose the Photos library keyword list. WaifuFinder never reads private Photos databases or substitutes AI-generated traits.")
                                if metadata.embeddedKeywords.isEmpty {
                                    row("Embedded IPTC/XMP keywords", metadata.extractionStatus == .complete ? "None recorded" : unavailableSizeText(metadata))
                                } else {
                                    row("Embedded IPTC/XMP keywords", metadata.embeddedKeywords.joined(separator: ", "))
                                }
                                if let value = metadata.title { row("Title", value) }
                                if let value = metadata.description { row("Description / caption", value) }
                            }
                            .padding(.top, 6)
                        }

                        if let message = metadata.extractionMessage {
                            Label(message, systemImage: "icloud.slash")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.top, 4)
                        }
                    }
                    .font(.system(size: 12))
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .scrollIndicators(.automatic)
                HStack {
                    Spacer()
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(metadata.copyableText(), forType: .string)
                    } label: {
                        Label("Copy Photo Info", systemImage: "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .help("Copy the available photo information as text")
                }
                .font(.system(size: 12))
            } else if isLoading {
                HStack(spacing: 9) {
                    ProgressView().controlSize(.small)
                    Text("Reading original photo information…")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
            } else {
                Text(errorMessage ?? "Photo information is unavailable.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
            }
        }
        .padding(14)
        .frame(width: 350)
        .frame(maxHeight: .infinity)
    }

    private func summary(_ metadata: PhotoMetadata) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .firstTextBaseline) {
                Text(metadata.filename)
                    .font(.system(size: 13, weight: .semibold))
                    .lineLimit(1)
                    .help(metadata.filename)
                Spacer(minLength: 6)
                Text(metadata.fileType)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Text(metadata.dimensionsText)
                Text("·")
                Text(metadata.fileSizeBytes.map(PhotoMetadataFormatter.fileSize) ?? unavailableSizeText(metadata))
            }
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(.secondary)
            if let date = metadata.captureDate ?? metadata.photosCreationDate {
                let label = if metadata.captureDate == nil { "Photos date" }
                    else if metadata.captureDateSource == .exifDigitized { "EXIF digitized" }
                    else { "Captured" }
                Text("\(label) · \(PhotoMetadataFormatter.dateTime(date))")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            HStack(alignment: .top, spacing: 5) {
                Image(systemName: "tag")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                    .padding(.top, 2)
                Text(metadata.summaryKeywordText)
                    .font(.system(size: 11))
                    .foregroundStyle(metadata.embeddedKeywords.isEmpty ? .tertiary : .secondary)
                    .lineLimit(2)
                    .textSelection(.enabled)
            }
            Text("Embedded IPTC/XMP only · Photos.app library keywords are not exposed by public PhotoKit APIs")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
        }
        .textSelection(.enabled)
    }

    private func row(_ label: String, _ value: String, tooltip: String? = nil) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 132, alignment: .leading)
            Text(value)
                .foregroundStyle(.primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .help(tooltip ?? label)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func unavailableSizeText(_ metadata: PhotoMetadata) -> String {
        switch metadata.extractionStatus {
        case .loadingOriginal: "Reading…"
        case .complete: "Unavailable"
        case .originalUnavailableLocally: "Unavailable locally"
        case .originalResourceMissing: "Unavailable"
        case .unreadableOriginal: "Unreadable original"
        }
    }
}
