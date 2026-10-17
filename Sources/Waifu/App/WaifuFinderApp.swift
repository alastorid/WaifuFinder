//
//  WaifuFinderApp.swift
//  WaifuFinder — find the best frames in a long video.
//
//  This file is the single @main entry point for the whole package and is owned
//  by the app scaffold. The UI layer lives in UI/ and is presented from here;
//  nothing else in the package declares a main.
//

import SwiftUI

/// The app's own identity.
///
/// Read from the bundle rather than hardcoded so the value shown to the user and
/// the value written into `Info.plist` cannot disagree — one build, one truth.
enum WaifuFinderIdentity {
    static let bundleIdentifier = "com.alastorid.waifufinder"
    static let name = "WaifuFinder"
    static let subtitle = "Find the best frames in a long video"
}

@main
struct WaifuFinderApp: App {
    init() {
        // Tracing is off unless `WAIFU_TRACE` or `--waifu-trace` asked for it, so this
        // is one check at launch and nothing else in the app pays for it.
        if CommandLine.arguments.contains("--firefox-native-host") { FirefoxIntegration.runNativeHost() }
        Diag.bootstrapFromEnvironment()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
        // The drop target is the whole window; a document-less app needs no
        // document group, and `WindowGroup` gives it one resizable window with
        // the standard toolbar for free.
        .defaultSize(width: 1280, height: 820)
        .windowStyle(.titleBar)
        .commands { DiagCommands() }

        Settings {
            WaifuSettingsView()
        }
    }
}

/// Debug-only affordances for tracing. `Diag.isEnabled` gates the work; the item is
/// always present so a bug that only reproduces once can be caught without a rebuild.
///
/// Deliberately a real menu rather than a hidden shortcut: turning tracing on is
/// something you do *while* looking at the app, mid-bug, and a menu item cannot
/// collide with a key the app itself binds.
struct DiagCommands: Commands {
    var body: some Commands {
        CommandMenu("Debug") {
            Button(Diag.isEnabled ? "Stop Tracing" : "Start Tracing…") {
                if Diag.isEnabled {
                    Diag.disable()
                } else {
                    DiagCategoryPicker.choose()
                }
            }
            .keyboardShortcut("d", modifiers: [.command, .shift, .option])

            if Diag.isEnabled {
                Divider()
                Button("Reveal Trace File in Finder") {
                    guard let url = Diag.currentLogURL else { return }
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
                Button("Mark This Moment") {
                    Diag.mark("manual marker")
                }
            }
        }
    }
}

/// Asks which categories to trace, so a keyboard investigation can avoid drowning in
/// loading and analysis chatter.
enum DiagCategoryPicker {
    @MainActor
    static func choose() {
        let alert = NSAlert()
        alert.messageText = "Start tracing"
        alert.informativeText = "Choose what to record. Leave all off to record everything."
        for category in Diag.Category.allCases {
            alert.addButton(withTitle: category.rawValue)
        }
        alert.addButton(withTitle: "Everything")
        alert.addButton(withTitle: "Cancel")

        let response = alert.runModal().rawValue
        let titles = Diag.Category.allCases.map(\.rawValue)
        if response == NSApplication.ModalResponse.alertFirstButtonReturn.rawValue + titles.count {
            Diag.enable()
        } else if response > NSApplication.ModalResponse.alertFirstButtonReturn.rawValue - 1,
                  response <= titles.count {
            Diag.enable(selecting: titles[response - 1])
        }
    }
}
