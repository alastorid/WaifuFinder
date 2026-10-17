import SwiftUI

struct FirefoxSettingsView: View {
    @AppStorage("firefoxSuffixes") private var suffixes = FirefoxIntegration.defaults
    @State private var registered = FirefoxIntegration.registered
    @State private var healthy = FirefoxIntegration.isHealthy
    @State private var message = ""

    private var statusTitle: String {
        if healthy { return "Working · Healthy" }
        return registered ? "Not connected" : "Not set up"
    }

    private var statusDetail: String {
        if healthy { return "Firefox is connected. Matching media links will open in WaifuFinder." }
        return registered
            ? "The app bridge is ready. Set up or open the add-on in Firefox to connect."
            : "Set up Firefox to install the add-on and connect the app bridge."
    }

    var body: some View {
        Form {
            Section("Firefox media links") {
                HStack(alignment: .top, spacing: 12) {
                    Circle()
                        .fill(healthy ? Color.green : Color.gray)
                        .frame(width: 12, height: 12)
                        .padding(.top, 4)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(statusTitle).font(.headline)
                        Text(statusDetail).foregroundStyle(.secondary)
                    }
                }

                TextField("URL suffixes", text: $suffixes)
                Text("Separate suffixes with commas. Matching applies to direct links opened in a tab, including query parameters. Embedded videos stay in Firefox.")
                    .foregroundStyle(.secondary)

                HStack {
                    Button("Set Up / Update Firefox") { setUpFirefox() }
                        .keyboardShortcut(.defaultAction)
                    Button("Unregister") { perform { try FirefoxIntegration.unregister() } }
                        .disabled(!registered)
                }

                Text("Set Up saves your suffixes, opens Firefox’s add-on loader, and highlights the WaifuFinder add-on file. In Firefox, choose ‘Load Temporary Add-on…’ and select that file. Firefox requires Mozilla signing for permanent installs; local installs must be repeated after Firefox restarts.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if !message.isEmpty {
                    Text(message).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .task {
            while !Task.isCancelled {
                refreshStatus()
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    private func setUpFirefox() {
        perform {
            try FirefoxIntegration.register(suffixText: suffixes)
            if !FirefoxIntegration.isHealthy {
                try FirefoxIntegration.openExtensionSetup()
            }
        }
    }

    private func perform(_ action: () throws -> Void) {
        do { try action(); message = "" }
        catch { message = error.localizedDescription }
        refreshStatus()
    }

    private func refreshStatus() {
        registered = FirefoxIntegration.registered
        healthy = FirefoxIntegration.isHealthy
    }
}
