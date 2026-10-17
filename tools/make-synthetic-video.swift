import Foundation

@main
struct SyntheticVideoCLI {
    static func main() throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard args.count == 5, let width = Int(args[1]), let height = Int(args[2]),
              let fps = Int(args[3]), let duration = Double(args[4]) else {
            throw VideoFixture.failure("Usage: make-synthetic-video.sh OUTPUT WIDTH HEIGHT FPS SECONDS")
        }
        let output = URL(fileURLWithPath: args[0])
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try VideoFixture.writeVideo(to: output, duration: duration, width: width, height: height, frameRate: fps)
        print("Created synthetic Apple-native video: \(output.path)")
    }
}
