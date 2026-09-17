import SwiftUI
import Yoob

struct ContentView: View {
    @State private var character = "luna-realistic"
    @State private var avatar = ContentView.makeAvatar("luna-realistic")

    static func makeAvatar(_ character: String) -> YoobAvatar {
        YoobAvatar(.cloud(character: character) { try await Backend.credentials(for: character) })
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            YoobAvatarView(avatar)
                .ignoresSafeArea()
                .background(Color(white: 0.08))

            VStack(spacing: 14) {
                status
                HStack(spacing: 12) {
                    Picker("Character", selection: $character) {
                        Text("Realistic").tag("luna-realistic")
                        Text("Anime").tag("luna-anime")
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 220)

                    Button(action: sayHello) {
                        Label(avatar.phase == .speaking ? "Stop" : "Say hello",
                              systemImage: avatar.phase == .speaking ? "stop.fill" : "waveform")
                            .frame(minWidth: 110)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canSpeak)
                }
            }
            .padding(16)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        #if DEBUG || YOOB_CHECKS
        .onAppear {
            if let arg = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix("--character=") }) {
                character = String(arg.dropFirst("--character=".count))
            }
        }
        #endif
        .task(id: character) {
            if avatar.character != character {
                await avatar.close()
                avatar = Self.makeAvatar(character)
            }
            try? await avatar.prepare()
            #if DEBUG || YOOB_CHECKS
            // Automated checks: `simctl launch … --say-hello` speaks once the character is ready.
            if ProcessInfo.processInfo.arguments.contains("--say-hello") { sayHello() }
            #endif
        }
    }

    private var canSpeak: Bool {
        switch avatar.phase {
        case .ready, .speaking: true
        default: false
        }
    }

    @ViewBuilder private var status: some View {
        switch avatar.phase {
        case .notPrepared, .warming:
            Label("Getting ready…", systemImage: "hourglass").font(.subheadline)
        case .downloading(let progress):
            VStack(alignment: .leading, spacing: 6) {
                Text("Downloading \(avatar.manifest?.displayName ?? "character") · \(Int(progress.fraction * 100))%")
                    .font(.subheadline)
                ProgressView(value: progress.fraction)
            }
        case .ready:
            Text("Ready. Tap Say hello.").font(.subheadline)
        case .speaking:
            Text("Speaking").font(.subheadline)
        case .failed(let error), .stopped(let error):
            VStack(spacing: 8) {
                Text(error.localizedDescription).font(.subheadline).multilineTextAlignment(.center)
                Button("Try again") { Task { try? await avatar.prepare() } }
            }
        }
    }

    private func sayHello() {
        if avatar.phase == .speaking { avatar.interrupt(); return }
        guard let url = Bundle.main.url(forResource: "hello-24k", withExtension: "pcm"),
              let pcm = try? Data(contentsOf: url) else { return }
        // Stream it in 100 ms packets, the way a realtime voice or TTS stream arrives.
        Task {
            for offset in stride(from: 0, to: pcm.count, by: 4800) {
                try avatar.speak(pcm: pcm.subdata(in: offset..<min(pcm.count, offset + 4800)), sampleRate: 24000)
                try await Task.sleep(for: .milliseconds(20))
            }
            avatar.endSpeech()
        }
    }
}
