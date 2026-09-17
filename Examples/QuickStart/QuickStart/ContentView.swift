import SwiftUI
import Yoob

struct ContentView: View {
    @State private var character = "luna-realistic"
    @State private var avatar = ContentView.makeAvatar("luna-realistic")
    @State private var conversation: YoobConversation?

    static func makeAvatar(_ character: String) -> YoobAvatar {
        YoobAvatar(.cloud(character: character) { try await Backend.credentials(for: character) })
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            YoobAvatarView(avatar)
                .ignoresSafeArea()
                .background(Color(white: 0.08))

            VStack(spacing: 12) {
                captions
                status
                HStack(spacing: 10) {
                    Button(action: toggleTalk) {
                        Label(isTalking ? "End" : "Talk", systemImage: isTalking ? "phone.down.fill" : "mic.fill")
                            .frame(minWidth: 84)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(isTalking ? .red : .accentColor)
                    .disabled(!canSpeak && !isTalking)

                    if isTalking { MicrophoneControls(microphone: avatar.microphone) }

                    Spacer(minLength: 0)
                    Button(action: sayHello) {
                        Image(systemName: avatar.phase == .speaking && !isTalking ? "stop.fill" : "waveform")
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Play sample")
                    .disabled(!canSpeak || isTalking)
                }
                Picker("Character", selection: $character) {
                    Text("Realistic").tag("luna-realistic")
                    Text("Anime").tag("luna-anime")
                }
                .pickerStyle(.segmented)
                .disabled(isTalking)
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

    private var isTalking: Bool {
        guard let state = conversation?.state else { return false }
        return state != .idle && state != .ended
    }

    private var canSpeak: Bool {
        switch avatar.phase {
        case .ready, .speaking: true
        default: false
        }
    }

    @ViewBuilder private var captions: some View {
        if let conversation, isTalking {
            VStack(alignment: .leading, spacing: 4) {
                if !conversation.assistantTranscript.isEmpty {
                    Text(conversation.assistantTranscript).font(.callout).lineLimit(3)
                }
                if !conversation.userTranscript.isEmpty {
                    Text(conversation.userTranscript).font(.callout).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .trailing).lineLimit(2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder private var status: some View {
        if let conversation, isTalking {
            Text(conversationText(conversation.state)).font(.subheadline)
        } else {
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
                Text(conversation?.lastError?.localizedDescription ?? "Ready. Tap Talk.").font(.subheadline)
            case .speaking:
                Text("Speaking").font(.subheadline)
            case .failed(let error), .stopped(let error):
                VStack(spacing: 8) {
                    Text(error.localizedDescription).font(.subheadline).multilineTextAlignment(.center)
                    Button("Try again") { Task { try? await avatar.prepare() } }
                }
            }
        }
    }

    private func conversationText(_ state: YoobConversation.State) -> String {
        switch state {
        case .connecting: "Connecting…"
        case .listening: "Listening. Just speak."
        case .thinking: "Thinking…"
        case .speaking: "Speaking. Talk to interrupt."
        case .idle, .ended: ""
        }
    }

    private func toggleTalk() {
        if isTalking { conversation?.stop(); return }
        var options = YoobConversation.Options()
        options.voice = "marin"
        options.instructions = "You are Luna, a warm, curious companion. Keep replies short and natural."
        options.greet = true
        let conversation = YoobConversation(avatar: avatar, options: options, clientSecret: Backend.openAIClientSecret)
        self.conversation = conversation
        Task { try? await conversation.start() }
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

/// Mute, input level and input choice for a live conversation.
struct MicrophoneControls: View {
    let microphone: YoobMicrophone

    var body: some View {
        HStack(spacing: 8) {
            Button { microphone.setMuted(!microphone.isMuted) } label: {
                Image(systemName: microphone.isMuted ? "mic.slash.fill" : "mic.fill")
            }
            .buttonStyle(.bordered)
            .tint(microphone.isMuted ? .red : nil)
            .accessibilityLabel(microphone.isMuted ? "Unmute" : "Mute")

            Gauge(value: microphone.level) { EmptyView() }
                .gaugeStyle(.accessoryLinearCapacity)
                .frame(width: 56)
                .accessibilityLabel("Microphone level")

            if microphone.inputs.count > 1 {
                Menu {
                    Button("Automatic") { try? microphone.select(inputID: nil) }
                    ForEach(microphone.inputs) { input in
                        Button(input.name) { try? microphone.select(inputID: input.id) }
                    }
                } label: {
                    Image(systemName: "chevron.up.chevron.down")
                }
                .accessibilityLabel("Choose microphone")
            }
        }
    }
}
