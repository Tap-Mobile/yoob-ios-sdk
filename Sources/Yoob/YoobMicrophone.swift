import Foundation
import Observation
@preconcurrency import AVFoundation

/// The user's microphone, captured with the system echo canceller so the character's own voice is removed.
/// Get it from `avatar.microphone`.
@MainActor @Observable
public final class YoobMicrophone {
    public enum State: Equatable, Sendable {
        case off, starting, live, muted
        case failed(YoobError)
    }

    public struct Input: Identifiable, Hashable, Sendable {
        public let id: String
        public let name: String
    }

    public private(set) var state: State = .off
    /// Input level from 0 to 1, updated about 20 times a second. 0 while muted.
    public private(set) var level: Double = 0
    /// Microphones the system offers (built-in, wired, Bluetooth). Refreshed when routes change.
    public private(set) var inputs: [Input] = []
    /// The chosen input, or nil for the system's choice.
    public private(set) var selectedInputID: String?
    public var isMuted: Bool { state == .muted }

    /// 24 kHz mono PCM16, called on an audio thread. Nothing is delivered while muted.
    @ObservationIgnored public var onAudio: (@Sendable (Data) -> Void)? {
        didSet { gate.set(onAudio: onAudio, open: state == .live) }
    }

    @ObservationIgnored private let player: SpeechPlayer
    @ObservationIgnored private let gate = CaptureGate()
    @ObservationIgnored private var routeObserver: NSObjectProtocol?

    init(player: SpeechPlayer) {
        self.player = player
        player.onCaptureFailed = { [weak self] message in
            Task { @MainActor in self?.fail(.renderer(message)) }
        }
    }

    /// Asks for microphone permission if needed and starts capturing.
    public func start(inputID: String? = nil) async throws {
        if state == .live || state == .muted { return }
        state = .starting
        guard await Self.requestPermission() else {
            throw fail(.permissionDenied("Microphone access is off. Turn it on in Settings › Privacy & Security › Microphone."))
        }
        let gate = self.gate
        gate.set(onAudio: onAudio, open: true)
        do {
            try player.startCapture { [weak self] pcm, rms in
                guard gate.deliver(pcm) else { return }
                let level = min(1, rms * 4)
                if gate.shouldReportLevel() { Task { @MainActor in self?.level = level } }
            }
        } catch {
            throw fail((error as? YoobError) ?? .renderer(error.localizedDescription))
        }
        refreshInputs()
        observeRoutes()
        if let inputID { try select(inputID: inputID) }
        state = .live
    }

    /// Switches to another input without stopping a conversation. Pass nil for the system's choice.
    public func select(inputID: String?) throws {
        selectedInputID = inputID
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        let port = inputID.flatMap { id in session.availableInputs?.first { $0.uid == id } }
        if inputID != nil, port == nil { throw YoobError.unsupported("that microphone is no longer connected") }
        do { try session.setPreferredInput(port) }
        catch { throw YoobError.renderer("couldn't switch microphone: \(error.localizedDescription)") }
        #endif
    }

    public func setMuted(_ muted: Bool) {
        guard state == .live || state == .muted else { return }
        player.setInputMuted(muted)
        gate.set(onAudio: onAudio, open: !muted)
        if muted { level = 0 }
        state = muted ? .muted : .live
    }

    /// Stops capturing; the microphone indicator turns off.
    public func stop() {
        guard state != .off else { return }
        gate.set(onAudio: nil, open: false)
        player.stopCapture()
        if let routeObserver { NotificationCenter.default.removeObserver(routeObserver) }
        routeObserver = nil
        level = 0
        state = .off
    }

    private func refreshInputs() {
        #if os(iOS)
        inputs = (AVAudioSession.sharedInstance().availableInputs ?? []).map { Input(id: $0.uid, name: $0.portName) }
        if let selected = selectedInputID, !inputs.contains(where: { $0.id == selected }) { selectedInputID = nil }
        #endif
    }

    private func observeRoutes() {
        #if os(iOS)
        guard routeObserver == nil else { return }
        routeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.refreshInputs() }
        }
        #endif
    }

    @discardableResult
    private func fail(_ error: YoobError) -> YoobError {
        gate.set(onAudio: nil, open: false)
        level = 0
        state = .failed(error)
        return error
    }

    private static func requestPermission() async -> Bool {
        #if os(iOS)
        if #available(iOS 17.0, *) {
            switch AVAudioApplication.shared.recordPermission {
            case .granted: return true
            case .denied: return false
            default: return await AVAudioApplication.requestRecordPermission()
            }
        }
        return false
        #else
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
        #endif
    }
}

/// Audio-thread side of the microphone: whether to deliver packets, and a throttle for level updates.
final class CaptureGate: @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable (Data) -> Void)?
    private var open = false
    private var lastLevel = ContinuousClock.now

    func set(onAudio: (@Sendable (Data) -> Void)?, open: Bool) {
        lock.lock(); handler = onAudio; self.open = open; lock.unlock()
    }

    func deliver(_ pcm: Data) -> Bool {
        lock.lock(); let handler = open ? handler : nil; let open = self.open; lock.unlock()
        handler?(pcm)
        return open
    }

    func shouldReportLevel() -> Bool {
        lock.lock(); defer { lock.unlock() }
        let now = ContinuousClock.now
        guard lastLevel.duration(to: now) >= .milliseconds(50) else { return false }
        lastLevel = now
        return true
    }
}
