import Foundation
import Yoob

/// What the segmenter drives. `YoobAvatar` in the app; a fake in tests.
@MainActor
protocol AvatarAudioSink: AnyObject {
    func appendAudio(pcm: Data, sampleRate: Int) throws
    func audioPlayed(samples: Int)
    func endSpeech()
    @discardableResult func interrupt() -> Int
}

extension YoobAvatar: AvatarAudioSink {}

/// One converted buffer from the agent's audio track, stamped when LiveKit handed it to the renderer.
struct AgentAudioChunk: Sendable {
    let pcm: Data
    /// RMS, 0–1.
    let level: Double
    let arrival: ContinuousClock.Instant

    var samples: Int { pcm.count / 2 }
    var duration: Duration { .microseconds(Int64(samples) * 1_000_000 / Int64(PCMConverter.outputRate)) }
}

/// Splits the agent's continuous audio track into utterances for the avatar and reports how much of each the
/// listener has heard.
///
/// **Segmenting.** With the agent's `lk.agent.state` attribute: an utterance starts at the first voiced audio while
/// the agent is `speaking` (leading silence is dropped; up to `preRoll` of audio that arrived just before the attribute
/// is kept). When the agent leaves `speaking`, audio keeps flowing to the avatar until it goes silent or `hangover`
/// passes. If the agent's voice was still sounding at that moment it was cut off (a barge-in), and the avatar is
/// interrupted; otherwise the utterance ends normally. Without the attribute, a silence gate is used: voice starts an
/// utterance and `silenceGate` without voice ends it (the trailing silence is not sent).
///
/// **Timing.** LiveKit renders a buffer when the audio device pulls it for playout, so the listener hears it about
/// `latency` later (the output latency plus the I/O buffer). Each buffer is scheduled at
/// `max(arrival + latency, end of the previous buffer)`, and the heard count advances through it in real time.
@MainActor
final class AgentSpeechSegmenter {
    struct Configuration {
        var voiceThreshold = 0.005
        var silenceGate: Duration = .milliseconds(600)
        var hangover: Duration = .milliseconds(400)
        var preRoll: Duration = .milliseconds(200)
    }

    private struct Scheduled {
        let start: Int
        let count: Int
        let playAt: ContinuousClock.Instant
        var end: ContinuousClock.Instant { playAt + .microseconds(Int64(count) * 1_000_000 / Int64(PCMConverter.outputRate)) }
    }

    private let sink: AvatarAudioSink
    var configuration: Configuration
    /// Delay between a buffer reaching the renderer and the listener hearing it. Read when an utterance starts.
    var latency: () -> Duration
    /// Called when an utterance starts (true) or is handed back to the avatar (false).
    var onActiveChanged: ((Bool) -> Void)?

    /// The agent's `lk.agent.state`, or nil when the agent does not publish it (silence-gate mode).
    private(set) var agentState: String?
    /// Audio is being sent to the avatar for an utterance.
    private(set) var isActive = false
    private var preRoll: [AgentAudioChunk] = []
    private var appended = 0
    private var schedule: [Scheduled] = []
    private var heardBase = 0
    private var reported = 0
    private var utteranceLatency: Duration = .zero
    /// endSpeech was called; heard samples are still reported until the avatar has played everything.
    private var awaitingPlayout = false
    private var lastChunkEnd: ContinuousClock.Instant?
    private var lastVoiceEnd: ContinuousClock.Instant?
    private var draining: (deadline: ContinuousClock.Instant, cutOff: Bool)?

    init(sink: AvatarAudioSink, configuration: Configuration = .init(), latency: @escaping () -> Duration = { .zero }) {
        self.sink = sink
        self.configuration = configuration
        self.latency = latency
    }

    // MARK: - Input

    func setAgentState(_ state: String?, at now: ContinuousClock.Instant) {
        let old = agentState
        agentState = state
        guard state != old else { return }
        if state == "speaking" {
            draining = nil
            if !isActive, preRoll.contains(where: isVoiced) { begin(with: []) }
        } else if state != nil, isActive, draining == nil {
            draining = (now + configuration.hangover, isSounding)
        } else if state == nil {
            draining = nil
        }
    }

    func receive(_ chunk: AgentAudioChunk) {
        let voiced = isVoiced(chunk)
        if let draining {
            // The agent stopped speaking; take what is still arriving until it goes quiet.
            if voiced, chunk.arrival < draining.deadline { append(chunk) } else { finish() }
        } else if isActive {
            if agentState == nil {
                if voiced {
                    append(chunk)
                } else if let lastVoiceEnd, chunk.arrival + chunk.duration - lastVoiceEnd >= configuration.silenceGate {
                    finish()
                } else {
                    heldSilence.append(chunk)
                }
            } else {
                append(chunk)
            }
        } else if voiced, agentState == nil || agentState == "speaking" {
            begin(with: [chunk])
        } else {
            remember(chunk)
        }
        tick(now: chunk.arrival)
    }

    /// Advances the heard count and the time-based ends. Call often (every 10–20 ms).
    func tick(now: ContinuousClock.Instant) {
        if let draining, now >= draining.deadline { finish() }
        if isActive, agentState == nil, let lastVoiceEnd, now - lastVoiceEnd >= configuration.silenceGate { finish() }
        reportHeard(now: now)
    }

    /// Stops at once (the session ended). The avatar is interrupted if it is speaking.
    func reset() {
        if isActive || awaitingPlayout { sink.interrupt() }
        let wasActive = isActive
        clear()
        preRoll = []
        agentState = nil
        if wasActive { onActiveChanged?(false) }
    }

    // MARK: - Utterances

    /// Silent buffers inside a silence-gated utterance, sent only if the voice resumes.
    private var heldSilence: [AgentAudioChunk] = []

    /// The voice was audible in the last 80 ms of audio sent: stopping now cuts it off mid-word.
    private var isSounding: Bool {
        guard let lastVoiceEnd, let lastChunkEnd else { return false }
        return lastChunkEnd - lastVoiceEnd < .milliseconds(80)
    }

    private func isVoiced(_ chunk: AgentAudioChunk) -> Bool { chunk.level >= configuration.voiceThreshold }

    private func remember(_ chunk: AgentAudioChunk) {
        preRoll.append(chunk)
        let limit = Self.samples(in: configuration.preRoll)
        var total = preRoll.reduce(0) { $0 + $1.samples }
        while total > limit, !preRoll.isEmpty { total -= preRoll.removeFirst().samples }
    }

    private func begin(with chunks: [AgentAudioChunk]) {
        let buffered = Array(preRoll.drop(while: { !isVoiced($0) }))
        preRoll = []
        clear()
        isActive = true
        utteranceLatency = latency()
        onActiveChanged?(true)
        for chunk in buffered + chunks { append(chunk) }
    }

    private func append(_ chunk: AgentAudioChunk) {
        guard chunk.samples > 0 else { return }
        if isVoiced(chunk) {
            for held in heldSilence { send(held) }
            heldSilence = []
            lastVoiceEnd = chunk.arrival + chunk.duration
        }
        lastChunkEnd = chunk.arrival + chunk.duration
        send(chunk)
    }

    private func send(_ chunk: AgentAudioChunk) {
        guard (try? sink.appendAudio(pcm: chunk.pcm, sampleRate: PCMConverter.outputRate)) != nil else { return }
        let playAt = max(chunk.arrival + utteranceLatency, schedule.last?.end ?? chunk.arrival + utteranceLatency)
        schedule.append(Scheduled(start: appended, count: chunk.samples, playAt: playAt))
        appended += chunk.samples
    }

    private func finish() {
        guard isActive else { draining = nil; return }
        let cutOff = draining?.cutOff ?? false
        if cutOff {
            sink.interrupt()
            clear()
        } else {
            sink.endSpeech()
            let keep = (schedule, appended, heardBase, reported)
            clear()
            (schedule, appended, heardBase, reported) = keep
            awaitingPlayout = appended > reported
        }
        onActiveChanged?(false)
    }

    private func clear() {
        isActive = false
        awaitingPlayout = false
        draining = nil
        heldSilence = []
        schedule = []
        appended = 0
        heardBase = 0
        reported = 0
        lastChunkEnd = nil
        lastVoiceEnd = nil
    }

    private func reportHeard(now: ContinuousClock.Instant) {
        guard isActive || awaitingPlayout else { return }
        while let first = schedule.first, first.end <= now {
            heardBase = first.start + first.count
            schedule.removeFirst()
        }
        var heard = heardBase
        if let first = schedule.first, now > first.playAt {
            heard = first.start + min(first.count, Self.samples(in: now - first.playAt))
        }
        if heard > reported {
            reported = heard
            sink.audioPlayed(samples: heard)
        }
        if awaitingPlayout, reported >= appended {
            awaitingPlayout = false
            schedule = []
        }
    }

    static func samples(in duration: Duration) -> Int {
        let (seconds, attoseconds) = duration.components
        return Int(seconds) * PCMConverter.outputRate + Int(attoseconds / 1_000_000_000_000) * PCMConverter.outputRate / 1_000_000
    }
}
