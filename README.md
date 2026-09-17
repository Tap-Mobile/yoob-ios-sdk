# Yoob for iOS

Talking characters that render on the iPhone. You give Yoob speech audio; it plays the audio and moves the face in sync,
at 25 fps, entirely on the device. Add Yoob voice for a live spoken conversation with no provider key, or bring your
own LLM and voice.

- **Small install.** The package adds about 1.8 MB to your app. Character files (26–40 MB) download on first use from
  `cdn.yoob.com` in verified, resumable chunks, and on a good connection the character's picture appears within about a second.
- **Any voice.** Talk through Yoob voice, or pass mono 16-bit PCM from OpenAI Realtime, Gemini Live, ElevenLabs, your
  own TTS, or a recording.
- **Private by design.** Faces render on the device. The SDK sends Yoob only what is listed under [Network](#network):
  conversation audio goes through Yoob only when you use Yoob voice.

Requires iOS 17 or later and Xcode 16 or later. The current preview release is 0.4.0; see [Changes in 0.4.0](#changes-in-040) if you are upgrading.

## Install

In Xcode choose **File › Add Package Dependencies…** and enter:

```
https://github.com/Yoob-com/yoob-ios-sdk
```

or add it to `Package.swift`:

```swift
.package(url: "https://github.com/Yoob-com/yoob-ios-sdk", from: "0.4.0")
```

## Quick start

### 1. Open a session on your backend

Create an API key in the [Yoob console](https://yoob.com/account/). Keep the key on your server; the app gets a
short-lived session instead. Your backend checks who is asking, rate-limits them, and asks for exactly the one
character the app wants (never `"*"`):

```sh
curl -X POST https://api2.yoob.com/api/v1/avatar/sessions \
  -H "Authorization: Bearer $YOOB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"characters": ["luna-realistic"]}'
```

```json
{ "session_token": "…", "download_token": "yg1.…", "heartbeat_seconds": 15 }
```

[`Examples/token-server`](Examples/token-server/server.mjs) is a runnable backend that does all of this. It refuses
every request until you replace its `requireUser()` with your own sign-in check; for local development, start it with
`YOOB_EXAMPLE_ALLOW_ANONYMOUS=1`. See [Security](#security).

### 2. Show the character

```swift
import SwiftUI
import Yoob

struct CharacterScreen: View {
    @State private var avatar = YoobAvatar(.cloud(character: "luna-realistic") {
        try await MyBackend.yoobSession()          // returns YoobCredentials
    })

    var body: some View {
        YoobAvatarView(avatar)
            .ignoresSafeArea()
            .task { try? await avatar.prepare() }
    }
}
```

`prepare()` opens the session, downloads what is missing and warms up the renderer. The character's poster and idle
motion show while the models download.

### 3. Make it talk

```swift
// Call for each chunk as it streams in (PCM16, mono, little-endian).
try avatar.speak(pcm: chunk, sampleRate: 24_000)

// After the last chunk. The face settles back to idle when the audio ends.
avatar.endSpeech()

// Barge-in: stop now. Returns how many samples were heard, for truncating a realtime reply.
let heard = avatar.interrupt()
```

The avatar holds the voice back for the first frame (up to `maxSyncDelay`, 1.8 s by default), so lips and voice
start together. Audio always plays, even if the character failed to load.

### Your app already plays the audio?

Render without playing, and report how much has been heard:

```swift
try avatar.appendAudio(pcm: chunk, sampleRate: 24_000)
avatar.audioPlayed(samples: samplesHeardSoFar)   // call often, e.g. from your audio tap
avatar.endSpeech()
```

## Talk with it: Yoob voice

No provider key needed; minutes are billed through your Yoob workspace. Yoob hosts the voice (OpenAI Realtime) and
tunes it for the characters: server turn detection, far-field noise reduction, captions and a 1.08 speaking speed.

```swift
var options = YoobConversation.Options()
options.greet = true

let conversation = YoobConversation(avatar: avatar, options: options) {
    try await MyBackend.yoobVoiceSession()     // returns YoobVoiceSession
}
try await conversation.start()                 // asks for the microphone
// conversation.state, .userTranscript and .assistantTranscript are observable, for captions.
conversation.stop()
```

Your backend asks Yoob for a voice session with your API key, and returns the response body unchanged:

```sh
curl -X POST https://api2.yoob.com/api/v1/voice/sessions \
  -H "Authorization: Bearer $YOOB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"voice": "marin", "instructions": "You are Luna, a warm, curious companion.", "max_seconds": 900}'
```

The `201` response has `voice_session_id`, `voice_token`, `url` (`wss://voice.yoob.com/v1/realtime?model=…`),
`model`, `max_seconds`, `credits_per_minute` and `expires_at`. The app decodes it with
`JSONDecoder().decode(YoobVoiceSession.self, from: data)`.
[`Examples/token-server`](Examples/token-server/server.mjs) has a `/yoob-voice` route.

- **One session per conversation.** A voice session opens exactly one connection, and its token must be used within 5
  minutes. `start()` asks for a new one every time, so don't cache it.
- **Voice and instructions.** Set them on your backend: the app can't change them, and the prompt never reaches the
  device. `voice` is one of `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin` or
  `cedar`, and `instructions` can be up to 8,000 characters. If the backend leaves them out, `options.voice` and
  `options.instructions` are used instead.
- **Always set them.** Send `voice` and `instructions` from your backend on every voice session; otherwise the app
  decides what the voice says on your bill.
- **Voice host.** The SDK connects only to `wss://*.yoob.com` and refuses any other session `url`. To run your own
  relay, set `options.voiceHosts = ["voice.example.com"]` (`*.example.com` matches subdomains).
- **Fixed settings.** Yoob sets the model, turn detection, noise reduction, transcription and speed, so those options
  are ignored.
- **Errors.** When the workspace is out of credit, the API answers `402 {"code": "quota_exceeded"}`, and decoding it
  throws `YoobError.outOfCredit`. A rejected API key throws `.unauthorized`. If the voice service ends the call,
  `lastError` is `.voiceSession(code:message:)`, and its `localizedDescription` is a message you can show the user:

| Close code | Message |
|---|---|
| 4001, 4002, 4003 | The voice session was refused, had expired, or was already used. Start the conversation again. |
| 4008 | This conversation reached its usage limit. |
| 4009 | This conversation reached its time limit (`max_seconds`). |
| 4010 | The conversation ended because it was idle for too long. |
| 4029 | Voice has reached its usage limit for now (too many conversations at once, or the daily quota). Try again later. |
| 1013 | Voice is busy right now. Try again in a moment. |
| 1011 | The voice service disconnected. Start the conversation again. |

The microphone stays open while the character speaks, so the user can interrupt it; iOS voice processing removes the
character's voice from what is captured. When the user starts talking, the reply stops and the model is told exactly
how much of it was heard.

Add `NSMicrophoneUsageDescription` to your Info.plist.

## Talk with it: your own OpenAI account

Pass `clientSecret` instead of a voice session, and OpenAI bills your account directly:

```swift
var options = YoobConversation.Options()
options.voice = "marin"
options.instructions = "You are Luna, a warm, curious companion."
options.greet = true

let conversation = YoobConversation(avatar: avatar, options: options) {
    try await MyBackend.openAIClientSecret()   // POST /v1/realtime/client_secrets on your server
}
try await conversation.start()
```

Barge-in, captions and `send(text:)` work the same as with Yoob voice. The defaults come from Yoob's latency
measurements:

| Option | Default | Why |
|---|---|---|
| `turnDetection` | `.serverVAD()`, 450 ms silence | Replies start about 0.8 s sooner than `.semantic` in Yoob's measurements |
| `noiseReduction` | `far_field` | A phone held away from the face |
| `speed` | `1.08` | Natural but snappy |

In a noisy room, raise the server VAD threshold instead of muting the microphone.

## Talk with it: Gemini Live

Bring your own Gemini voice with `YoobGeminiConversation`. It has the same states, transcripts and barge-in as
`YoobConversation`.

```swift
var options = YoobGeminiConversation.Options()
options.voice = "Kore"
options.systemInstruction = "You are Luna, a warm, curious companion."
options.greet = true

let conversation = YoobGeminiConversation(avatar: avatar, options: options) {
    try await MyBackend.geminiToken()          // an ephemeral token's `name`, created on your server
}
try await conversation.start()                 // asks for the microphone
// conversation.state, .userTranscript and .assistantTranscript are observable, for captions.
conversation.stop()
```

The app never sees your Gemini API key. Your backend creates a single-use
[ephemeral token](https://ai.google.dev/gemini-api/docs/live-api/ephemeral-tokens) and returns its `name`:

```js
// POST /gemini-token on your server (Node, @google/genai)
const token = await ai.authTokens.create({
  config: {
    uses: 1,
    expireTime: new Date(Date.now() + 30 * 60_000).toISOString(),    // messages stop after this
    newSessionExpireTime: new Date(Date.now() + 60_000).toISOString(), // the app must connect before this
    liveConnectConstraints: { model: "gemini-3.8-live" },              // optional: lock the model
  },
});
return { name: token.name };
```

The REST equivalent is `POST https://generativelanguage.googleapis.com/v1beta/auth_tokens` with the `x-goog-api-key`
header. Settings locked with `liveConnectConstraints` take precedence over the ones the app sends.

The conversation connects to Gemini's `BidiGenerateContentConstrained` WebSocket with the token. It converts the
microphone's 24 kHz audio to the 16 kHz Gemini expects, and plays Gemini's 24 kHz replies through the avatar.

| Option | Default | Why |
|---|---|---|
| `model` | `gemini-3.8-live` | Google's recommended low-latency native-audio Live model |
| `voice` | Gemini's choice | Any prebuilt voice name, for example `Kore` or `Puck` |
| `activityDetection.startSensitivity` | `.high` | Quick barge-in. Use `.low` in noisy rooms. |
| `activityDetection.endSensitivity` | `.high` | Ends the user's turn sooner |
| `activityDetection.silenceMS` | `450` | The silence window Yoob measured as fastest with OpenAI |
| `activityDetection.prefixPaddingMS` | `100` | Short enough for one-word answers |
| `inputTranscription` / `outputTranscription` | `true` | Captions for both sides |

`send(text:)` sends a typed turn. `goAwayTimeLeft` is set when Gemini is about to close the connection: audio-only
sessions last up to 15 minutes. Gemini doesn't report when a user turn ends, so a spoken turn goes from `.listening`
straight to `.speaking`. `.thinking` appears after `greet` and `send(text:)`.

## Microphone controls

```swift
let mic = avatar.microphone
mic.setMuted(!mic.isMuted)
Gauge(value: mic.level) { EmptyView() }               // 0...1, about 20 updates a second
ForEach(mic.inputs) { input in                         // built-in, wired and Bluetooth inputs
    Button(input.name) { try? mic.select(inputID: input.id) }
}
```

`mic.state` is `.off`, `.starting`, `.live`, `.muted` or `.failed(error)`. Using your own voice stack? Set
`mic.onAudio` (24 kHz PCM16) and call `try await mic.start()`.

## Characters

| Id | Style | Download | On device |
|---|---|---|---|
| `luna-realistic` | Photoreal | 26 MB | 34 MB |
| `luna-anime` | Anime | 40 MB | 46 MB |

Pin a version with `YoobAvatar(source, version: "2026.09.17.1")`. Without one, the newest compatible version is used,
and an update downloads only the files that changed.

## States

`avatar.phase` is observable:

| Phase | Meaning |
|---|---|
| `.downloading(progress)` | Files are downloading. `progress.fraction` goes from 0 to 1. |
| `.warming` | Files are ready; the renderer is starting. |
| `.ready` | Idle and ready to speak. |
| `.speaking` | Audio is playing and the face follows it. |
| `.failed(error)` | Loading failed. Call `prepare()` again to resume. |
| `.stopped(error)` | The session ended and the character stopped rendering. See below. |

The session stops when the workspace is out of credit (`.outOfCredit`), when Yoob refuses the session or its API key
was revoked (`.unauthorized`), or with `.sessionEnded` after three heartbeats in a row fail, when a sandbox session
reaches its time limit, or when the workspace is suspended. The avatar then stops rendering,
`speak` throws the same error, and `avatar.onSessionEnded` is called. `prepare()` opens a new session.

```swift
avatar.onSessionEnded = { error in showBanner(error.localizedDescription) }
```

`avatar.stats` counts frames shown and skipped. `avatar.lastRendererError` says why rendering stopped, if it did.

## Network

This is everything the SDK sends to Yoob:

| Call | When | Contents |
|---|---|---|
| Character files from `cdn.yoob.com` | First use and version updates | Your download grant |
| `POST api2.yoob.com/api/v1/sessions/heartbeat` | Every 15 s from the start of `prepare()` | Your session token |
| `POST api2.yoob.com/api/v1/sessions/end` | `close()` | Your session token |
| `wss://voice.yoob.com/v1/realtime` | Yoob voice conversations | Your voice token, microphone audio, typed text |

Heartbeats are how session time is metered. With Yoob voice, `voice.yoob.com` relays the conversation to OpenAI and
meters its minutes. With your own OpenAI account or Gemini, audio goes directly from the device to that provider. Call `await avatar.close()` when the character leaves the screen.
When the app returns to the foreground, `await avatar.refreshSession()` checks the session at once and opens a new one
if Yoob ended it while the app was suspended.

## Offline and shipped files

A complete download is reused when the CDN can't be reached, but `prepare()` still opens a session with your backend,
and heartbeats must reach Yoob: a character can't run without a metered session.

To ship a character inside your app, save the signed manifest the CDN serves
(`https://cdn.yoob.com/v1/characters/<id>/<version>.json`, unchanged) as `character.signed.json` next to the files it
lists, and use `.local`:

```swift
let pack = Bundle.main.url(forResource: "luna-realistic", withExtension: nil)!
let avatar = YoobAvatar(.local(pack) { try await MyBackend.yoobSession() })
```

The SDK verifies the manifest signature against its pinned key and every file's SHA-256 before use, and refuses
unsigned or modified packs with `YoobError.invalidAssets`. The files load from disk, so the character works without
the CDN, but the session is opened and metered like a cloud one.

## Performance

- **Rendering:** the realistic renderer uses the GPU and falls back to the CPU if a GPU returns empty frames (the iOS
  Simulator does).
- **Test hardware:** the engines are the ones the Luna app runs on iPhone Air (A19 Pro). On a Mac with Apple silicon they render at 45–80 fps.
- **Older iPhones:** test on your oldest supported device before shipping.
- **Simulator:** it works but renders slowly. Build Release to judge motion there.

## Example

[`Examples/QuickStart`](Examples/QuickStart) is a one-screen app with both characters, a Talk button that uses Yoob
voice, and a sample greeting (an AI-generated voice). Run [`Examples/token-server`](Examples/token-server/server.mjs)
next to it:

```sh
YOOB_API_KEY=yoob_test_… YOOB_EXAMPLE_ALLOW_ANONYMOUS=1 node Examples/token-server/server.mjs   # 127.0.0.1:3100
cd Examples/QuickStart && xcodegen generate && open QuickStart.xcodeproj
```

`YOOB_EXAMPLE_ALLOW_ANONYMOUS=1` lets anyone who can reach the server mint sessions, so use it only on your own
machine. Before you deploy a token server, replace `requireUser()` with your own sign-in check.

## Security

- **Keys stay on your server.** A Yoob API key never belongs in an app, a web page or a repository. The API rejects
  key calls that come from a browser, and an app binary can be taken apart, so keep the key in your server's
  environment. The app only ever holds a session token, a download grant and a voice token.
- **Test keys for development.** A `yoob_test_` key opens sandbox sessions that don't use credits and are limited in
  length (a few minutes each, with a daily total). Sandbox mode comes from the key alone; there is no request flag
  for it.
- **Grants are short-lived and per character.** A download grant covers the characters its session was opened for and
  expires soon. Heartbeats may hand the SDK a renewed grant, which it uses from the next download request on. A voice
  token opens one conversation and must be used within 5 minutes.
- **Heartbeats are enforced.** They start when `prepare()` opens the session. A refused session (401 or 403), an
  exhausted workspace (402), or three failed heartbeats in a row stop the character (see [States](#states)).
  Transient failures are retried with backoff within those three attempts. When Yoob ends a session that went quiet
  (an app suspended in the background), the SDK asks your `credentials` closure for a new one.
- **Signed packs only.** Downloaded and shipped packs must carry a manifest signed by Yoob, and every file is checked
  against it.
- **Voice only goes to Yoob.** Yoob voice connects only to `wss://*.yoob.com` unless you set `voiceHosts`.
- **What your token server must do.** The example server does each of these; keep them when you write your own:
  1. Authenticate the user before minting anything, and fail closed.
  2. Rate-limit sessions per user.
  3. Accept only the character ids you offer, and send exactly that one character. Never `"*"`.
  4. Always set the voice and instructions for voice sessions on the server.
  5. Build the request body yourself. Don't pass fields from the app through to Yoob.
  6. Keep the key in the server's environment, and return Yoob's response without logging tokens.

## Changes in 0.4.0

- `.local(url)` is now `.local(url, credentials:)`. Local packs need `character.signed.json` and a metered session.
- Heartbeats start with `prepare()` and are enforced. New `YoobError.sessionEnded`, `avatar.onSessionEnded` and
  `avatar.refreshSession()`.
- Heartbeat replies may carry a renewed download grant (`download_token`, `download_token_expires_at`; `grant` and
  `grant_expires_at` are also accepted); older replies still work. `stop` with `sandbox-limit` or `suspended` ends the
  session as `.sessionEnded`, and `key-revoked` as `.unauthorized`.
- Yoob voice connects only to `wss://*.yoob.com` unless `YoobConversation.Options.voiceHosts` says otherwise.
- The example token server fails closed without auth, allowlists characters (`YOOB_CHARACTERS`, default
  `luna-realistic,luna-anime`), rate-limits each user, always pins the voice prompt, and never passes request fields
  such as `is_sandbox` through.

## License

The SDK source is Apache-2.0. Character model files are licensed separately and are not in this repository; see
[NOTICE](NOTICE).
