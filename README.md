# Yoob for iOS

Talking characters that render on the iPhone. You give Yoob speech audio; it plays the audio and moves the face in sync,
at 25 fps, entirely on the device. Your LLM, voice and UI stay yours.

- **Small install.** The package adds about 1.8 MB to your app. Character files (26–40 MB) download on first use from
  `cdn.yoob.com` in verified, resumable chunks, and on a good connection the character's picture appears within about a second.
- **Any voice.** Pass mono 16-bit PCM from OpenAI Realtime, ElevenLabs, your own TTS, or a recording.
- **Private by design.** The SDK sends Yoob only the three things listed under [Network](#network). No audio, text or
  frames ever leave the device through Yoob.

Requires iOS 17 or later and Xcode 16 or later. The current preview release is 0.1.0.

## Install

In Xcode choose **File › Add Package Dependencies…** and enter:

```
https://github.com/Tap-Mobile/yoob-ios-sdk
```

or add it to `Package.swift`:

```swift
.package(url: "https://github.com/Tap-Mobile/yoob-ios-sdk", from: "0.1.0")
```

## Quick start

### 1. Open a session on your backend

Create an API key in the [Yoob console](https://yoob.com/account/). Keep the key on your server; the app gets a
short-lived session instead:

```sh
curl -X POST https://api.yoob.com/api/v1/avatar/sessions \
  -H "Authorization: Bearer $YOOB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"characters": ["luna-realistic"]}'
```

```json
{ "session_token": "…", "download_token": "yg1.…", "heartbeat_seconds": 15 }
```

[`Examples/token-server`](Examples/token-server/server.mjs) is a complete 30-line backend.

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

`prepare()` downloads what is missing and warms up the renderer. The character's poster and idle motion show while the
models download.

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
| `.stopped(error)` | The session ended, for example `.outOfCredit`. |

`avatar.stats` counts frames shown and skipped. `avatar.lastRendererError` says why rendering stopped, if it did.

## Network

This is everything the SDK sends to Yoob:

| Call | When | Contents |
|---|---|---|
| Character files from `cdn.yoob.com` | First use and version updates | Your download grant |
| `POST api.yoob.com/api/v1/sessions/heartbeat` | Every 15 s while prepared | Your session token |
| `POST api.yoob.com/api/v1/sessions/end` | `close()` | Your session token |

Heartbeats are how session time is metered. Call `await avatar.close()` when the character leaves the screen.

## Offline and shipped files

A complete download is reused offline. To ship a character inside your app instead, use `.local(url)` with a pack
directory. Local packs make no network calls.

## Performance

- **Rendering:** the realistic renderer uses the GPU and falls back to the CPU if a GPU returns empty frames (the iOS
  Simulator does).
- **Test hardware:** the engines are the ones the Luna app runs on iPhone Air (A19 Pro). On a Mac with Apple silicon they render at 45–80 fps.
- **Older iPhones:** test on your oldest supported device before shipping.
- **Simulator:** it works but renders slowly. Build Release to judge motion there.

## Example

[`Examples/QuickStart`](Examples/QuickStart) is a one-screen app with both characters and a sample greeting (an
AI-generated voice).

```sh
cd Examples/QuickStart && xcodegen generate && open QuickStart.xcodeproj
```

## License

The SDK source is Apache-2.0. Character model files are licensed separately and are not in this repository; see
[NOTICE](NOTICE).
