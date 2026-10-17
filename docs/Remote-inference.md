# Remote inference — server and client

One WaifuFinder can run the models and another can borrow them. The borrowing machine
does the browsing, the grouping, the exporting and the compositing; the machine that
owns the weights does the arithmetic.

```text
Client (owns the video)                    Server (owns the weights)
┌──────────────────────────────┐          ┌──────────────────────────────┐
│ WaifuSession                  │          │ RemoteServer                  │
│   ├ RemoteTaggerBackend   ────┼─ frames ─┼─▶ WDTaggerBackend            │
│   └ RemoteUncoverBackend  ────┼─ job    ─┼─▶ TemporalVideoUpscaler       │
│                                │          │                              │
│ UI labels get a ᴿ superscript │          │ Settings → Remote lists the   │
│ when the server can do it     │          │ connected client and can kick │
└──────────────┬───────────────┘          └──────────────┬───────────────┘
               │   UDP beacon, 1 Hz                     │   TCP, one connection
               └──── 239.255.42.99:42888 ───────────────┘
```

## The load-bearing decision

**Remote execution is a `TaggerBackend` and a `TemporalVideoUpscaler`.**

Those two protocols already exist (`Engine/TaggerBackend.swift`,
`Model/UncoverTypes.swift`), and every piece of behaviour around them — progress,
cancellation, result caching, comparison, compositing, export — is written against the
protocol rather than against a concrete model. A remote backend is therefore not a new
subsystem. It is one more implementation of a seam the codebase already has, and the
amount of code that has to learn about remoteness is the two factory lines that pick a
backend.

Everything else in this document is detail around that.

## What crosses the socket

Only the two things that actually need the model, and nothing else.

| Job | Upstream | Downstream |
|---|---|---|
| Tagging | 1…8 analysis-resolution JPEG thumbnails, or pre-rendered 448×448 RGBA8 | one `Float32[10_861]` probability row per frame |
| Uncover | `manifest.json` plus the `degraded/%06d.png` frames | `model`, `seconds`, `outputScale`, and `reconstructed/%06d.png` bytes |

Three consequences worth stating, because they are where the design would otherwise
look wrong:

- **`original/` never travels.** `UncoverRenderer` composites the reconstructed region
  over the original frame, and the client already holds the original frames in its own
  prepared directory. Only `degraded/` is the model's input, so only `degraded/` is sent.
- **The client runs the threshold.** `TagVocabulary.result(probabilities:threshold:)`
  turns a probability row into a rating, a top-tag list and a general score. The server
  ships the probability rows and nothing else; the client applies its own
  `settings.taggerThreshold`. A remote result is therefore shaped exactly like a local
  one and goes through exactly the same `TagProbabilityCodec.encode` path, so tagging
  cache rows written by a server and rows written locally are interchangeable.
- **The client needs the vocabulary, not the weights.** Filtering, facets and
  autocomplete all need `TagVocabulary`. The server ships `wd-tagger-labels.csv` once,
  zlib-compressed, cached on the client under its `cacheFingerprint`. A client with no
  local tagger installation can still tag, filter and search.

## The vocabulary unlocks the "Set up WD Tagger" gate

`WaifuSession.startTagging` currently refuses without a local model, and
`session.tagVocabulary == nil` is what makes the panels offer "Set up WD Tagger in
Settings…" instead of a Start button. With a server attached that gate is wrong: the
server has the weights and the client has the labels.

The change is to make vocabulary resolution fall back to the remote one:

```swift
// startTagging
guard WDTaggerConfiguration.load() != nil || RemoteClient.shared.vocabulary != nil else {
    errorMessage = "Install WD Tagger in Settings, or connect to a server that has it."
    return
}
```

and `tagVocabulary` is published from the local vocabulary when there is one, otherwise
from the downloaded remote vocabulary. Everything downstream — candidate selection,
cache keys, facet index, `TagResultMemory` — already takes a `TagVocabulary` value and
is unaffected.

## Discovery: a once-per-second beacon

The server broadcasts a small UDP datagram every **1.0 s** while the service is
running. It is a hint, not a credential: it carries no key material, and everything it
claims is re-verified by the authenticated TCP handshake.

- **Group** `239.255.42.99:42888` (`WF` in hex, matching `WFR1`). Multicast is primary
  because it does not depend on the subnet mask.
- **Broadcast** `255.255.255.255:42888` is sent as well, best effort, for networks that
  filter multicast. A client only needs one of the two.
- A client binds the port and collects beacons; an entry expires after 3 missed
  beacons, so a server that quits disappears within about three seconds.
- A manual address always works and is never gated on the beacon, including on a
  network that blocks both.

### Beacon datagram

Fixed 40-byte header, little-endian, then two length-prefixed UTF-8 fields. Total is
capped at 128 bytes so it fits inside one Ethernet MTU unfragmented.

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `WFRB` |
| 4 | 1 | beacon version (1) |
| 5 | 1 | protocol version (1) |
| 6 | 2 | flags |
| 8 | 4 | TCP port the server actually bound |
| 12 | 8 | `serverInstance` — first 8 bytes of a per-launch UUID |
| 20 | 8 | `keyID` — first 8 bytes of `SHA256("WF1/keyid" ‖ PSK)` |
| 28 | 4 | sent-at, unix milliseconds, low 32 bits |
| 32 | 4 | reserved, zero |
| 36 | 4 | `nameLength`, then `name` (UTF-8, NUL-free, ≤ 63 bytes) |
| … | 1 | `versionLength`, then `version` (UTF-8, ≤ 31 bytes) |

Flags: bit 0 accepting connections, bit 1 a client is connected, bit 2 tagger ready,
bits 3…7 the five `UncoverAlgorithm` cases in `allCases` order, available.

`keyID` is a hash prefix of a 256-bit random secret, so publishing it reveals nothing
and lets the client say **Ready** versus **Key needed** for an autodiscovered server
without opening a socket. It is also how the client learns that a server rotated its key.

The client's connect address is the **source IP of the beacon datagram**, not the name in
the payload — a `.local` name can resolve differently on two machines.

## The shared key

- 32 bytes from `SystemRandomNumberGenerator` (which is `SecRandomCopyBytes` under it).
- Stored in the **Keychain**, `kSecClassGenericPassword`, service
  `com.alastorid.waifufinder`, account `remote-server-key`. A secret deserves the
  system store; nothing else in the app currently keeps one, so this is a deliberate new
  `Security` dependency and nothing more.
- `RemoteKeyStore` is a protocol with a Keychain implementation and a `0600` file
  implementation used by tests, so the protocol is exercised without a keychain prompt.
- Generated on first start of the service, and on **Rotate Key…**.

### Display format

```
WF1-3F9A21C7-B40E5D18-6A2C9E05-D7718B34-9C0AE652-1D84F7B0-52C6A93E-8B1D4705-A913
```

Eight groups of eight uppercase hex characters (the 256-bit key), then a four-character
check group: the first two bytes of `SHA256("WF1/check" ‖ key)`. Parsing rejects a bad
check character immediately, so a mistyped key fails in the field with a clear message
instead of producing a confusing authentication error.

Hex is used because it is unambiguous to read aloud and to compare by eye. It is a human
presentation of the key and never appears on the wire — see the framing rules below.

### Rotation

**Rotate Key…** asks for confirmation, generates a new key, replaces the Keychain item,
updates the `keyID` in the next beacon, and disconnects the live session with
`close{reason: "keyRotated"}`. The client then re-reads the beacon, notices the `keyID`
no longer matches the stored key, and shows **Server key changed — paste the new key**
instead of retrying a handshake that cannot succeed.

## Frame layout

One TCP connection, in both directions, carrying a stream of length-prefixed frames.
Little-endian, which is correct because both supported architectures are little-endian —
the same argument `WDTaggerWorker.decodeProbabilities` already makes.

```text
byte  0 ┌──────────────┬─────────┬─────────┬──────────┬───────────┬───────────┐
        │ magic "WFR1" │ version │ opcode  │  flags   │ streamID  │ byteCount │
byte 24 └──────────────┴─────────┴─────────┴──────────┴───────────┴───────────┘
        ├───────────────────────── 24-byte header ──────────────────────────┤
        │  payload, byteCount bytes, exactly as written                    │
        └───────────────────────────────────────────────────────────────────┘
```

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic, `0x57 0x46 0x52 0x31` (`"WFR1"`) |
| 4 | 2 | version (1) |
| 6 | 2 | opcode |
| 8 | 4 | flags |
| 12 | 4 | `streamID` |
| 16 | 4 | `byteCount` |
| 20 | 4 | reserved, zero |

Flags: bit 0 `final` — last frame of this message; bit 1 `zlib` — payload is
zlib-compressed; bit 2 `error` — the message ended in an error.

**A message is one frame if it fits, otherwise a run of frames sharing `streamID` where
only the last carries `final`.** Maximum frame payload is 1 MiB and maximum message size
is 64 MiB, so a 300-frame Uncover job is bounded at 64 MiB of upstream PNG and cannot
make either side allocate without limit.

### What "raw" means here, precisely

- **JSON never carries binary.** A JSON payload holds strings, numbers, booleans and
  arrays of those. If a value is bytes, it is not in the JSON.
- **Bytes travel as bytes.** A thumbnail, an RGBA plane, a probability row, a PNG
  frame and the compressed vocabulary are each the payload of their own frames, byte for
  byte as produced. Nothing is hex-encoded, base64-encoded or otherwise transformed on
  the way to the socket.
- **The transport is a cipher, not an encoding.** After the handshake each frame payload
  is sealed with ChaCha20-Poly1305 (see below). The bytes inside are still the raw bytes;
  sealing is confidentiality and integrity, not a way of smuggling binary through text.

The rule that follows from this: if the client already holds the exact bytes a model
wants, those bytes are what is sent. The cached analysis thumbnail goes upstream as the
JPEG that is already in the thumbnail cache — it is not decoded to pixels and
re-encoded.

## Opcodes

Connection-level messages use `streamID = 0`. Requests use a client-chosen `streamID ≥ 1`
that every frame of that request and its response shares.

| Opcode | Name | Direction | Payload |
|---|---|---|---|
| `0x0001` | `hello` | C→S | JSON: protocol, app version, client name, client nonce |
| `0x0002` | `challenge` | S→C | JSON: server name, app version, server nonce, `keyID` |
| `0x0003` | `auth` | C→S | binary: 32-byte client HMAC |
| `0x0004` | `authResult` | S→C | JSON: ok, session id, then the server proof |
| `0x0005` | `capabilities` | S→C | JSON: the capability document |
| `0x0006` | `ping` | C→S | JSON: monotonic counter |
| `0x0007` | `pong` | S→C | JSON: the counter echoed |
| `0x0008` | `close` | S→C | JSON: reason |
| `0x0009` | `error` | S→C | JSON: code, message |
| `0x0101` | `tagBatch` | C→S | binary: images |
| `0x0102` | `tagResult` | S→C | binary: probability rows |
| `0x0110` | `vocabularyRequest` | C→S | empty |
| `0x0111` | `vocabularyResponse` | S→C | binary: fingerprint + zlib CSV |
| `0x0120` | `uncoverBegin` | C→S | JSON: manifest, algorithm, `allowUnsafeMemory` |
| `0x0121` | `uncoverInput` | C→S | binary: one `degraded/%06d.png` |
| `0x0122` | `uncoverInputEnd` | C→S | empty |
| `0x0123` | `uncoverProgress` | S→C | JSON: progress message |
| `0x0124` | `uncoverResultBegin` | S→C | JSON: model, seconds, outputScale |
| `0x0125` | `uncoverResult` | S→C | binary: one `reconstructed/%06d.png` |
| `0x0126` | `uncoverResultEnd` | S→C | empty |
| `0x0130` | `cancelJob` | C→S | JSON: the job's `streamID` |

Progress frames use the job's own `streamID`, so a status line is always attributable to
the job that produced it and cannot be confused with a different request.

## Binary payload shapes

A `WireReader`/`WireWriter` pair covers `u8`, `u16`, `u32`, `u64`, `f32`, `f64`,
length-prefixed UTF-8 strings, length-prefixed byte runs, and arrays of the same. Every
read is bounds-checked and every count is validated against the remaining payload before
anything is allocated.

### `tagBatch` → `tagResult`

```text
u8   batchCount        1…8
u8   codec             0 = JPEG/PNG container, 1 = raw RGBA8 at 448×448
u16  reserved
     batchCount × { u32 byteLength ‖ bytes }
```

```text
u8   resultCount
u64  vocabularyFingerprint        the server's TagVocabulary.cacheFingerprint
     resultCount × { u32 byteLength ‖ Float32[10_861] little-endian ‖ bytes }
```

The probability rows are exactly the bytes `WDTaggerBackend` already produces. The client
runs `TagVocabulary.result(probabilities:threshold:)` itself.

Wire cost is 43 444 bytes per frame, so an eight-frame batch is about 348 KB
downstream. Quantising to `Float16`-range `UInt16` would halve it, and is left as a
negotiated future option behind a capability flag rather than done now: a remote result
should be byte-identical to a local one, and the on-disk tag cache would otherwise hold
rows that local tagging could never reproduce.

### `vocabularyResponse`

```text
u64  cacheFingerprint
u32  uncompressedByteCount
     zlib-compressed CSV bytes
```

`Compression` is already linked for `TagProbabilityCodec`. 308 KB of CSV goes over as
roughly 60 KB, once per fingerprint, then never again.

### `uncoverInput` / `uncoverResult`

Both are a single PNG file's bytes. The frame index is already implied by arrival order
and by the manifest, so there is no per-frame header — one payload, one file.

## The handshake

Four messages, then every subsequent frame is sealed. The exchange is deliberately
shaped like `Noise_NNpsk0`: two ephemeral nonces, one HMAC from each side, and no
long-term key material ever on the wire.

```text
C → S  hello        { protocol, appVersion, clientName, clientNonce: 32 bytes }
S → C  challenge    { serverName, appVersion, serverNonce: 32 bytes, keyID }
C → S  auth         HMAC-SHA256(PSK, "WF1/auth" ‖ clientNonce ‖ serverNonce)   32 bytes
S → C  authResult   { ok, sessionID, capabilities, serverProof }                sealed
```

- `serverProof` is `HMAC-SHA256(PSK, "WF1/server" ‖ sessionID ‖ clientNonce ‖ serverNonce)`,
  so the client also learns it is talking to the holder of the key and not to a passive
  listener that merely replayed a challenge.
- Session keys, `HKDF-SHA256` with `salt = clientNonce ‖ serverNonce`:
  `c2s` uses info `"waifufinder-remote/v1 c2s"`, `s2c` uses `"… s2c"`.
- Each frame is sealed with ChaCha20-Poly1305 under its direction's key. The nonce is a
  4-byte direction constant, `0x00000001` for client-to-server and `0x00000002` for
  server-to-client, followed by a per-direction 8-byte counter that starts at 1 and
  increments once per frame. The 24-byte header is authenticated as associated data, so
  an attacker cannot rewrite an opcode or a length.
- Replaying a sealed frame fails the counter check, and the connection is torn down.

CryptoKit is already linked for `SHA256` in `ModelInstaller`. No new dependency, and
nothing that touches the repository's licensing rule.

The reason for encrypting at all: the payloads are the user's own video frames. A
pre-shared key that only authenticated would let anyone on the LAN watch.

## Single connection, ordered work

One client at a time, as specified.

- The listener accepts a connection. While one is live, further connections are answered
  with `error{code: "busy"}` and closed immediately, before the handshake.
- **Kick** sends `close{reason: "kickedByServer"}`, cancels the socket, and releases the
  model references. The client shows *Disconnected — kicked by this Mac's owner* and
  does not reconnect until asked.
- All requests on the connection run through **one serial executor in arrival order**.
  There is no work stealing and no priority: a tagging batch and a reconstruction both
  serialise against the same GPU, and pretending otherwise would only move the queue.
- The client may keep two tag batches in flight, because `TaggingBatchPipeline` fetches
  the next batch while the current one is consumed. Two outstanding requests is the
  design's flow-control window; a third is rejected with `busy` rather than queued
  silently.
- The server pings every 15 s and drops a connection with no traffic for 60 s.

## Server behaviour while serving

The server is an ordinary WaifuFinder process that also answers requests. It uses the
same reference counting the UI uses, so the Models tab is truthful while remote work is
running:

- `ModelLibrary.shared.acquireTagger()` / `releaseTagger()` around a tag session, and
  `acquire(algorithm)` / `release(algorithm)` around a reconstruction. A model in remote
  use shows **In use** in Settings → Models and cannot be removed or disconnected.
- `ModelLibrary.shared.active` blocks server-side jobs, exactly as it already blocks
  `UncoverSession.reconstruct`.
- The tagger backend is held warm in an actor and created on first use, because the
  225 MB safetensors load is far too expensive to repeat per eight-frame batch. It is
  dropped after 60 s idle and immediately on `.waifuModelsChanged`.
- `.waifuModelsChanged` on the server pushes a fresh `capabilities` frame to the
  connected client, so the ᴿ markers update live as models are installed or removed.

The server refuses a job it cannot run and says why, in the same words the local UI
would use: `error{code: "modelUnavailable", message: "Install WD Tagger in Settings →
Models."}`.

### Capability document

```jsonc
{
  "protocol": 1,
  "appVersion": "1.4.0",
  "serverName": "Studio",
  "sessionID": "…",
  "tagger": {
    "available": true,
    "imageSize": 448, "batchSize": 8, "headCount": 10861,
    "vocabularyFingerprint": 8123456789012345678
  },
  "uncover": [
    { "algorithm": "seedVR2", "available": true, "message": "SeedVR2 3B · native Swift MLX upscaling",
      "scales": [1,2,4,8,16,32,64], "minimumFrames": 5, "requiresUnsafeMemoryOptIn": false,
      "maximumPaddedArea": null, "maximumOutputPixels": 16000000, "outputScale": 1 },
    { "algorithm": "pgtFormer", "available": false, "message": "Not installed on this server." }
  ],
  "vision": { "available": true }
}
```

It is `UpscalerCapabilities` verbatim, so `UncoverSession.canReconstruct` and
`UncoverSession.limitation` compute correctly on the client against the **server's**
limits without any remote-specific branch.

## Client behaviour

### Connection status is reachability

There is no separate health probe. One state machine is the answer to "can I reach the
server":

```text
offline → discovering → connecting → authenticating → connected
   ▲          │             │             │              │
   └──────────┴─────────────┴─────────────┴──────────────┘
        failed(reason)          reconnecting(backoff)
```

- Idle ping every 5 s; no pong within 20 s drops the connection.
- Reconnect uses exponential backoff, 1 s doubling to 30 s, with ±20 % jitter, and a new
  random client nonce each attempt.
- An authentication failure does not retry on a timer. The client waits for the user,
  because retrying a key that is known to be wrong only produces noise.

### Address

`host:port` or just `host`, default port 47410. Validated for a plausible host and a
1…65535 port before any socket is opened, in the spirit of `MediaImport.webURL` being
the one gate for pasted URLs. A port the server could not bind is reported in the panel
with the port it got instead, and in the beacon.

### First connection to an autodiscovered server

1. The client has a key for that `host:port` → connect.
2. Otherwise → show **Key needed** and offer a paste field with the key in plain text, no
   masking, because a masked 68-character field cannot be verified by eye.
3. Store it under `host:port` alongside the last-seen `keyID` and display name.

## The Settings → Remote tab

A fourth tab in `WaifuSettingsView`, `@AppStorage("settingsTab") == "remote"`, following
the existing Firefox tab's shape.

```text
REMOTE

  Serve inference from this Mac                    [ toggle ]
  Status: Advertising on 239.255.42.99:42888 · TCP 47410
  Address  [ 192.168.1.24:47410                 ] [Copy]
  Access key
    WF1-3F9A21C7-B40E5D18-…-A913                  [Reveal] [Copy] [Rotate…]
    Generated 17 Oct 2026 · never sent over the network, only HMACs are

  Connected client
    ● Studio MacBook Pro — 192.168.1.31 — tagging, 4 batches/s — since 00:04:11
                                            [Kick]
    (none) No client is connected.

  ─────────────────────────────────────────────────────────────

  Use an inference server                          [ toggle ]
  Status: Connected to Studio MacBook Pro · 1.4.0 · 3 models ready

  Discovered servers
    ● Studio MacBook Pro   192.168.1.31   1.4.0   Ready       [Connect] [Forget]
    ○ Attic Mac            192.168.1.40   1.3.2   Key needed [Enter key…]
    (none) No WaifuFinder servers are advertising on this network.

  Manual address  [ 192.168.1.40:47410          ] [Connect]
  Access key      [                                       ] (shown for manual)
```

Server controls and client controls are independent: this Mac may be both, and either
side may be off.

Starting and stopping is the server's prerogative at any moment. The toggle persists, so
the service returns on the next launch, and the stop button offers **Stop now** versus
**Stop after the current job** rather than silently abandoning a reconstruction.

## The ᴿ superscript

A label shows a raised `R` when — and only when — the request would actually be executed
on the server: the connection is `connected`, and the server advertises the specific
capability that request needs.

```swift
/// Appends a superscript "R" to an inference control's label when that request
/// will be executed on the connected server.
struct RemoteMark: ViewModifier {
    let available: Bool
    func body(content: Content) -> some View {
        HStack(spacing: 1) {
            content
            if available {
                Text("R").font(.system(size: 9, weight: .semibold))
                    .baselineOffset(5).foregroundStyle(.secondary)
                    .accessibilityLabel("runs on the connected server")
            }
        }
    }
}
```

It lands on three places, because they answer three different questions:

| Control | Question | Marked when |
|---|---|---|
| `Start tagging` (both the panel button and `TaggingStartButton`) | can this machine's frames be tagged at all | connected and `tagger.available` |
| `Uncover Waifu` in the preview header | is there anything to open a remote session for | connected and **any** `uncover[].available` |
| `Reconstruct` in `UncoverView` | will *this* algorithm run remotely | connected and `uncover[selectedAlgorithm].available` |

The distinction matters. `Uncover Waifu` only opens the workspace, so marking it says
"the server can do something here". `Reconstruct` is the actual inference start, so it is
marked per selected algorithm and disappears when the client switches to an algorithm the
server does not have — at which point the button silently becomes local again, which is
the correct behaviour and needs no second label.

Each marked control also gets a `.help()` naming the server, so the marker is
self-explaining rather than a glyph the user has to decode.

## Layering

`Contracts.swift` declares the order and this amends it:

```text
Model ─┬─► Video ─► Vision ─► Grouping ─┐
       ├─► Models ──────────────────────┼─► Remote ─► Engine ─► UI
       └─► Uncover ─────────────────────┘
```

`Remote` sits above `Model`, `Models` and `Uncover`, and below `Engine`. It may not
import `UI`. `Engine` and `UI` depend on `Remote`; nothing in `Remote` knows a
`WaifuSession` exists. That is what keeps the remote backends testable against a real
socket with no app state at all.

## Files

```text
Sources/Waifu/Remote/WireProtocol.swift         header, opcodes, WireReader/WireWriter, framing
Sources/Waifu/Remote/RemoteCrypto.swift         key generation, hex/check format, HKDF, HMAC, sealing
Sources/Waifu/Remote/RemoteBeacon.swift         beacon encode/decode, multicast sender, receiver
Sources/Waifu/Remote/RemoteCapabilities.swift   the capability document and its codable shape
Sources/Waifu/Remote/RemoteKeyStore.swift       protocol + Keychain + file implementations
Sources/Waifu/Remote/RemoteServer.swift         listener, one session, serial executor, clients, kick
Sources/Waifu/Remote/RemoteClient.swift         connection state machine, beacon cache, request table
Sources/Waifu/Remote/RemoteTaggerBackend.swift  TaggerBackend over the socket
Sources/Waifu/Remote/RemoteUncoverBackend.swift TemporalVideoUpscaler over the socket
Sources/Waifu/Remote/RemoteSettingsView.swift   the Settings tab
Sources/Waifu/UI/RemoteMark.swift               the ᴿ label modifier
```

Touch points outside that list, deliberately few:

- `SettingsView.swift` — one more `tabItem`.
- `WDTaggerBackend.swift` — extract `probabilities(batch:)` from `tag(batch:)`, so the
  remote path can ship rows without a threshold baked in. `tag(batch:)` becomes a thin
  caller of it, and the existing tests keep exercising the same code.
- `WaifuSession.swift` — vocabulary fallback in `startTagging`, and the two backend
  factory lines in `runTaggingPipeline` and `openUncover(at:)`.
- `TaggingPanelStatus.swift`, `FramePreviewView.swift`, `UncoverView.swift` — `.modifier(RemoteMark(available:))`.
- `build.sh` — `NSLocalNetworkUsageDescription` in `Info.plist`, plus `-framework
  Network` and `-framework Security`. `run-tests.sh` needs `-framework Network` too.

`Network.framework` is Apple; `Security` is Apple; nothing in this design adds a
dependency of any licence.

## Tests

`Tests/RemoteProtocolTests.swift`, registered in `Tests/main.swift`. All of it runs over
a real `NWListener` bound to `127.0.0.1:0` on loopback, so no local-network permission
prompt and no beacon are involved.

- **Framing** — round trip for every opcode; a message split across frames reassembles;
  wrong magic, wrong version, reserved-nonzero and `byteCount` over the 1 MiB cap are all
  rejected without allocating; a truncated stream is an error, never a partial message.
- **Crypto** — key hex and check round trip, a flipped character is rejected, HKDF and
  HKDF vectors are pinned, a flipped ciphertext bit fails the tag, a repeated nonce
  fails, and a rewritten header fails because it is authenticated as associated data.
- **Beacon** — encode/decode round trip including a maximal name and version, expiry
  after three missed beats, and `keyID` matching the stored key.
- **Capabilities** — codable round trip including `null` limits and the non-commercial
  model being absent rather than present-and-false.
- **End to end** — real handshake, `capabilities` push, an eight-frame `tagBatch`
  answered with eight rows of 10 861 floats that match a local `WDTaggerBackend` run on
  the same images, a vocabulary round trip whose fingerprint matches
  `TagVocabulary.cacheFingerprint`, an Uncover job whose returned PNGs are byte-identical
  to a local `UncoverTestBackend` run, cancellation mid-job, kick, and busy rejection of a
  second connection.

The interesting assertion is the byte-identical one: it is what makes "remote results are
the same as local results" a tested claim rather than a comment.

## Not in scope, and why

- **Several clients.** One connection, as specified. The serial executor is already the
  right shape for a queue, but there is no per-client identity, quota or fairness
  policy, and adding one would be a design of its own.
- **Bonjour.** The beacon covers discovery as asked for. Bonjour would add an
  `NSBonjourServices` entry and a browse path for no gain here.
- **Video streaming to the client.** Analysis, grouping, thumbnails and export all stay
  on the client. Only model inputs and model outputs move.
- **Progress for Vision.** Vision runs in-process on the client, needs no model files and
  is not remote. The capability document says so explicitly rather than leaving the
  client to guess.