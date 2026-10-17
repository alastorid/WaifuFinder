# Remote inference

A Mac can share its installed tagging and reconstruction models with one other Mac.
The client owns browsing, Apple Vision analysis, grouping, source media, compositing
and export. The server executes model inference. Settings → Remote controls both
roles independently; enabling a role is an explicit action in the current app session.

## Integration

`RemoteTaggerBackend` implements `TaggerBackend`; `RemoteUncoverBackend` implements
`TemporalVideoUpscaler`. `WaifuSession` selects a remote backend when the connected
server advertises the relevant capability, and otherwise selects the local backend.
An Uncover workspace captures its backend choices when opened, so a running job does
not change machines when capabilities change. Reopen the workspace to choose again.

The server sends full Float32 tagging probabilities. The client applies its threshold
with `TagVocabulary.result`, preserving the normal tagging cache format. The remote
vocabulary is downloaded, checked against its fingerprint and published to the filtering
and tagging UI. The client needs labels, but does not need local model weights.

Reconstruction sends the prepared manifest and `degraded/` PNGs. `original/` and
`baseline/` stay on the client. Reconstructed PNGs are written into the client's own
`reconstructed/` directory; existing rendering and export code handles the result.
The manifest includes source metadata, including its path; it does not upload source
video contents. A raised **R** on an inference control identifies remote execution,
with a tooltip naming the server. Vision always runs locally.

## Connection and keys

The listener uses TCP port **47410** by default. Manual addresses accept a hostname,
IPv4 address or bracketed IPv6 address with an optional port. Discovery uses a small
UDP beacon every second on **239.255.42.99:42888**, plus best-effort broadcast to
**255.255.255.255:42888**. Entries expire after three missed beats. The source IP of
the datagram is the address to dial; its advertised name is display data only.

A beacon contains version, actual TCP port, process instance, capability flags, name,
app version, and an eight-byte `keyID`. It contains no access key. Beacon claims are
not trusted to authorize work; TCP authentication is always required. Manual addresses
work on networks that block discovery.

The access key is 32 random bytes. Its display form is `WF1-`, eight groups of eight
hex characters, then a four-character checksum. The checksum catches typing errors.
`keyID` is the first eight bytes of SHA256(`"WF1/keyid"` + key); the check group is the
first two bytes of SHA256(`"WF1/check"` + key).

Server and client keys use separate Keychain accounts. Client accounts are keyed by
normalized server address. A private `0600` file is used if Keychain storage fails.
Tests inject file stores and never need login Keychain access. Rotation replaces the
server key and disconnects the current session. Kick disconnects the client without
changing the key. Refused keys and explicit server disconnections require user action;
transient network errors use reconnect backoff from one to thirty seconds.

## Authenticated transport

This is a custom PSK challenge/response protocol using CryptoKit primitives. It is
not an implementation of Noise and offers no forward secrecy if the PSK is later
compromised. Both Macs must be trusted with the frames and model outputs they receive.

1. Client sends `hello` with protocol version, app version, name and a fresh 32-byte nonce.
2. Server sends `challenge` with its name, version, fresh 32-byte nonce and key identifier.
3. Client sends HMAC-SHA256(key, `"WF1/auth"` + clientNonce + serverNonce).
4. Server installs session keys and sends an **encrypted** `authResult` containing a
   session identifier, capabilities and HMAC-SHA256(key, `"WF1/server"` + sessionID +
   clientNonce + serverNonce). The client requires and verifies this proof.

HKDF-SHA256 derives separate 32-byte direction keys, with the two nonces as salt and
info `"waifufinder-remote/v1 c2s"` or `"waifufinder-remote/v1 s2c"`. ChaCha20-Poly1305
seals each subsequent frame. Its 12-byte nonce is a four-byte direction constant
(1 for c2s, 2 for s2c) followed by an eight-byte increasing counter, both big-endian.
The counter starts at one. The header is authenticated as associated data. Authentication
failure, replay or malformed framing closes the connection. Parsing pauses at the
client proof so buffered encrypted bytes cannot be interpreted as plaintext during
key installation.

## Framing and payloads

Each frame has a 24-byte header, followed by payload and, after key installation,
a 16-byte authentication tag. Header integers are little-endian:

| Offset | Bytes | Field |
|---|---:|---|
| 0 | 4 | `WFR1` |
| 4 | 2 | version, 1 |
| 6 | 2 | opcode |
| 8 | 4 | flags: final, zlib, error |
| 12 | 4 | stream ID |
| 16 | 4 | plaintext payload length |
| 20 | 4 | reserved, zero |

Frame payloads are limited to 1 MiB and messages to 64 MiB. Fragments of a message
are contiguous and share opcode, stream and flags; only the final fragment sets final.
A reconstruction has at most 300 input frames and 64 MiB of aggregate uploaded PNGs.
Length-prefixed readers validate remaining bytes before reading or allocating.

Connection control uses stream zero. Requests use nonzero client-assigned stream IDs;
responses and progress use the same ID. Protocol cancellation can refer to the request
ID on stream zero. App-side task cancellation closes the current connection, waking
all its readers and cancelling server work; normal transient reconnect can follow.
The client allows two outstanding requests. Buffered messages are capped at 64 MiB
and 1,024 envelopes. Model work runs in arrival order on one serial executor.

JSON encodes control values, including Codable handshake nonces and proofs. Media,
probability rows and compressed vocabulary travel as binary payloads, never JSON.
`WireWriter` and `WireReader` define their little-endian binary layout:

- Tag batch: u8 count (1–8), u8 codec (0 encoded image, 1 RGBA8), u16 reserved zero,
  then count × { u32 byte length, bytes }. RGBA8 is 448×448. The current backend
  renders model input pixels on the client; the encoded convenience API creates JPEGs.
- Tag result: u8 count, u64 vocabulary fingerprint, then count × { u32 byte length,
  little-endian Float32 probability row }. Each row has 10,861 heads.
- Vocabulary: u64 fingerprint, u32 uncompressed count, then a compressed envelope
  (u64 uncompressed count, Compression-framework zlib bytes). Both size and vocabulary
  fingerprint are checked.
- Reconstruction: `uncoverBegin` control, one `uncoverInput` binary PNG per manifest
  frame, and `uncoverInputEnd`. Reply: progress controls, result metadata (model,
  seconds, outputScale), one `uncoverResult` PNG per frame, and `uncoverResultEnd`.
  Both sides consume the terminator, even when they already have the expected count.

Opcodes and control Codable definitions live in `WireProtocol.swift`. Compressed
messages are reassembled before decompression, whose declared size is bounded.
The tagger stays warm across requests, with model references reflected in Settings →
Models. Reconstructions acquire and release the same model references as local jobs.

## Ownership and verification

`RemoteLink` owns TCP framing and encryption state in one actor. It submits an entire
message before suspending, preserving fragment and cipher-counter order across tasks.
`RemoteConnection` owns per-stream inboxes. `RemoteServerSession` owns authentication
and request dispatch; `RemoteModelCatalog` supplies models through injectable closures.
The production catalog uses the local installed backends. Neither remote backend
needs to know the UI's progress, cache or export implementation.

`Tests/RemoteProtocolTests.swift` covers binary bounds, splitting, key formatting,
directional derivation, replay and ciphertext/header tampering, compression, beacon
expiry, capabilities and labels. End-to-end cases use real loopback sockets with
injected deterministic model backends: admission/refusal, vocabulary verification,
bit-identical tagging rows, PNG round trips, cancellation with model-reference release,
busy rejection of a second client, kick and reconnect. These tests validate
the transport and integration; they do not measure hardware model accuracy or LAN
multicast routing. Run `./run-tests.sh --only remote` or the full `./run-tests.sh`.

The app links only Apple frameworks for this feature: Network, CryptoKit, Compression
and Security. No third-party dependency or media executable is added. Interactive
Settings checks must use `./run-ui-tests.sh SCRIPT [ARGUMENTS...]` as required by
`AGENTS.md`.
