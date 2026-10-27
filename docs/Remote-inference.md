# Remote inference

A Mac can share its installed tagging, reconstruction, SAFIRE and TruFor models with one other Mac.
The client owns browsing, Apple Vision analysis, grouping, source media, compositing
and export. The server executes model inference. Settings → Remote controls both
roles independently. Client server entries retain their address and enabled state.

## Integration

`RemoteTaggerBackend` implements `TaggerBackend`; `RemoteUncoverBackend` implements
`TemporalVideoUpscaler`. `WaifuSession` selects a remote backend when the connected
server advertises the relevant capability, and otherwise selects the local backend.
An Uncover workspace captures its backend choices when opened, so a running job does
not change machines when capabilities change. Reopen the workspace to choose again.

`RemoteInferencePool` owns one independently authenticated `RemoteClient` per configured
server. The primary connection also serves Uncover. Tagging snapshots connected,
deduplicated endpoints with the selected vocabulary fingerprint; incompatible labels
are excluded. `BalancedTaggerBackend` reserves the least-loaded lane before sending
a batch, counting requests through preparation, transport and completion. Equal depths
rotate fairly. Each remote lane has an eight-batch window. Option-start adds a local
lane with capacity two, after verifying that local labels match. Each frame is inferred
once. Source batches enter the scheduler before pixel loading, so preparation runs
independently in the selected lanes instead of serially limiting every server. Video
thumbnails upload as the original cached JPEG bytes, without client decoding, resizing
or another lossy encoding. Photos upload the exact model-input RGBA bytes; peers that
advertise `rgba8LZ4` receive losslessly compressed inputs, with a raw fallback when
compression would enlarge a frame. Older version-2 servers keep receiving raw RGBA8.
Image decoding on the server runs off its request-dispatch actor. Vocabulary downloads
from connected servers start concurrently. Completed frame updates commit immediately
so a slow server cannot hold up
refilling a faster server's queue. Follow-up tagging preserves the local participation
choice; cancellation joins in-flight tasks before releasing local model resources.

The native worker reads Float32 model output and quantizes it on CPU before IPC. Replies
carry exactly four UInt8 rating probabilities plus sparse UInt8 trait scores,
LZ4-compressed using the same `WDT4` payload retained locally. There is no winner
byte. The client derives the rating and applies its display threshold. Trait scores
use the fixed 0.35 storage floor; all four ratings retain their full 0...1 range. The remote
vocabulary is downloaded, checked against its fingerprint and published to the filtering
and tagging UI. The client needs labels, but does not need local model weights.

Reconstruction sends the prepared manifest and `degraded/` PNGs. `original/` and
`baseline/` stay on the client. Reconstructed PNGs are written into the client's own
`reconstructed/` directory; existing rendering and export code handles the result.
The manifest includes source metadata, including its path; it does not upload source
video contents. A raised **R** on an inference control identifies remote execution,
with a tooltip naming the server. Vision always runs locally.

## SAFIRE and TruFor

SAFIRE spatial analysis (photo preview and lineage groups) and TruFor preview evidence
prefer the first connected compatible server, using local inference when no compatible
server is connected. The client needs no local weights for remote execution. A job
captures its selected server; a failure is reported without silently changing models.
Existing SAFIRE evidence is reused only for the selected checkpoint and photo identity.
TruFor album scoring distributes batches across compatible servers; Option-start also
includes this Mac when local weights are installed. Duplicate endpoints form one lane.

Settings → Remote lists SAFIRE and TruFor readiness on each server card and advertises
both in discovery. Photo preview controls show the raised **R** when a compatible server
is selected. SAFIRE requires matching runtime and preprocessing revisions and a valid
checkpoint fingerprint. TruFor additionally pins the exact weights and FP32 execution
revision, and remains available only in noncommercial builds.

Both paths send lossless PNG input through the authenticated encrypted connection.
SAFIRE returns binary spatial probabilities and source partitions with photo/content
identity. TruFor returns ordered scalar scores or streams binary anomaly, reliability
and Noiseprint++ maps in bounded chunks followed by an explicit terminator. Native
Swift workers execute inference on the server. Canceling either forensic request stops
only its stream, preserving the connection for other inference jobs.

## Connection and keys

The listener uses TCP port **47410** by default. Manual addresses accept a hostname,
IPv4 address or bracketed IPv6 address with an optional port. Discovery uses a small
UDP beacon every second on **239.255.42.99:42888**, plus best-effort IPv4 subnet
broadcast on each active network interface. Broadcast uses a UDP socket because
Network.framework does not support UDP broadcast. Entries expire after three missed
beats. The source IP of the datagram is the address to dial; its advertised name is
display data only.

A beacon contains version, actual TCP port, process instance, capability flags, name,
app version, and an eight-byte `keyID`. It contains no access key. Beacon claims are
not trusted to authorize work; TCP authentication is always required. Manual addresses
work on networks that block discovery.

The access key is 32 random bytes. Its display form is `WF1-`, eight groups of eight
hex characters, then a four-character checksum. The checksum catches typing errors.
`keyID` is the first eight bytes of SHA256(`"WF1/keyid"` + key); the check group is the
first two bytes of SHA256(`"WF1/check"` + key).

Server and client keys are stored in separate files under the app's Application
Support directory with `0600` permissions. Client files are keyed by normalized server
address. The app does not access Keychain; any existing Keychain entries are left
untouched and are not imported. Rotation replaces the server key and disconnects the
current session. Kick disconnects the client without changing the key. Refused keys and
explicit server disconnections require user action; transient network errors use
reconnect backoff from one to thirty seconds.

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
| 4 | 2 | version, 2 |
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
Remote tagging keeps up to eight batches (64 images) in flight **per server**. Two
servers have independent windows of eight, for sixteen pending batches in total.
Image preparation, upload, server inference and result delivery overlap; independent
frame updates commit on completion so a faster server keeps taking new work. The
server bounds tagging tasks to eight and the model serializes GPU
access. Two additional request slots allow control work. Buffered messages remain
capped at 64 MiB and 1,024 envelopes. Cancelling the pipeline joins pending work.

JSON encodes control values, including Codable handshake nonces and proofs. Media,
probability rows and compressed vocabulary travel as binary payloads, never JSON.
`WireWriter` and `WireReader` define their little-endian binary layout:

- Tag batch: u8 count (1–8), u8 codec (0 encoded image, 1 RGBA8, 2 RGBA8/LZ4),
  u16 reserved zero, then count × { u32 byte length, bytes }. RGBA8 is 448×448.
  Codec 2 requires the server's optional `tagger.inputCodecs` capability to include
  `2`. Each frame begins with a mode byte: `0` followed by exactly 802,816 raw bytes,
  or `1` followed by Apple LZ4 bytes that decode to exactly 802,816 bytes. In either
  case model pixels are unchanged. Cached video JPEGs use codec 0 verbatim.
- Tag result: u8 count, u64 vocabulary fingerprint, then count × { u32 byte length,
  `WDT4` bytes }. Each row decompresses to four rating bytes, a UInt16 trait-pair
  count, then sparse UInt16 head index / UInt8 score pairs. Apple LZ4 compression
  happens before encryption. The client validates count, fingerprint, lengths,
  head order and score codes; the frame pipeline keeps the received bytes without
  converting them back to Float32 for re-encoding. Dense score arrays are only
  temporary inputs to vocabulary label reconstruction.
- Protocol version 2 requires both peers to be updated. Older Float32 responses
  are unsupported; the handshake rejects protocol version 1.
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
expiry, capabilities and labels. SAFIRE and TruFor loopback cases also cover spatial evidence,
score order, incompatible revisions and cancellation without disconnecting. End-to-end cases use real loopback sockets with
injected deterministic model backends: admission/refusal, vocabulary verification,
identical local/remote UInt8 score payloads and all four rating probabilities, PNG round trips, cancellation with model-reference release,
busy rejection of a second client, kick and reconnect. These tests validate
the transport and integration; they do not measure hardware model accuracy or LAN
multicast routing. Run `./run-tests.sh --only remote` or the full `./run-tests.sh`.

Tagging scaling regressions fill two independent servers' request windows before
allowing model replies, verify byte-exact cached-JPEG and lossless RGBA input, and
compare one versus two servers over real encrypted loopback sockets. The throughput
case uses a serialized 150 ms model stub on each server and 128 generated images;
it checks scheduling and transport scaling, not physical GPU or LAN performance.
The October 26, 2026 run completed in **2.55 s with one server** and **1.29 s with
two servers (1.98× throughput)**. Both servers filled their eight-request windows;
cached-JPEG and lossless-RGBA pixel parity checks passed.

The app links only Apple frameworks for this feature: Network, CryptoKit and Compression. No third-party dependency or media executable is added. Interactive UI tests
are prohibited by `AGENTS.md`; use noninteractive protocol and runtime checks.


## RealESRGAN photo reconstruction

The existing reconstruction transport also carries native photo requests. x2 and
x4 advertise distinct algorithm identifiers plus optional photo-model fingerprint
and runtime revision fields. Older servers remain valid for their existing
reconstruction jobs; clients do not infer native-photo support from those older
capabilities. A compatible enabled server gives the corresponding x2 or x4 button
the existing raised R indicator.

Native photo requests pin the variant, official checkpoint fingerprint and runtime
revision in the prepared manifest. The server executes the same native Swift/Core
ML backend as local inference. Replies must name the selected model, report its
native scale, and decode to the exact expected dimensions. Cancellation abandons
only the photo stream through the existing cancel-job control, and a two-minute
photo timeout cancels that stream. Ordinary video reconstruction retains its
existing cancellation and timeout policy. Stale preview generations cannot commit
late results. Prepared upload/result PNGs are temporary and removed after the job;
completed preview images are cached in RAM with a 128 MiB bound.
