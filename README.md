# Video Query Rust (`vq`)

`vq` is a new, standalone Rust command-line implementation of local semantic
video search. It does not contain the original iOS project's Swift, Objective-C,
project files, resources, or copied implementation code.

The command scans video frames, ranks keyframes in Rust, embeds selected images
with a local Chinese-first image/text model, persists the vectors in SQLite, and
searches them with Chinese or English text. It can also transcribe speech from
audio or video files completely locally.

## What is implemented

- Streaming, bounded-memory frame analysis for MP4, MOV, MKV, WebM, AVI, MTS,
  and M2TS files.
- Allocation-free Rust luma scoring using Laplacian variance, gradient energy,
  exposure, contrast, and clipping penalties.
- Best-frame selection near a timestamp or within fixed time segments.
- Batched, native Rust model inference through Hugging Face Candle.
- Local Chinese/English text-to-image retrieval with
  `OFA-Sys/chinese-clip-vit-base-patch16`.
- Persistent SQLite/WAL index with content fingerprints and incremental skips.
- Parallel exact cosine search with Rayon.
- Human-readable output and `--json` automation output.
- Local multilingual audio/video transcription with timestamped segments.

FFmpeg and ffprobe are used as the codec boundary. Orchestration, frame quality
analysis, visual preprocessing and inference, indexing, ranking, and CLI logic
are implemented in Rust. Speech inference is delegated to the separately
installed whisper.cpp runtime so Apple Silicon can use its Metal backend.
Keeping codec decode in FFmpeg gives reliable hardware/format support without
importing any code from the iOS application.

## Why this embedding model

The default is [Chinese-CLIP ViT-B/16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16):
an Apache-2.0, approximately 188M parameter dual encoder trained specifically
on Chinese image/text pairs. Its weights are about 753 MB and it runs locally
on CPU, so it is considerably more affordable than current 2B+ multimodal
embedding models while preserving strong Chinese retrieval behavior.

This is an intentional cost/Chinese-accuracy choice, not a claim that the 2022
checkpoint is the largest or newest model available. Newer models such as
[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) can
be stronger but require several times more memory and compute. The index records
the exact model identity and refuses to mix vectors from incompatible models.

## Why this speech model

The default speech model is
[`SenseVoiceSmall Q8`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF),
paired with the 1.7 MB
[`FSMN-VAD`](https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF). SenseVoice is
a Chinese-first, non-autoregressive 234M-parameter model that recognizes
Mandarin, Cantonese, English, Japanese, and Korean. It also reports language,
emotion, and audio-event tags such as music, applause, laughter, coughing, and
sneezing.

The Q8 weights are about 235 MB. The upstream 184-clip Mandarin benchmark
reports 8.17% character error for Q8 SenseVoice versus 23.15% for
whisper.cpp large-v3-turbo on the same CPU-oriented test. Results depend on
the recording domain, but this is a much better cost/accuracy fit for a
Chinese-first local CLI.

`vq` downloads a checksum-pinned native FunASR runtime for Windows x64, macOS
ARM64, or Linux x64/ARM64. No Python environment or API key is required.
Whisper remains available as an optional engine when timestamped segments,
macOS Intel, or languages outside SenseVoice's released five-language
checkpoint matter more.

## Install a release

Prebuilt `vq` archives are published for:

- Windows x64
- Linux x64 and ARM64
- macOS Apple Silicon and Intel

Download the archive for your platform from the
[latest GitHub release](https://github.com/jo32/video-query-rs/releases/latest),
verify it against `SHA256SUMS`, extract it, and put `vq` (or `vq.exe`) on
`PATH`. Each archive also contains the license and third-party notices.

The CLI binary is self-contained, but FFmpeg/ffprobe remain external codec
dependencies. Models and the small SenseVoice runtime are verified and
downloaded on first use; model weights are not bundled in release archives.

## Requirements

- Windows 10/11, Linux, or macOS
- Rust 1.91 or newer only when building from source
- FFmpeg and ffprobe on `PATH`
- Optional: [uv](https://docs.astral.sh/uv/) for the project-local `yt-dlp`
  video downloader
- Apple Accelerate is enabled on macOS
- Roughly 1.2 GB free for both default model caches, plus extracted keyframes
- Optional: `whisper-cli` for `vq transcribe --engine whisper`

On macOS:

```sh
brew install ffmpeg rust

# Optional timestamped/broader-language transcription:
brew install whisper-cpp
```

On Windows, install FFmpeg through your preferred package manager and ensure
`ffmpeg.exe` and `ffprobe.exe` are on `PATH`. SenseVoice is the Chinese-first
default on Windows x64, Linux x64/ARM64, and macOS Apple Silicon. On macOS
Intel, install `whisper-cli` and use the Whisper engine.

## Build

```sh
git clone https://github.com/jo32/video-query-rs.git
cd video-query-rs
cargo build --release
./target/release/vq doctor
```

The executable is `target/release/vq` (`target\release\vq.exe` on Windows).

## Download a video

Install the pinned project tools and download a video with `yt-dlp`:

```sh
uv sync
uv run yt-dlp "VIDEO_URL"
```

For the best available video and audio merged into an MP4 file:

```sh
uv run yt-dlp -f "bv*+ba/b" --merge-output-format mp4 "VIDEO_URL"
```

Only download content you have permission to save and follow the source site's
terms. `yt-dlp` invokes the separately installed FFmpeg when it needs to merge
or convert media.

## Usage

Check whether all image and speech models/runtimes are already available without
accessing the network:

```sh
vq model status
vq --json model status
```

Download every model/runtime supported on the current platform: Chinese-CLIP,
SenseVoice, FSMN-VAD, the native SenseVoice runtime, and the optional Whisper
fallback. SenseVoice is skipped where no compatible native runtime exists:

```sh
vq model fetch
```

The engine-specific commands remain available when you only want one speech
stack. Transcription also downloads the selected engine's missing components
lazily, while index/search lazily download Chinese-CLIP:

```sh
vq model status-speech
vq model fetch-speech

# Optional Whisper fallback:
vq model status-whisper
vq model fetch-whisper
```

Download state is written to stderr as stable, agent-readable lines. Progress
is reported every 5%, while JSON command results remain clean on stdout:

```text
[model] status=missing files=pytorch_model.bin action=download
[model] download-start file=pytorch_model.bin total_bytes=753177983 total=718.3 MiB
[model] download-progress file=pytorch_model.bin percent=25 downloaded_bytes=... total_bytes=753177983
[model] download-complete file=pytorch_model.bin bytes=753177983
[model] status=ready action=continue
```

The cache honors `HF_HOME`; otherwise it defaults to
`~/.cache/huggingface/hub`. Empty or missing files are treated as unavailable.

Transcribe Mandarin, Cantonese, English, Japanese, Korean, or mixed speech from
any FFmpeg-readable audio or video file. SenseVoice is the default:

```sh
vq transcribe meeting.m4a
vq transcribe interview.mp4 --language zh --metadata
vq transcribe podcast.mp3 --output transcript.txt
vq --json transcribe meeting.m4a --output transcript.json
```

SenseVoice detects the spoken language; `--language` acts as an expected-language
check and emits a warning on a mismatch. JSON output always contains detected
language, emotion, and audio-event fields. Add `--metadata` to include them in
human-readable output.

Use Whisper when segment timestamps or a broader language set is required:

```sh
vq transcribe interview.mp4 --engine whisper --language zh --timestamps
vq transcribe lecture.mp3 --engine whisper --prompt "专有名词：滨海新区，抚养权"
```

SenseVoice does not currently expose segment timestamps through its native
runtime. All inference stays on the local machine after the first fetch.

Find one high-quality frame every ten seconds without loading an ML model:

```sh
vq keyframes movie.mp4 --segment-seconds 10 --output-dir ./keyframes
```

Find the sharpest frame within three seconds of a timestamp:

```sh
vq keyframes movie.mp4 --at 60 --radius 3 --output-dir ./keyframes
```

Index a file or a directory recursively:

```sh
vq index ~/Movies --segment-seconds 10
```

Search in Chinese:

```sh
vq search "海边奔跑的一只狗"
```

Use `--json` before the subcommand for machine-readable results:

```sh
vq --json search "夜晚的城市街道" --limit 20
```

By default, the index is stored under `~/.video-query`. Override it with
`--index-dir <directory>` or the `VQ_INDEX_DIR` environment variable.

## Keyframe algorithm

FFmpeg samples and scales frames to a maximum 640-pixel long edge, then streams
8-bit luma planes over a pipe. Rust evaluates each frame without intermediate
image allocations:

```text
focus = sqrt(var(laplacian)) + 0.35 * RMS(sobel-like gradient)
score = focus * exposure_weight * clipping_weight * contrast_weight
```

The scale cap makes work predictable for 4K/8K input, while fixed-size raw
frames make the hot loop cache-friendly. For semantic indexing, the best-scoring
frame in each time segment is decoded once at higher resolution and embedded in
batches.

## Index layout

```text
~/.video-query/
├── index.sqlite3       video metadata, frame scores, and 512-f32 vectors
└── frames/<fingerprint>/
    └── *.jpg           selected frames returned by search
```

Changing model/schema identity is detected on open. Use a separate index
directory after such a change.

## Verification

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Unit tests cover scoring, padded frame storage, segment selection, vector
serialization, model metadata safety, and ranking. The integration test creates
a real video with FFmpeg and exercises decode, scoring, selection, and JPEG
extraction.

## License

This implementation is MIT licensed. Chinese-CLIP is separately distributed
under Apache-2.0 and is downloaded from its official Hugging Face repository.
