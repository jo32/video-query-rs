# Video analysis workflow reference

## Pipeline contract

`prepare_video.py` accepts a URL or local video and creates a self-contained analysis bundle. It intentionally separates deterministic evidence gathering from Codex interpretation. The bundled `bootstrap_vq.py` prefers signed GitHub release metadata plus `SHA256SUMS`; it builds source only when no compatible release can run. `bootstrap_whisper.py` installs checksum-pinned official whisper.cpp runtimes on Windows/Linux when timestamped ASR is required.

```text
<analysis-dir>/
├── manifest.json
├── report.md                         # created after synthesis
├── source/                           # downloaded video and yt-dlp sidecars
└── raw/
    ├── metadata.json
    ├── description.txt               # when available
    ├── video-probe.json
    ├── transcript.json               # authoritative timed transcript
    ├── transcript.md
    ├── transcript-sensevoice.json    # optional Chinese-first wording check
    ├── candidates.json
    ├── analysis.json                 # Codex fills this
    ├── keyframe-selection.json
    ├── keyframe-observations.json    # Codex fills observations
    ├── index-stats.json
    ├── index/
    ├── keyframes/
    ├── search/
    └── logs/
```

For URL input, yt-dlp requests the cleaned info JSON, description, thumbnail, authored subtitles, and automatic subtitles. It limits subtitle languages to Chinese, Cantonese, English, Japanese, and Korean by default; override with `--subtitle-languages` when the source is known to use another language. Comments are deliberately excluded because they are expensive, noisy, and may introduce personal data.

If a usable timed subtitle exists, it becomes `transcript.json` and no redundant ASR is run. Otherwise the script uses `vq transcribe --engine whisper --timestamps`; during this ASR fallback it also produces a SenseVoice transcript by default as a Chinese-first full-text cross-check. SenseVoice has stronger Chinese recognition in this project but does not expose segment timestamps, so never substitute its untimed text for exact temporal evidence.

Model acquisition follows the first command that needs each model. Subtitle-backed preparation does not touch ASR caches. Whisper is checked immediately before ASR fallback, SenseVoice immediately before its optional cross-check, and Chinese-CLIP immediately before indexing. The relevant `vq` command then downloads only its own missing model. The manifest records `model_policy` and before/after readiness under `models`. `--no-model-fetch` changes this to cache-only operation.

If the media has neither subtitles nor an audio stream, preparation records an explicit `engine: none` transcript and continues with timeline coverage and visual evidence. The final report must disclose that no language evidence was available.

The index is isolated under the analysis directory. This prevents semantic results from unrelated videos. Candidate selection combines:

- explicit visual language such as “chart,” “on screen,” “如图,” “曲线,” “演示,” or “代码”;
- quantitative claims and topic-transition phrases;
- chapter starts when publisher chapters exist;
- evenly spaced coverage so visually important silent moments are not completely omitted.

Each candidate is refined with `vq keyframes --at` for a sharp nearby still. The highest-ranked cues are also used as Chinese-CLIP queries against the isolated index. Similarity is only a discovery signal.

## `raw/analysis.json` schema

Keep the JSON valid UTF-8 and preserve these keys:

```json
{
  "summary": "A concise, evidence-backed account of the whole video.",
  "key_points": [
    "Major point with a timestamp such as 03:12.000."
  ],
  "topics": [
    "Primary topic",
    "Secondary topic"
  ],
  "timeline": [
    {
      "start_seconds": 0.0,
      "end_seconds": 95.0,
      "topic": "Opening thesis",
      "evidence": "The speaker states ... at 00:14.200; a title slide appears at 00:16.000."
    }
  ],
  "limitations": [
    "Automatic subtitles are uncertain for product names."
  ]
}
```

Do not put speculative visual conclusions in `analysis.json` before opening the relevant images. It is fine to revise the file after visual inspection.

## Keyframe observation contract

`keyframe-observations.json` already contains selection facts. Preserve `id`, `source`, `timestamp_seconds`, `path`, `query`, reasons, similarity, and quality fields. Fill:

```json
{
  "observation": "Visible scene, chart, code, slide, people, or objects.",
  "visible_text": "Important legible text; empty when none is reliable.",
  "relevance": "How the image relates to the nearby spoken claim.",
  "confidence": "high"
}
```

Use `low` confidence for tiny or blurred text. Do not silently repair visible text using the transcript; state disagreements.

## Manual recovery commands

Use the exact `vq` executable in `manifest.json` under `tools.vq.path` and the video under `manifest.video`.

Extract a sharper frame around a known moment:

```sh
"<vq>" --json keyframes "<video>" \
  --at <seconds> --radius 4 \
  --output-dir "<analysis-dir>/raw/keyframes/manual/<name>"
```

Search the isolated visual index in Chinese or English:

```sh
"<vq>" --json --index-dir "<analysis-dir>/raw/index" \
  search "<concrete visual description>" --limit 5
```

Good queries describe visible content: “带红色下降曲线的白底图表” is better than an abstract claim such as “the economy is bad.” Add manually chosen frames to the observation JSON with `source` set to `manual-timing` or `manual-embedding`.

## Failure and cost handling

- `vq` absent or unhealthy: download the matching asset from `jo32/video-sherlock` releases and verify `SHA256SUMS`. If that is unavailable or cannot run, build the current checkout or clone the requested GitHub tag and run `cargo build --release --locked`.
- Models absent: do not run broad `vq model fetch`. Let the active `vq transcribe`, `vq index`, or `vq search` command acquire only its own model.
- `--no-model-fetch`: required uncached Whisper or Chinese-CLIP stages fail before inference. The optional SenseVoice cross-check is skipped when uncached.
- yt-dlp absent: with `--install-missing`, prefer the project's pinned `uv` environment, then Homebrew on macOS.
- FFmpeg absent: with `--install-missing` on macOS, install through Homebrew; otherwise report the narrow prerequisite.
- No subtitles: install/use `whisper-cli` and create timestamped local ASR. On Windows/Linux, `--install-missing` downloads a checksum-pinned official whisper.cpp runtime; on macOS it uses Homebrew.
- Private, DRM-protected, login-gated, or unsupported URLs: do not bypass protections. Ask the user for an authorized local file or cookies/configuration they control. After explicit authorization, pass the selected store through `prepare_video.py --cookies-from-browser <browser>`; never read browser cookies by default.
- Very long videos: first use `--metadata-only` to inspect duration and transcript, then run the complete command when the user confirms the compute/storage cost.

## Final quality bar

The final report must let a reader distinguish four evidence types: publisher metadata, subtitle/ASR statements, embedding-based discovery, and direct frame observations. Claims from the speaker are not automatically facts. Include timestamps for important language claims and image timestamps for visual claims.
