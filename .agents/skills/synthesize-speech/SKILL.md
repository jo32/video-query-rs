---
name: synthesize-speech
description: Generate local speech and WAV narration from text or UTF-8 text files with Video Sherlock's vq speak command and Qwen3-TTS on Apple silicon. Use when Codex is asked to speak text, create a voiceover or narration, play synthesized speech, adjust voice/language/speed, inspect or prefetch TTS assets, or clone a voice from authorized reference audio.
---

# Synthesize Speech

Create local 24 kHz WAV speech with `vq speak`. Keep TTS independent from video analysis: accept standalone text, or consume a finished script/report from another workflow without changing its evidence or conclusions.

## 1. Resolve `vq` and confirm support

Resolve the executable in this order:

1. Use an exact `vq` path supplied by the user or recorded under `tools.vq.path` in an existing analysis `manifest.json`.
2. Use `vq` from `PATH`.
3. In a Video Sherlock checkout, use a current `target/release/vq` or `target/debug/vq` only after confirming it contains the `speak` command.

Run `"<vq>" --json doctor` when platform or runtime readiness is uncertain. Qwen3-TTS through MLX-Audio requires an Apple-silicon Mac and `uv`; other `vq` features remain cross-platform. If `vq` is unavailable, install a checksum-verified Video Sherlock release or build the current checkout with Cargo before continuing.

## 2. Prepare the speech text

Preserve user-provided wording unless the user asks for editing. For a report, article, or Markdown document, create a clean UTF-8 narration file containing only speakable prose: remove URLs, image paths, table syntax, citation markup, and other material that should not be read aloud. Do not synthesize secrets or private content into a shared location.

Use positional text for a short passage and `--text-file` for longer content:

```sh
"<vq>" speak "你好，这是本地生成的语音。" \
  --output "/absolute/path/speech.wav"

"<vq>" speak \
  --text-file "/absolute/path/narration.txt" \
  --output "/absolute/path/narration.wav"
```

Always use a `.wav` output path. Add `--voice`, `--language`, or `--speed` only when requested or needed for the supplied text. Add `--play` only when the user explicitly asks to hear the result immediately.

## 3. Keep model acquisition lazy

Let the first `vq speak` call create the pinned MLX-Audio environment and download `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit`; later calls reuse both caches. Allow roughly 1.9 GB of model cache and about 5 GB of available unified memory.

Do not run broad `vq model fetch`. Use these component commands only when the user asks to inspect readiness or prefetch TTS without generating speech:

```sh
"<vq>" model status-tts
"<vq>" model fetch-tts
```

If the user prohibits downloads, run `"<vq>" --json model status-tts` first and stop if `ready` is false. Do not initiate `vq speak` against a missing cache in that case.

## 4. Clone voices only with permission

Require confirmation that the user has the reference speaker's authorization. Supply the reference audio and its exact transcript together:

```sh
"<vq>" speak "要生成的句子" \
  --reference-audio "/absolute/path/reference.wav" \
  --reference-text "参考音频中准确说出的文字" \
  --output "/absolute/path/cloned.wav"
```

Never infer consent from possession of an audio file. If authorization is unclear, use a built-in voice instead.

## 5. Verify and return the result

Confirm that the output exists and is non-empty. When media inspection tools are available, verify that it is a readable WAV before declaring success. Return the absolute output path and state the model, voice, language, and whether playback occurred. For narration derived from a video report, keep the narration beside the report when practical and do not alter the report's source evidence.
