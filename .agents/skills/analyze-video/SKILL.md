---
name: analyze-video
description: Analyze a local video or downloadable video URL end to end. Use when Codex needs to acquire video metadata and subtitles with yt-dlp, fall back to local audio-to-text, download or prepare the cross-platform Video Sherlock (vq) CLI and only the models required by active stages, index frames for Chinese/English semantic search, infer important moments from timed speech, inspect keyframes visually, preserve raw evidence, and deliver one comprehensive Markdown report.
---

# Analyze Video

Produce an evidence-backed video report from publisher metadata, timed language evidence, semantic frame retrieval, and direct inspection of selected frames. Keep every intermediate artifact so another agent can audit or resume the work.

## Required workflow

Read [references/workflow.md](references/workflow.md) before running the pipeline. Follow the stages in order; transcript-first reasoning is mandatory.

### 1. Establish source and output

Accept either an HTTP(S) video URL or an absolute/local media path. Pick a dedicated output directory that does not contain unrelated user files. Only download media the user is authorized to save, and honor the source site's terms.

Briefly tell the user before the first run that missing packages or stage-specific models may be built/downloaded. Do not quote one combined download size: subtitles, metadata-only mode, ASR fallback, and indexing need different resources.

### 2. Prepare all machine-readable evidence

Resolve this skill's directory, then run:

```sh
python3 <skill-dir>/scripts/prepare_video.py \
  "<URL-or-video>" \
  --output-dir "<analysis-dir>" \
  --install-missing
```

Add `--language zh` for known Mandarin/Cantonese content. Omit `--install-missing` only when the user prohibits package-manager changes. Do not replace this script with ad hoc commands: it accepts an already healthy `vq`, otherwise downloads a checksum-verified GitHub release for Windows, Linux, or macOS, and uses a GitHub source build only as the final fallback. It verifies prerequisites only when their stage needs them, downloads useful sidecars, normalizes subtitles or transcribes locally, indexes the video, ranks timed cues, runs embedding searches, and extracts candidate frames.

Model downloads are lazy and stage-scoped:

- authored or automatic subtitles avoid both ASR model downloads;
- Whisper downloads only when timed ASR fallback is required;
- SenseVoice downloads only when that ASR cross-check is enabled;
- Chinese-CLIP downloads only when visual indexing runs;

Never call broad `vq model fetch` as part of the normal skill workflow. Use `--no-model-fetch` when the user prohibits model downloads; required uncached stages then fail narrowly, while an uncached optional SenseVoice cross-check is skipped.

If a site requires browser cookies and the user explicitly authorizes access to a browser cookie store, add `--cookies-from-browser <browser>` (for example, `--cookies-from-browser chrome`). Never enable browser-cookie access implicitly.

If preparation fails, inspect `raw/logs/` and fix the narrow failing stage. Re-run the same command; generated stages are designed to be safely refreshed. Never delete the analysis directory to recover unless the user explicitly asks.

### 3. Understand language evidence first

Read these in order:

1. `raw/metadata.json` and `raw/description.txt` when present.
2. `raw/transcript.md` completely.
3. `raw/transcript-sensevoice.json` when present. Use it as a Chinese-first wording cross-check; keep Whisper/subtitle segments as the timing source.
4. `raw/candidates.json` to understand why moments were proposed.

Write `raw/analysis.json` using the exact schema in the workflow reference. Summarize the video's actual argument or events, identify key points and topics, and make a time-bounded topic timeline. Cite transcript timestamps in evidence strings. Explicitly record uncertainty, subtitle/ASR errors, sponsorship, missing context, or unverifiable claims under `limitations`.

### 4. Inspect selected frames with Codex vision

Read `raw/keyframe-observations.json`. For every listed frame:

1. Open the local image with Codex's image-viewing ability; use original detail when small text, charts, code, or UI is present.
2. Describe only what is visibly supported.
3. Transcribe important visible text conservatively.
4. Explain whether the frame supports, contradicts, or merely illustrates its subtitle/search cue.
5. Record confidence as `high`, `medium`, or `low`.

Update the same JSON without dropping its selection metadata. Compare subtitle-timed and embedding-selected results when they target the same claim. If all proposed frames miss an important visual, use the manual recovery commands in the workflow reference, append the better frame to the observations, and inspect it.

### 5. Flatten and verify

Build the report only after language synthesis and visual inspection are complete:

```sh
python3 <skill-dir>/scripts/assemble_report.py "<analysis-dir>" --strict
```

Open `report.md` and verify that:

- no pending placeholders remain;
- the summary is supported by transcript or visual evidence;
- timestamps and image links resolve;
- key visuals are explained, not merely listed;
- raw artifacts remain under `raw/` and `source/`;
- limitations distinguish publisher claims, ASR/subtitle text, and direct visual observations.

Return the absolute paths to `report.md` and the analysis directory. In the response, state which transcript source was used and whether any stage was unavailable.

If the user also requests spoken narration, complete and verify the report first, then follow the separate `synthesize-speech` skill using a clean narration script derived from the finished report.

## Non-negotiable evidence rules

- Read subtitles/transcript before interpreting frames.
- Never infer motion, causality, or off-screen events from a still image alone.
- Treat embedding similarity as a retrieval hint, not proof.
- Preserve contradictory evidence and low-confidence OCR.
- Do not call a draft complete if strict report assembly fails.
