#!/usr/bin/env python3
"""Flatten prepared video evidence and Codex observations into Markdown."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final Markdown video-analysis report"
    )
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require synthesis and visual observations",
    )
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def format_time(seconds: float | None) -> str:
    millis = max(0, round(float(seconds or 0) * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"
    return f"{minutes:02}:{secs:02}.{millis:03}"


def table_text(value: Any) -> str:
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


def markdown_path(path: str, report_dir: Path, analysis_dir: Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        value = analysis_dir / value
    try:
        return os.fspath(value.resolve().relative_to(report_dir.resolve()))
    except ValueError:
        return os.fspath(value.resolve())


def fail_if_incomplete(analysis: dict[str, Any], observations: dict[str, Any]) -> None:
    missing: list[str] = []
    if not str(analysis.get("summary") or "").strip():
        missing.append("raw/analysis.json: summary")
    if not analysis.get("key_points"):
        missing.append("raw/analysis.json: key_points")
    frames = observations.get("frames") or []
    if frames and any(
        not str(frame.get("observation") or "").strip() for frame in frames
    ):
        missing.append(
            "raw/keyframe-observations.json: observation for every selected frame"
        )
    if missing:
        raise ValueError("strict report validation failed: " + "; ".join(missing))


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    raw = analysis_dir / "raw"
    output = (args.output or (analysis_dir / "report.md")).expanduser().resolve()
    manifest = load_json(analysis_dir / "manifest.json", {})
    metadata = load_json(raw / "metadata.json", {})
    transcript = load_json(raw / "transcript.json", {})
    analysis = load_json(raw / "analysis.json", {})
    observations = load_json(raw / "keyframe-observations.json", {"frames": []})
    if args.strict:
        fail_if_incomplete(analysis, observations)

    title = metadata.get("title") or Path(str(manifest.get("video") or "Video")).stem
    lines = [f"# {title}", "", "## Source", ""]
    lines.extend(
        [
            "| Field | Value |",
            "|---|---|",
            f"| Source | {table_text(metadata.get('source_url') or manifest.get('requested_source'))} |",
            f"| Creator | {table_text(metadata.get('uploader'))} |",
            f"| Upload date | {table_text(metadata.get('upload_date'))} |",
            f"| Duration | {format_time(metadata.get('duration_seconds'))} |",
            f"| Transcript | {table_text(manifest.get('transcript_engine'))} ({table_text(manifest.get('language'))}) |",
            f"| Indexed frames | {table_text((manifest.get('index_stats') or {}).get('frames'))} |",
            "",
            "## Summary",
            "",
            str(
                analysis.get("summary")
                or "_[Pending Codex synthesis from the transcript and frames.]_"
            ),
            "",
        ]
    )

    key_points = analysis.get("key_points") or []
    lines.extend(["## Key points", ""])
    if key_points:
        lines.extend(f"- {point}" for point in key_points)
    else:
        lines.append("- _Pending Codex synthesis._")
    lines.append("")

    topics = analysis.get("topics") or []
    if topics:
        lines.extend(["## Topics", ""])
        lines.extend(f"- {topic}" for topic in topics)
        lines.append("")

    timeline = analysis.get("timeline") or []
    lines.extend(["## Timeline", ""])
    if timeline:
        lines.extend(["| Time | Topic | Evidence |", "|---|---|---|"])
        for item in timeline:
            start = format_time(item.get("start_seconds"))
            end = (
                format_time(item.get("end_seconds"))
                if item.get("end_seconds") is not None
                else ""
            )
            label = f"{start}–{end}" if end else start
            lines.append(
                f"| {label} | {table_text(item.get('topic'))} | {table_text(item.get('evidence'))} |"
            )
    else:
        lines.append("_No synthesized topic timeline was recorded._")
    lines.append("")

    lines.extend(["## Visual evidence", ""])
    frames = observations.get("frames") or []
    if frames:
        for frame in frames:
            timestamp = format_time(frame.get("timestamp_seconds"))
            source = frame.get("source") or "selected frame"
            lines.extend(
                [
                    f"### {timestamp} — {source}",
                    "",
                    f"![Video frame at {timestamp}]({markdown_path(frame['path'], output.parent, analysis_dir)})",
                    "",
                    str(frame.get("observation") or "_[Pending visual inspection.]_"),
                ]
            )
            if frame.get("visible_text"):
                lines.extend(["", f"Visible text: {frame['visible_text']}"])
            if frame.get("relevance"):
                lines.extend(["", f"Why it matters: {frame['relevance']}"])
            if frame.get("query"):
                lines.extend(["", f"Selection cue: “{frame['query']}”"])
            lines.append("")
    else:
        lines.extend(["_No visual frames were selected._", ""])

    description = str(metadata.get("description") or "").strip()
    if description:
        lines.extend(["## Published description", "", description, ""])

    lines.extend(["## Transcript", ""])
    segments = transcript.get("segments") or []
    if segments:
        for segment in segments:
            lines.append(
                f"- **{format_time(segment.get('start_seconds'))}–{format_time(segment.get('end_seconds'))}:** "
                f"{str(segment.get('text') or '').strip()}"
            )
    else:
        lines.append(str(transcript.get("text") or "_No transcript available._"))
    lines.append("")

    limitations = analysis.get("limitations") or []
    lines.extend(["## Method and limitations", ""])
    lines.append(
        "The report combines publisher metadata, timed subtitles or local speech recognition, "
        "subtitle-guided frame extraction, and Chinese-CLIP semantic frame retrieval. Visual "
        "claims come from the selected frames; transcript claims remain subject to subtitle/ASR error."
    )
    if limitations:
        lines.extend(["", *[f"- {item}" for item in limitations]])
    lines.extend(
        [
            "",
            "## Raw artifacts",
            "",
            "- [Run manifest](manifest.json)",
            "- [Metadata](raw/metadata.json)",
            "- [Transcript JSON](raw/transcript.json)",
            "- [Transcript Markdown](raw/transcript.md)",
            "- [Candidate timestamps](raw/candidates.json)",
            "- [Selected keyframes](raw/keyframe-selection.json)",
            "- [Visual observations](raw/keyframe-observations.json)",
            "- [Analysis synthesis](raw/analysis.json)",
            "- [Execution logs](raw/logs/)",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
