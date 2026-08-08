#!/usr/bin/env python3
"""Prepare a URL or local video for evidence-backed Codex analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

MEDIA_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".mts",
    ".m2ts",
}
SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
VISUAL_MARKERS: dict[str, tuple[str, ...]] = {
    "chart or quantitative visual": (
        "chart",
        "graph",
        "plot",
        "curve",
        "axis",
        "diagram",
        "图表",
        "图形",
        "曲线",
        "坐标",
        "走势图",
        "柱状图",
        "饼图",
    ),
    "table or structured data": (
        "table",
        "spreadsheet",
        "matrix",
        "表格",
        "数据表",
        "矩阵",
    ),
    "screen, slide, or demonstration": (
        "on screen",
        "this slide",
        "as you can see",
        "look at",
        "shown here",
        "demo",
        "demonstrate",
        "screen",
        "slide",
        "如图",
        "看这里",
        "屏幕",
        "画面",
        "幻灯片",
        "演示",
        "展示",
        "我们看到",
        "可以看到",
    ),
    "code, command, or interface": (
        "code",
        "terminal",
        "command",
        "dashboard",
        "interface",
        "button",
        "menu",
        "代码",
        "终端",
        "命令",
        "仪表盘",
        "界面",
        "按钮",
        "菜单",
    ),
    "physical object or scene": (
        "camera",
        "device",
        "product",
        "machine",
        "map",
        "地图",
        "设备",
        "产品",
        "机器",
        "镜头",
    ),
}
TRANSITION_MARKERS = (
    "first",
    "second",
    "next",
    "finally",
    "in summary",
    "let's look",
    "now",
    "首先",
    "其次",
    "接下来",
    "然后",
    "最后",
    "总结",
    "现在",
)


class PreparationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/inspect a video, build a local vq index, and select evidence frames."
    )
    parser.add_argument("source", help="A video URL or local media file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--vq", type=Path, help="Explicit vq executable")
    parser.add_argument(
        "--vq-repo",
        default=os.environ.get("VQ_GITHUB_REPO", "jo32/video-sherlock"),
        help="GitHub owner/repository used for vq release downloads",
    )
    parser.add_argument(
        "--vq-version",
        default=os.environ.get("VQ_VERSION", "latest"),
        help="vq release tag, or latest",
    )
    parser.add_argument(
        "--no-vq-download",
        action="store_true",
        help="Skip prebuilt vq release downloads and build from source",
    )
    parser.add_argument(
        "--language", default="auto", help="Speech/subtitle language hint"
    )
    parser.add_argument(
        "--subtitle-languages",
        default="zh.*,yue.*,en.*,ja.*,ko.*",
        help="yt-dlp subtitle language expression",
    )
    parser.add_argument(
        "--cookies-from-browser",
        help=(
            "Explicitly authorized browser cookie store for yt-dlp "
            "(for example: chrome)"
        ),
    )
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--embedding-queries", type=int, default=4)
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Install or download missing supported prerequisites when possible",
    )
    parser.add_argument(
        "--no-model-fetch",
        action="store_true",
        help="Use cached models only; fail when a required stage-specific model is missing",
    )
    parser.add_argument(
        "--no-sensevoice-crosscheck",
        action="store_true",
        help="Skip the Chinese-first full-text transcript when ASR is required",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Acquire metadata/media and transcript, but skip index and keyframes",
    )
    args = parser.parse_args()
    if args.segment_seconds <= 0:
        parser.error("--segment-seconds must be greater than zero")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be greater than zero")
    if args.embedding_queries < 0:
        parser.error("--embedding-queries cannot be negative")
    return args


def run(
    command: Sequence[str | os.PathLike[str]],
    *,
    log_path: Path | None = None,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [os.fspath(part) for part in command]
    result = subprocess.run(
        rendered,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "$ "
            + " ".join(rendered)
            + "\n\nSTDOUT\n"
            + result.stdout
            + "\nSTDERR\n"
            + result.stderr,
            encoding="utf-8",
        )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise PreparationError(
            f"command failed ({result.returncode}): {' '.join(rendered)}\n{detail}"
        )
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value, flags=re.IGNORECASE))


def find_project_root(explicit: Path | None) -> Path:
    if explicit:
        root = explicit.expanduser().resolve()
        if not (root / "Cargo.toml").is_file():
            raise PreparationError(f"Cargo.toml not found under --project-root {root}")
        return root
    for candidate in Path(__file__).resolve().parents:
        cargo = candidate / "Cargo.toml"
        if cargo.is_file() and "video-sherlock" in cargo.read_text(encoding="utf-8"):
            return candidate
    raise PreparationError(
        "could not locate the Video Sherlock project; pass --project-root"
    )


def command_version(command: Sequence[str]) -> str:
    result = run(command, check=False)
    return (
        (result.stdout or result.stderr).strip().splitlines()[0]
        if (result.stdout or result.stderr).strip()
        else "unknown"
    )


def install_brew_formula(formula: str) -> None:
    brew = shutil.which("brew")
    if not brew:
        raise PreparationError(f"{formula} is missing and Homebrew is unavailable")
    run([brew, "install", formula])


def vq_works(candidate: Path) -> bool:
    if not candidate.is_file():
        return False
    probe = run([candidate, "--version"], check=False)
    return probe.returncode == 0 and "vq " in (probe.stdout + probe.stderr)


def ensure_vq(project_root: Path, args: argparse.Namespace, logs: Path) -> Path:
    candidates: list[Path] = []
    if args.vq:
        candidates.append(args.vq.expanduser())
    if os.environ.get("VQ_BIN"):
        candidates.append(Path(os.environ["VQ_BIN"]).expanduser())
    on_path = shutil.which("vq")
    if on_path:
        candidates.append(Path(on_path))
    local_name = "vq.exe" if os.name == "nt" else "vq"
    candidates.append(project_root / "target" / "release" / local_name)
    for candidate in candidates:
        if vq_works(candidate):
            return candidate.resolve()
    bootstrap = Path(__file__).with_name("bootstrap_vq.py")
    command = [
        sys.executable,
        bootstrap,
        "--repo",
        args.vq_repo,
        "--version",
        args.vq_version,
        "--project-root",
        project_root,
    ]
    if args.no_vq_download:
        command.append("--no-download")
    result = run(command, log_path=logs / "vq-bootstrap.log")
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise PreparationError("vq bootstrap did not return an executable path")
    binary = Path(output_lines[-1])
    if not vq_works(binary):
        raise PreparationError(f"vq bootstrap returned an unusable binary: {binary}")
    return binary.resolve()


def ensure_yt_dlp(project_root: Path, install_missing: bool) -> list[str]:
    direct = shutil.which("yt-dlp")
    if direct:
        return [direct]
    module_probe = run([sys.executable, "-m", "yt_dlp", "--version"], check=False)
    if module_probe.returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]
    uv = shutil.which("uv")
    if uv and (project_root / "pyproject.toml").is_file():
        probe = run(
            [uv, "run", "--no-sync", "--project", project_root, "yt-dlp", "--version"],
            check=False,
        )
        if probe.returncode == 0:
            return [uv, "run", "--project", os.fspath(project_root), "yt-dlp"]
        if install_missing:
            run([uv, "sync", "--project", project_root])
            return [uv, "run", "--project", os.fspath(project_root), "yt-dlp"]
    if install_missing and sys.platform == "darwin":
        install_brew_formula("yt-dlp")
        direct = shutil.which("yt-dlp")
        if direct:
            return [direct]
    if install_missing:
        run([sys.executable, "-m", "pip", "install", "--user", "yt-dlp[default]"])
        module_probe = run([sys.executable, "-m", "yt_dlp", "--version"], check=False)
        if module_probe.returncode == 0:
            return [sys.executable, "-m", "yt_dlp"]
    raise PreparationError(
        "yt-dlp is missing; install it or rerun with --install-missing"
    )


def ensure_ffmpeg(install_missing: bool) -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if install_missing and sys.platform == "darwin":
        install_brew_formula("ffmpeg")
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise PreparationError("ffmpeg and ffprobe are required on PATH")


def parse_json_stdout(result: subprocess.CompletedProcess[str], label: str) -> Any:
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PreparationError(f"{label} returned invalid JSON: {error}") from error


MODEL_STATUS_COMMANDS: dict[str, tuple[str, ...]] = {
    "embedding": ("model", "status"),
    "whisper": ("model", "status-whisper"),
    "sensevoice": ("model", "status-speech"),
}


def component_model_status(
    vq: Path,
    component: str,
    logs: Path,
    phase: str,
) -> dict[str, Any]:
    try:
        status_command = MODEL_STATUS_COMMANDS[component]
    except KeyError as error:
        raise PreparationError(f"unknown model component: {component}") from error
    result = run(
        [vq, "--json", *status_command],
        log_path=logs / f"model-{component}-status-{phase}.log",
    )
    status = parse_json_stdout(result, f"vq {' '.join(status_command)}")
    if component == "embedding":
        status = status.get("embedding") or {}
    if not isinstance(status, dict):
        raise PreparationError(f"vq returned invalid {component} model status")
    return status


def prepare_model_for_use(
    vq: Path,
    component: str,
    logs: Path,
    no_model_fetch: bool,
    models: dict[str, dict[str, Any]],
    *,
    required: bool,
) -> bool:
    status = component_model_status(vq, component, logs, "before")
    ready = bool(status.get("ready"))
    models[component] = {
        "model": status.get("model"),
        "required": required,
        "ready_before": ready,
        "ready_after": ready,
        "downloaded_on_use": False,
    }
    if ready:
        return True
    if no_model_fetch:
        if required:
            raise PreparationError(
                f"{component} model is required but not cached and --no-model-fetch was set"
            )
        models[component]["skipped"] = "not cached under --no-model-fetch"
        print(
            f"[prepare] model={component} status=missing action=skip-optional",
            file=sys.stderr,
        )
        return False
    print(
        f"[prepare] model={component} status=missing action=download-on-use",
        file=sys.stderr,
    )
    return True


def finish_model_use(
    vq: Path,
    component: str,
    logs: Path,
    models: dict[str, dict[str, Any]],
    *,
    required: bool,
) -> None:
    status = component_model_status(vq, component, logs, "after")
    ready = bool(status.get("ready"))
    record = models.setdefault(component, {})
    record["model"] = status.get("model") or record.get("model")
    record["ready_after"] = ready
    record["downloaded_on_use"] = not bool(record.get("ready_before")) and ready
    if required and not ready:
        raise PreparationError(
            f"{component} command completed but its model cache is still incomplete"
        )


def acquire_url(
    source: str,
    source_dir: Path,
    yt_dlp: Sequence[str],
    subtitle_languages: str,
    cookies_from_browser: str | None,
    logs: Path,
) -> Path:
    template = source_dir / "%(id)s.%(ext)s"
    command = [
        *yt_dlp,
        "--no-playlist",
        "--no-write-playlist-metafiles",
        "--write-info-json",
        "--clean-info-json",
        "--write-description",
        "--write-thumbnail",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        subtitle_languages,
        "--sub-format",
        "vtt/srt/best",
        "--convert-subs",
        "srt",
        "--format",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--output",
        os.fspath(template),
        "--print",
        "after_move:filepath",
    ]
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.append(source)
    result = run(command, log_path=logs / "yt-dlp.log")
    printed_paths = [
        Path(line.strip()) for line in result.stdout.splitlines() if line.strip()
    ]
    for candidate in reversed(printed_paths):
        if candidate.is_file() and candidate.suffix.lower() in MEDIA_EXTENSIONS:
            return candidate.resolve()
    media = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.suffix.lower() in MEDIA_EXTENSIONS
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not media:
        raise PreparationError("yt-dlp finished but no supported video file was found")
    return media[0].resolve()


def probe_video(video: Path, logs: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            video,
        ],
        log_path=logs / "ffprobe.log",
    )
    return parse_json_stdout(result, "ffprobe")


def parse_timestamp(value: str) -> float:
    normalized = value.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        raise ValueError(f"invalid subtitle timestamp: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_caption(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    text = re.sub(r"\{\\[^}]+}", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_subtitle(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8-sig", errors="replace").replace(
        "\r\n", "\n"
    )
    pattern = re.compile(
        r"(?m)^(?P<start>\d{1,2}:\d{2}(?::\d{2})?[,.]\d{3})\s+-->\s+"
        r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[,.]\d{3})[^\n]*\n"
        r"(?P<text>.*?)(?=\n\s*\n|\Z)",
        flags=re.DOTALL,
    )
    segments: list[dict[str, Any]] = []
    for match in pattern.finditer(content):
        text = clean_caption(match.group("text").replace("\n", " "))
        if not text:
            continue
        segment = {
            "start_seconds": round(parse_timestamp(match.group("start")), 3),
            "end_seconds": round(parse_timestamp(match.group("end")), 3),
            "text": text,
        }
        if segments and segments[-1]["text"] == text:
            segments[-1]["end_seconds"] = segment["end_seconds"]
        else:
            segments.append(segment)
    if not segments:
        raise PreparationError(f"no timed cues could be parsed from {path}")
    return segments


def subtitle_score(path: Path, language: str) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if path.suffix.lower() == ".srt":
        score += 3
    if language != "auto" and re.search(
        rf"[._-]{re.escape(language.lower())}(?:[._-]|$)", name
    ):
        score += 8
    if language == "auto" and any(
        token in name for token in (".zh", "-zh", ".yue", "-yue")
    ):
        score += 5
    if any(token in name for token in (".en", "-en")):
        score += 2
    if "live_chat" in name:
        score -= 100
    return (-score, len(name), name)


def choose_subtitle(
    source_dir: Path, video: Path, language: str
) -> tuple[Path | None, list[Path]]:
    roots = {source_dir.resolve(), video.parent.resolve()}
    video_stem = video.stem.casefold()
    subtitles = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUBTITLE_EXTENSIONS
            and path.name.casefold().startswith(video_stem)
        },
        key=lambda path: subtitle_score(path, language),
    )
    return (subtitles[0] if subtitles else None), subtitles


def infer_subtitle_language(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    match = re.search(
        r"[._-](zh(?:-[a-z]+)?|yue|en|ja|ko)(?:[._-]|$)", path.name, re.IGNORECASE
    )
    return match.group(1) if match else "unknown"


def transcript_from_subtitle(path: Path, video: Path, language: str) -> dict[str, Any]:
    segments = parse_subtitle(path)
    return {
        "engine": "subtitle",
        "input": os.fspath(video),
        "model": None,
        "language": infer_subtitle_language(path, language),
        "text": " ".join(segment["text"] for segment in segments),
        "segments": segments,
        "emotions": [],
        "audio_events": [],
        "subtitle_file": os.fspath(path),
    }


def ensure_whisper_runtime(install_missing: bool) -> None:
    if shutil.which("whisper-cli"):
        return
    if install_missing and sys.platform == "darwin":
        install_brew_formula("whisper-cpp")
    elif install_missing:
        bootstrap = Path(__file__).with_name("bootstrap_whisper.py")
        result = run([sys.executable, bootstrap])
        output_lines = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        if output_lines:
            cli = Path(output_lines[-1]).resolve()
            os.environ["PATH"] = (
                os.fspath(cli.parent) + os.pathsep + os.environ.get("PATH", "")
            )
            if os.name != "nt":
                os.environ["LD_LIBRARY_PATH"] = (
                    os.fspath(cli.parent)
                    + os.pathsep
                    + os.environ.get("LD_LIBRARY_PATH", "")
                )
    if not shutil.which("whisper-cli"):
        raise PreparationError(
            "no usable subtitles were found and whisper-cli is missing; install whisper-cpp "
            "or rerun with --install-missing"
        )


def transcribe_video(
    vq: Path,
    video: Path,
    language: str,
    raw_dir: Path,
    logs: Path,
    install_missing: bool,
    sensevoice_crosscheck: bool,
    no_model_fetch: bool,
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ensure_whisper_runtime(install_missing)
    prepare_model_for_use(
        vq,
        "whisper",
        logs,
        no_model_fetch,
        models,
        required=True,
    )
    transcript_path = raw_dir / "transcript.json"
    whisper = run(
        [
            vq,
            "--json",
            "transcribe",
            video,
            "--engine",
            "whisper",
            "--language",
            language,
            "--timestamps",
            "--output",
            transcript_path,
        ],
        log_path=logs / "transcribe-whisper.log",
    )
    transcript = parse_json_stdout(whisper, "vq transcribe --engine whisper")
    finish_model_use(vq, "whisper", logs, models, required=True)
    write_json(transcript_path, transcript)
    if sensevoice_crosscheck and prepare_model_for_use(
        vq,
        "sensevoice",
        logs,
        no_model_fetch,
        models,
        required=False,
    ):
        sense_path = raw_dir / "transcript-sensevoice.json"
        sense = run(
            [
                vq,
                "--json",
                "transcribe",
                video,
                "--engine",
                "sensevoice",
                "--language",
                language,
                "--output",
                sense_path,
            ],
            log_path=logs / "transcribe-sensevoice.log",
            check=False,
        )
        finish_model_use(vq, "sensevoice", logs, models, required=False)
        if sense.returncode == 0:
            try:
                write_json(sense_path, json.loads(sense.stdout))
            except json.JSONDecodeError:
                pass
    return transcript


def format_time(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"
    return f"{minutes:02}:{secs:02}.{millis:03}"


def write_transcript_markdown(path: Path, transcript: dict[str, Any]) -> None:
    lines = [
        "# Transcript",
        "",
        f"- Engine/source: `{transcript.get('engine', 'unknown')}`",
        f"- Language: `{transcript.get('language', 'unknown')}`",
        "",
    ]
    segments = transcript.get("segments") or []
    if segments:
        for segment in segments:
            lines.append(
                f"- **{format_time(float(segment['start_seconds']))}–"
                f"{format_time(float(segment['end_seconds']))}:** {segment['text']}"
            )
    else:
        lines.extend([transcript.get("text", ""), ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def duration_from_probe(probe: dict[str, Any]) -> float:
    try:
        return float(probe.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        return 0.0


def has_audio_stream(probe: dict[str, Any]) -> bool:
    return any(
        stream.get("codec_type") == "audio" for stream in probe.get("streams") or []
    )


def empty_transcript(video: Path) -> dict[str, Any]:
    return {
        "engine": "none",
        "input": os.fspath(video),
        "model": None,
        "language": "none",
        "text": "",
        "segments": [],
        "emotions": [],
        "audio_events": [],
        "reason": "No subtitles or audio stream were available.",
    }


def cue_score(text: str) -> tuple[float, list[str]]:
    lowered = text.casefold()
    score = 0.0
    reasons: list[str] = []
    for reason, markers in VISUAL_MARKERS.items():
        hits = [marker for marker in markers if marker.casefold() in lowered]
        if hits:
            score += 2.2 + min(1.2, 0.3 * (len(hits) - 1))
            reasons.append(reason)
    if any(marker in lowered for marker in TRANSITION_MARKERS):
        score += 0.7
        reasons.append("topic transition")
    if re.search(r"(?:\d+(?:\.\d+)?\s*%|[$€¥£]\s*\d|\d{3,})", text):
        score += 0.8
        reasons.append("quantitative claim")
    return score, reasons


def evenly_spaced(
    items: Sequence[dict[str, Any]], count: int
) -> Iterable[dict[str, Any]]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [
        items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)
    ]


def build_candidates(
    transcript: dict[str, Any],
    metadata: dict[str, Any],
    duration: float,
    limit: int,
) -> list[dict[str, Any]]:
    latest_frame_time = max(0.0, duration - 0.5)

    def safe_frame_time(value: float) -> float:
        if duration <= 0:
            return max(0.0, value)
        return min(max(0.0, value), latest_frame_time)

    segments = transcript.get("segments") or []
    proposed: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        score, reasons = cue_score(text)
        if score:
            start = float(segment.get("start_seconds", 0))
            end = float(segment.get("end_seconds", start))
            proposed.append(
                {
                    "timestamp_seconds": round(
                        safe_frame_time((start + end) / 2), 3
                    ),
                    "cue_start_seconds": start,
                    "cue_end_seconds": end,
                    "query": text[:240],
                    "score": score,
                    "reasons": reasons,
                    "origin": "transcript",
                }
            )
    for chapter in metadata.get("chapters") or []:
        try:
            timestamp = float(chapter.get("start_time", 0)) + 0.8
        except (TypeError, ValueError):
            continue
        title = str(chapter.get("title") or "chapter transition").strip()
        proposed.append(
            {
                "timestamp_seconds": round(safe_frame_time(timestamp), 3),
                "cue_start_seconds": timestamp,
                "cue_end_seconds": float(chapter.get("end_time") or timestamp),
                "query": title,
                "score": 1.4,
                "reasons": ["chapter transition"],
                "origin": "chapter",
            }
        )
    fallback_count = min(6, limit)
    for segment in evenly_spaced(segments, fallback_count):
        start = float(segment.get("start_seconds", 0))
        end = float(segment.get("end_seconds", start))
        proposed.append(
            {
                "timestamp_seconds": round(
                    safe_frame_time((start + end) / 2), 3
                ),
                "cue_start_seconds": start,
                "cue_end_seconds": end,
                "query": str(segment.get("text", ""))[:240],
                "score": 0.25,
                "reasons": ["timeline coverage"],
                "origin": "coverage",
            }
        )
    if not segments and duration > 0:
        for fraction in (0.08, 0.3, 0.5, 0.7, 0.92):
            proposed.append(
                {
                    "timestamp_seconds": round(duration * fraction, 3),
                    "cue_start_seconds": None,
                    "cue_end_seconds": None,
                    "query": str(metadata.get("title") or "representative video frame")[
                        :240
                    ],
                    "score": 0.2,
                    "reasons": ["timeline coverage without timed transcript"],
                    "origin": "coverage",
                }
            )
    proposed.sort(key=lambda item: (-item["score"], item["timestamp_seconds"]))
    selected: list[dict[str, Any]] = []
    for item in proposed:
        if any(
            abs(item["timestamp_seconds"] - other["timestamp_seconds"]) < 5.0
            for other in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: item["timestamp_seconds"])
    for number, item in enumerate(selected, 1):
        item["id"] = f"candidate-{number:02d}"
    return selected


def relative(path: Path, root: Path) -> str:
    try:
        return os.fspath(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return os.fspath(path.resolve())


def extract_timed_frames(
    vq: Path,
    video: Path,
    candidates: list[dict[str, Any]],
    raw_dir: Path,
    logs: Path,
    root: Path,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for candidate in candidates:
        frame_dir = raw_dir / "keyframes" / "timed" / candidate["id"]
        frame_dir.mkdir(parents=True, exist_ok=True)
        result = run(
            [
                vq,
                "--json",
                "keyframes",
                video,
                "--at",
                str(candidate["timestamp_seconds"]),
                "--radius",
                "3",
                "--output-dir",
                frame_dir,
            ],
            log_path=logs / f"keyframe-{candidate['id']}.log",
        )
        values = parse_json_stdout(result, "vq keyframes")
        if not values:
            continue
        keyframe = values[0]
        frame_millis = round(float(keyframe.get("timestamp_seconds", 0)) * 1000)
        image = frame_dir / f"{max(0, frame_millis):012d}.jpg"
        if not image.is_file():
            images = sorted(
                frame_dir.glob("*.jpg"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not images:
                continue
            image = images[0]
        frames.append(
            {
                "id": candidate["id"],
                "source": "subtitle-timing",
                "timestamp_seconds": keyframe.get(
                    "timestamp_seconds", candidate["timestamp_seconds"]
                ),
                "path": relative(image, root),
                "query": candidate["query"],
                "reasons": candidate["reasons"],
                "quality": keyframe.get("quality"),
            }
        )
    return frames


def semantic_search(
    vq: Path,
    index_dir: Path,
    candidates: list[dict[str, Any]],
    query_limit: int,
    raw_dir: Path,
    logs: Path,
    root: Path,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    search_dir = raw_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        candidates, key=lambda item: (-item["score"], item["timestamp_seconds"])
    )
    for candidate in ranked[:query_limit]:
        query = candidate["query"].strip()
        if not query:
            continue
        result = run(
            [vq, "--json", "--index-dir", index_dir, "search", query, "--limit", "3"],
            log_path=logs / f"search-{candidate['id']}.log",
        )
        matches = parse_json_stdout(result, "vq search")
        write_json(search_dir / f"{candidate['id']}.json", matches)
        if not matches:
            continue
        best = matches[0]
        frame_path = Path(best["frame_path"])
        frames.append(
            {
                "id": candidate["id"] + "-semantic",
                "source": "embedding-search",
                "timestamp_seconds": best["timestamp_seconds"],
                "path": relative(frame_path, root),
                "query": query,
                "reasons": candidate["reasons"],
                "similarity": best["similarity"],
                "quality_score": best["quality_score"],
            }
        )
    return frames


def deduplicate_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for frame in frames:
        timestamp = float(frame["timestamp_seconds"])
        duplicate = next(
            (
                existing
                for existing in selected
                if abs(float(existing["timestamp_seconds"]) - timestamp) < 1.0
            ),
            None,
        )
        if duplicate:
            duplicate.setdefault("alternate_sources", []).append(frame["source"])
            continue
        selected.append(frame)
    selected.sort(key=lambda item: float(item["timestamp_seconds"]))
    return selected


def load_metadata(
    source_dir: Path, video: Path, source: str, probe: dict[str, Any]
) -> dict[str, Any]:
    info_files = sorted(source_dir.glob("*.info.json"))
    exact_info = source_dir / f"{video.stem}.info.json"
    if exact_info.is_file():
        info_files = [exact_info, *[path for path in info_files if path != exact_info]]
    info = read_json(info_files[0], {}) if info_files else {}
    return {
        "title": info.get("title") or video.stem,
        "source_url": info.get("webpage_url") or (source if is_url(source) else None),
        "extractor": info.get("extractor_key") or info.get("extractor"),
        "uploader": info.get("uploader") or info.get("channel"),
        "upload_date": info.get("upload_date"),
        "duration_seconds": info.get("duration") or duration_from_probe(probe),
        "description": info.get("description") or "",
        "chapters": info.get("chapters") or [],
        "tags": info.get("tags") or [],
        "categories": info.get("categories") or [],
        "thumbnail": info.get("thumbnail"),
        "local_video": os.fspath(video),
        "info_json": os.fspath(info_files[0]) if info_files else None,
    }


def initialize_analysis(raw_dir: Path) -> None:
    analysis_path = raw_dir / "analysis.json"
    if not analysis_path.exists():
        write_json(
            analysis_path,
            {
                "summary": "",
                "key_points": [],
                "topics": [],
                "timeline": [],
                "limitations": [],
            },
        )


def initialize_observations(raw_dir: Path, frames: list[dict[str, Any]]) -> None:
    existing = read_json(raw_dir / "keyframe-observations.json", {"frames": []})
    existing_by_path = {
        frame.get("path"): frame
        for frame in existing.get("frames") or []
        if frame.get("path")
    }
    merged = []
    for frame in frames:
        previous = existing_by_path.get(frame.get("path"), {})
        merged.append(
            {
                **frame,
                "observation": previous.get("observation", ""),
                "visible_text": previous.get("visible_text", ""),
                "relevance": previous.get("relevance", ""),
                "confidence": previous.get("confidence", ""),
            }
        )
    write_json(
        raw_dir / "keyframe-observations.json",
        {"frames": merged},
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    source_dir = output_dir / "source"
    raw_dir = output_dir / "raw"
    logs = raw_dir / "logs"
    for directory in (output_dir, source_dir, raw_dir, logs):
        directory.mkdir(parents=True, exist_ok=True)

    project_root = find_project_root(args.project_root)
    vq = ensure_vq(project_root, args, logs)
    ensure_ffmpeg(args.install_missing)
    yt_dlp: list[str] | None = None

    doctor_result = run([vq, "--json", "doctor"], log_path=logs / "doctor.log")
    doctor = parse_json_stdout(doctor_result, "vq doctor")
    models: dict[str, dict[str, Any]] = {}
    sensevoice_crosscheck = not args.no_sensevoice_crosscheck and bool(
        doctor.get("sensevoice_runtime") or doctor.get("sensevoice_runtime_supported")
    )

    if is_url(args.source):
        yt_dlp = ensure_yt_dlp(project_root, args.install_missing)
        video = acquire_url(
            args.source,
            source_dir,
            yt_dlp,
            args.subtitle_languages,
            args.cookies_from_browser,
            logs,
        )
    else:
        video = Path(args.source).expanduser().resolve()
        if not video.is_file() or video.suffix.lower() not in MEDIA_EXTENSIONS:
            raise PreparationError(f"unsupported or missing local video: {video}")

    probe = probe_video(video, logs)
    write_json(raw_dir / "video-probe.json", probe)
    metadata = load_metadata(source_dir, video, args.source, probe)
    write_json(raw_dir / "metadata.json", metadata)
    if metadata.get("description"):
        (raw_dir / "description.txt").write_text(
            metadata["description"].rstrip() + "\n", encoding="utf-8"
        )

    chosen_subtitle, subtitles = choose_subtitle(source_dir, video, args.language)
    transcript: dict[str, Any]
    if chosen_subtitle:
        try:
            transcript = transcript_from_subtitle(chosen_subtitle, video, args.language)
            write_json(raw_dir / "transcript.json", transcript)
        except PreparationError as error:
            print(
                f"[prepare] subtitle rejected ({error}); falling back to ASR",
                file=sys.stderr,
            )
            transcript = transcribe_video(
                vq,
                video,
                args.language,
                raw_dir,
                logs,
                args.install_missing,
                sensevoice_crosscheck,
                args.no_model_fetch,
                models,
            )
            chosen_subtitle = None
    elif has_audio_stream(probe):
        transcript = transcribe_video(
            vq,
            video,
            args.language,
            raw_dir,
            logs,
            args.install_missing,
            sensevoice_crosscheck,
            args.no_model_fetch,
            models,
        )
    else:
        print(
            "[prepare] no subtitles or audio stream; continuing with visual evidence",
            file=sys.stderr,
        )
        transcript = empty_transcript(video)
        write_json(raw_dir / "transcript.json", transcript)
    write_transcript_markdown(raw_dir / "transcript.md", transcript)

    duration = float(metadata.get("duration_seconds") or duration_from_probe(probe))
    candidates = build_candidates(transcript, metadata, duration, args.max_candidates)
    write_json(raw_dir / "candidates.json", candidates)
    frames: list[dict[str, Any]] = []
    index_stats: dict[str, Any] | None = None
    index_dir = raw_dir / "index"
    if not args.metadata_only:
        prepare_model_for_use(
            vq,
            "embedding",
            logs,
            args.no_model_fetch,
            models,
            required=True,
        )
        index_result = run(
            [
                vq,
                "--json",
                "--index-dir",
                index_dir,
                "index",
                video,
                "--segment-seconds",
                str(args.segment_seconds),
            ],
            log_path=logs / "index.log",
        )
        finish_model_use(vq, "embedding", logs, models, required=True)
        index_stats = parse_json_stdout(index_result, "vq index")
        write_json(raw_dir / "index-stats.json", index_stats)
        timed = extract_timed_frames(vq, video, candidates, raw_dir, logs, output_dir)
        semantic = semantic_search(
            vq,
            index_dir,
            candidates,
            args.embedding_queries,
            raw_dir,
            logs,
            output_dir,
        )
        frames = deduplicate_frames(timed + semantic)
        write_json(raw_dir / "keyframe-selection.json", frames)
        initialize_observations(raw_dir, frames)
    initialize_analysis(raw_dir)

    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_source": args.source,
        "project_root": os.fspath(project_root),
        "output_dir": os.fspath(output_dir),
        "video": os.fspath(video),
        "subtitle_selected": os.fspath(chosen_subtitle) if chosen_subtitle else None,
        "subtitles_found": [os.fspath(path) for path in subtitles],
        "transcript_engine": transcript.get("engine"),
        "language": transcript.get("language"),
        "candidate_count": len(candidates),
        "selected_frame_count": len(frames),
        "index_stats": index_stats,
        "tools": {
            "vq": {
                "path": os.fspath(vq),
                "version": command_version([os.fspath(vq), "--version"]),
            },
            "yt_dlp": {
                "command": list(yt_dlp),
                "version": command_version([*yt_dlp, "--version"]),
            }
            if yt_dlp
            else None,
            "ffmpeg": command_version(["ffmpeg", "-version"]),
        },
        "doctor": doctor,
        "model_policy": "cache-only" if args.no_model_fetch else "lazy-download",
        "models": models,
        "models_ready": all(
            bool(model.get("ready_after")) or not bool(model.get("required"))
            for model in models.values()
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
