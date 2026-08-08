from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "analyze-video"
    / "scripts"
    / "prepare_video.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_video", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
prepare_video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_video)


class PrepareVideoTests(unittest.TestCase):
    def test_missing_model_is_deferred_to_the_stage_command(self) -> None:
        status = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ready": False, "model": "whisper-model"}),
            stderr="",
        )
        models: dict[str, dict[str, object]] = {}

        with mock.patch.object(prepare_video, "run", return_value=status) as run:
            should_run = prepare_video.prepare_model_for_use(
                Path("vq"),
                "whisper",
                Path("logs"),
                False,
                models,
                required=True,
            )

        self.assertTrue(should_run)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args.args[0],
            [Path("vq"), "--json", "model", "status-whisper"],
        )
        self.assertFalse(models["whisper"]["ready_before"])
        self.assertFalse(models["whisper"]["downloaded_on_use"])

    def test_cache_only_policy_fails_for_a_required_missing_model(self) -> None:
        status = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"embedding": {"ready": False, "model": "clip"}}),
            stderr="",
        )

        with mock.patch.object(prepare_video, "run", return_value=status):
            with self.assertRaisesRegex(
                prepare_video.PreparationError,
                "embedding model is required but not cached",
            ):
                prepare_video.prepare_model_for_use(
                    Path("vq"),
                    "embedding",
                    Path("logs"),
                    True,
                    {},
                    required=True,
                )

    def test_cache_only_policy_skips_an_optional_missing_model(self) -> None:
        status = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ready": False, "model": "sensevoice"}),
            stderr="",
        )
        models: dict[str, dict[str, object]] = {}

        with mock.patch.object(prepare_video, "run", return_value=status):
            should_run = prepare_video.prepare_model_for_use(
                Path("vq"),
                "sensevoice",
                Path("logs"),
                True,
                models,
                required=False,
            )

        self.assertFalse(should_run)
        self.assertIn("skipped", models["sensevoice"])

    def test_model_record_marks_a_lazy_download_after_use(self) -> None:
        before = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ready": False, "model": "whisper-model"}),
            stderr="",
        )
        after = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"ready": True, "model": "whisper-model"}),
            stderr="",
        )
        models: dict[str, dict[str, object]] = {}

        with mock.patch.object(
            prepare_video, "run", side_effect=[before, after]
        ):
            prepare_video.prepare_model_for_use(
                Path("vq"),
                "whisper",
                Path("logs"),
                False,
                models,
                required=True,
            )
            prepare_video.finish_model_use(
                Path("vq"),
                "whisper",
                Path("logs"),
                models,
                required=True,
            )

        self.assertTrue(models["whisper"]["ready_after"])
        self.assertTrue(models["whisper"]["downloaded_on_use"])

    def test_browser_cookie_store_is_forwarded_only_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory)
            video = source_dir / "downloaded.mp4"
            video.touch()
            result = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=f"{video}\n", stderr=""
            )

            with mock.patch.object(prepare_video, "run", return_value=result) as run:
                selected = prepare_video.acquire_url(
                    "https://example.com/video",
                    source_dir,
                    ["yt-dlp"],
                    "en.*",
                    "chrome",
                    source_dir / "logs",
                )

            command = run.call_args.args[0]
            self.assertEqual(selected, video.resolve())
            self.assertEqual(
                command[-3:],
                ["--cookies-from-browser", "chrome", "https://example.com/video"],
            )

            with mock.patch.object(
                prepare_video, "run", return_value=result
            ) as run_without_cookies:
                prepare_video.acquire_url(
                    "https://example.com/video",
                    source_dir,
                    ["yt-dlp"],
                    "en.*",
                    None,
                    source_dir / "logs",
                )

            command_without_cookies = run_without_cookies.call_args.args[0]
            self.assertNotIn("--cookies-from-browser", command_without_cookies)

    def test_candidate_timestamps_stay_inside_video_duration(self) -> None:
        transcript = {
            "segments": [
                {
                    "start_seconds": 9.8,
                    "end_seconds": 10.4,
                    "text": "This chart shows 1000 results",
                }
            ]
        }

        candidates = prepare_video.build_candidates(transcript, {}, 10.0, 4)

        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate["timestamp_seconds"] <= 9.5 for candidate in candidates)
        )

    def test_exact_local_info_json_is_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory)
            video = source_dir / "clip.mp4"
            video.touch()
            (source_dir / "another.info.json").write_text(
                json.dumps({"title": "Wrong title"}), encoding="utf-8"
            )
            exact_info = source_dir / "clip.info.json"
            exact_info.write_text(
                json.dumps(
                    {
                        "title": "Exact title",
                        "webpage_url": "https://example.com/exact",
                    }
                ),
                encoding="utf-8",
            )

            metadata = prepare_video.load_metadata(
                source_dir,
                video,
                str(video),
                {"format": {"duration": "12.5"}},
            )

            self.assertEqual(metadata["title"], "Exact title")
            self.assertEqual(metadata["source_url"], "https://example.com/exact")
            self.assertEqual(metadata["info_json"], str(exact_info))


if __name__ == "__main__":
    unittest.main()
