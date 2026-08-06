use std::process::{Command, Stdio};
use video_sherlock::video::{ScanOptions, best_per_segment, extract_jpeg, scan_quality};

#[test]
fn decodes_scores_selects_and_extracts_real_video() {
    let ffmpeg = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if !matches!(ffmpeg, Ok(status) if status.success()) {
        eprintln!("skipping FFmpeg integration test: ffmpeg is unavailable");
        return;
    }

    let directory = tempfile::tempdir().unwrap();
    let video = directory.path().join("fixture.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=2",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&video)
        .status()
        .unwrap();
    assert!(status.success());

    let candidates = scan_quality(
        &video,
        ScanOptions {
            frames_per_second: 4.0,
            maximum_long_edge: 160,
            ..ScanOptions::default()
        },
    )
    .unwrap();
    assert_eq!(candidates.len(), 8);

    let selected = best_per_segment(candidates, 1.0).unwrap();
    assert_eq!(selected.len(), 2);
    assert!(selected.iter().all(|frame| frame.quality.score > 0.0));

    let jpeg = directory.path().join("selected.jpg");
    extract_jpeg(&video, selected[0].timestamp_seconds, &jpeg, Some(128)).unwrap();
    let decoded = image::open(jpeg).unwrap();
    assert_eq!(decoded.width(), 128);
    assert_eq!(decoded.height(), 72);
}
