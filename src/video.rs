//! Streaming video decode boundary.
//!
//! Codec parsing is delegated to the system FFmpeg binary. Frames are scaled
//! and converted to one-byte luma before entering Rust, keeping IPC volume and
//! frame-scoring cost bounded for 4K/8K sources.

use crate::keyframe::{KeyframeCandidate, score_luma};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub duration_seconds: f64,
    pub frames_per_second: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ScanOptions {
    pub frames_per_second: f64,
    pub start_seconds: f64,
    pub duration_seconds: Option<f64>,
    pub maximum_long_edge: usize,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            frames_per_second: 4.0,
            start_seconds: 0.0,
            duration_seconds: None,
            maximum_long_edge: 640,
        }
    }
}

#[derive(Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
    format: ProbeFormat,
}

#[derive(Deserialize)]
struct ProbeStream {
    width: Option<usize>,
    height: Option<usize>,
    avg_frame_rate: Option<String>,
}

#[derive(Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

pub fn ensure_ffmpeg_available() -> Result<()> {
    for binary in ["ffmpeg", "ffprobe"] {
        let status = Command::new(binary)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| format!("failed to launch {binary}"))?;
        if !status.success() {
            bail!("{binary} is installed but did not start successfully")
        }
    }
    Ok(())
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .with_context(|| format!("failed to run ffprobe for {}", path.display()))?;
    if !output.status.success() {
        bail!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }

    let parsed: ProbeOutput = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("invalid ffprobe JSON for {}", path.display()))?;
    let stream = parsed
        .streams
        .first()
        .ok_or_else(|| anyhow::anyhow!("{} has no video stream", path.display()))?;
    let width = stream.width.context("video stream has no width")?;
    let height = stream.height.context("video stream has no height")?;
    let duration_seconds = parsed
        .format
        .duration
        .as_deref()
        .and_then(|value| value.parse().ok())
        .filter(|value: &f64| value.is_finite() && *value > 0.0)
        .context("video has no finite positive duration")?;
    let frames_per_second = stream
        .avg_frame_rate
        .as_deref()
        .and_then(parse_fraction)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(30.0);

    Ok(VideoInfo {
        width,
        height,
        duration_seconds,
        frames_per_second,
    })
}

/// Stream sampled luma frames and score them in Rust.
pub fn scan_quality(path: &Path, options: ScanOptions) -> Result<Vec<KeyframeCandidate>> {
    if !(options.frames_per_second.is_finite() && options.frames_per_second > 0.0) {
        bail!("scan FPS must be finite and greater than zero")
    }
    if !(options.start_seconds.is_finite() && options.start_seconds >= 0.0) {
        bail!("scan start must be finite and non-negative")
    }
    if let Some(duration) = options.duration_seconds
        && !(duration.is_finite() && duration > 0.0)
    {
        bail!("scan duration must be finite and greater than zero")
    }
    if options.maximum_long_edge < 32 {
        bail!("maximum scan edge must be at least 32 pixels")
    }

    let info = probe(path)?;
    let (width, height) = scaled_dimensions(info.width, info.height, options.maximum_long_edge);
    let mut command = Command::new("ffmpeg");
    command.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    if options.start_seconds > 0.0 {
        command.args(["-ss", &format_seconds(options.start_seconds)]);
    }
    command.arg("-i").arg(path);
    if let Some(duration) = options.duration_seconds {
        command.args(["-t", &format_seconds(duration)]);
    }
    command
        .args(["-an", "-sn", "-dn", "-vf"])
        .arg(format!(
            "fps={:.8},scale={width}:{height}:flags=fast_bilinear,format=gray",
            options.frames_per_second
        ))
        .args(["-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start ffmpeg for {}", path.display()))?;
    let mut stdout = child.stdout.take().context("ffmpeg stdout was not piped")?;
    let mut stderr = child.stderr.take().context("ffmpeg stderr was not piped")?;
    let stderr_thread = thread::spawn(move || {
        let mut message = String::new();
        let _ = stderr.read_to_string(&mut message);
        message
    });

    let frame_bytes = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("scaled frame dimensions overflow"))?;
    let mut buffer = vec![0_u8; frame_bytes];
    let mut candidates = Vec::new();
    let mut frame_index = 0_u64;

    loop {
        match read_exact_or_eof(&mut stdout, &mut buffer)? {
            false => break,
            true => {
                let quality = score_luma(&buffer, width, height, width)?;
                candidates.push(KeyframeCandidate {
                    timestamp_seconds: options.start_seconds
                        + frame_index as f64 / options.frames_per_second,
                    quality,
                });
                frame_index += 1;
            }
        }
    }

    let status = child.wait().context("failed waiting for ffmpeg")?;
    let stderr_message = stderr_thread.join().unwrap_or_default();
    if !status.success() {
        bail!(
            "ffmpeg frame scan failed for {}: {}",
            path.display(),
            stderr_message.trim()
        )
    }
    Ok(candidates)
}

pub fn best_near(
    path: &Path,
    at_seconds: f64,
    radius_seconds: f64,
    frames_per_second: f64,
) -> Result<KeyframeCandidate> {
    if !(at_seconds.is_finite() && at_seconds >= 0.0) {
        bail!("target timestamp must be finite and non-negative")
    }
    if !(radius_seconds.is_finite() && radius_seconds >= 0.0) {
        bail!("search radius must be finite and non-negative")
    }
    let info = probe(path)?;
    let start = (at_seconds - radius_seconds).max(0.0);
    let end = (at_seconds + radius_seconds).min(info.duration_seconds);
    let candidates = scan_quality(
        path,
        ScanOptions {
            frames_per_second,
            start_seconds: start,
            duration_seconds: Some((end - start).max(1.0 / frames_per_second)),
            ..ScanOptions::default()
        },
    )?;
    candidates
        .into_iter()
        .max_by(|left, right| {
            left.quality
                .score
                .total_cmp(&right.quality.score)
                .then_with(|| {
                    let right_distance = (right.timestamp_seconds - at_seconds).abs();
                    let left_distance = (left.timestamp_seconds - at_seconds).abs();
                    right_distance.total_cmp(&left_distance)
                })
        })
        .context("ffmpeg returned no frames in the requested interval")
}

/// Pick the best candidate in every fixed time segment.
pub fn best_per_segment(
    candidates: impl IntoIterator<Item = KeyframeCandidate>,
    segment_seconds: f64,
) -> Result<Vec<KeyframeCandidate>> {
    if !(segment_seconds.is_finite() && segment_seconds > 0.0) {
        bail!("segment duration must be finite and greater than zero")
    }
    let mut selected: BTreeMap<usize, KeyframeCandidate> = BTreeMap::new();
    for candidate in candidates {
        let segment = (candidate.timestamp_seconds / segment_seconds).floor() as usize;
        selected
            .entry(segment)
            .and_modify(|current| {
                if candidate.quality.score > current.quality.score {
                    *current = candidate.clone();
                }
            })
            .or_insert(candidate);
    }
    Ok(selected.into_values().collect())
}

pub fn extract_jpeg(
    video: &Path,
    timestamp_seconds: f64,
    output: &Path,
    maximum_long_edge: Option<usize>,
) -> Result<()> {
    if !(timestamp_seconds.is_finite() && timestamp_seconds >= 0.0) {
        bail!("frame timestamp must be finite and non-negative")
    }
    if maximum_long_edge == Some(0) {
        bail!("maximum extraction edge must be greater than zero")
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            &format_seconds(timestamp_seconds),
            "-i",
        ])
        .arg(video)
        .args(["-frames:v", "1"]);
    if let Some(edge) = maximum_long_edge {
        command.args(["-vf", &format!("scale='min({edge},iw)':-2")]);
    }
    let output_result = command
        .args(["-q:v", "2"])
        .arg(output)
        .output()
        .with_context(|| format!("failed to extract frame from {}", video.display()))?;
    if !output_result.status.success() {
        bail!(
            "ffmpeg failed to extract {}: {}",
            output.display(),
            String::from_utf8_lossy(&output_result.stderr).trim()
        )
    }
    Ok(())
}

pub fn is_supported_video(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "mp4" | "mov" | "m4v" | "mkv" | "webm" | "avi" | "mts" | "m2ts"
            )
        })
        .unwrap_or(false)
}

pub fn collect_videos(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut videos = Vec::new();
    for input in inputs {
        if input.is_file() {
            if is_supported_video(input) {
                videos.push(input.canonicalize()?);
            }
            continue;
        }
        if input.is_dir() {
            for entry in walkdir::WalkDir::new(input).follow_links(false) {
                let entry = entry?;
                if entry.file_type().is_file() && is_supported_video(entry.path()) {
                    videos.push(entry.path().canonicalize()?);
                }
            }
            continue;
        }
        bail!("input does not exist: {}", input.display())
    }
    videos.sort();
    videos.dedup();
    Ok(videos)
}

fn read_exact_or_eof(reader: &mut impl Read, buffer: &mut [u8]) -> Result<bool> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..])? {
            0 if filled == 0 => return Ok(false),
            0 => bail!("ffmpeg ended with a partial raw video frame"),
            count => filled += count,
        }
    }
    Ok(true)
}

fn scaled_dimensions(width: usize, height: usize, maximum_long_edge: usize) -> (usize, usize) {
    let long_edge = width.max(height);
    if long_edge <= maximum_long_edge {
        return (width, height);
    }
    let scale = maximum_long_edge as f64 / long_edge as f64;
    let scaled_width = ((width as f64 * scale).round() as usize).max(2);
    let scaled_height = ((height as f64 * scale).round() as usize).max(2);
    (scaled_width, scaled_height)
}

fn parse_fraction(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/')?;
    let numerator: f64 = numerator.parse().ok()?;
    let denominator: f64 = denominator.parse().ok()?;
    (denominator != 0.0).then_some(numerator / denominator)
}

fn format_seconds(value: f64) -> String {
    format!("{value:.6}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyframe::FrameQuality;

    fn candidate(timestamp: f64, score: f64) -> KeyframeCandidate {
        KeyframeCandidate {
            timestamp_seconds: timestamp,
            quality: FrameQuality {
                score,
                ..FrameQuality::default()
            },
        }
    }

    #[test]
    fn parses_frame_rate_fraction() {
        assert_eq!(parse_fraction("30000/1001").unwrap(), 30000.0 / 1001.0);
        assert!(parse_fraction("30/0").is_none());
    }

    #[test]
    fn scales_without_changing_aspect_ratio_materially() {
        assert_eq!(scaled_dimensions(3840, 2160, 640), (640, 360));
        assert_eq!(scaled_dimensions(320, 240, 640), (320, 240));
    }

    #[test]
    fn selects_best_frame_in_each_segment() {
        let selected = best_per_segment(
            [
                candidate(0.0, 1.0),
                candidate(4.0, 3.0),
                candidate(11.0, 2.0),
                candidate(15.0, 4.0),
            ],
            10.0,
        )
        .unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].timestamp_seconds, 4.0);
        assert_eq!(selected[1].timestamp_seconds, 15.0);
    }
}
