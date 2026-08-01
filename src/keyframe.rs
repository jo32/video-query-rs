//! High-performance, allocation-free frame quality analysis.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

const TARGET_LONG_EDGE_SAMPLES: usize = 640;

/// Measurements used to rank a decoded video frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameQuality {
    pub score: f64,
    pub laplacian_variance: f64,
    pub gradient_rms: f64,
    pub mean_luma: f64,
    pub contrast: f64,
    pub clipped_fraction: f64,
    pub sampled_pixels: usize,
}

/// A timestamped candidate produced by the streaming decoder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeyframeCandidate {
    pub timestamp_seconds: f64,
    pub quality: FrameQuality,
}

/// Score an 8-bit luma plane without allocating intermediate images.
pub fn score_luma(
    plane: &[u8],
    width: usize,
    height: usize,
    bytes_per_row: usize,
) -> Result<FrameQuality> {
    if width < 3 || height < 3 {
        bail!("luma frame must be at least 3x3 pixels")
    }
    if bytes_per_row < width {
        bail!("row stride {bytes_per_row} is smaller than width {width}")
    }
    let required = bytes_per_row
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("luma frame dimensions overflow"))?;
    if plane.len() < required {
        bail!(
            "luma plane has {} bytes, but {required} are required",
            plane.len()
        )
    }

    let step = width.max(height).div_ceil(TARGET_LONG_EDGE_SAMPLES).max(1);

    let mut samples = 0_u64;
    let mut luma_sum = 0.0_f64;
    let mut luma_squared_sum = 0.0_f64;
    let mut laplacian_sum = 0.0_f64;
    let mut laplacian_squared_sum = 0.0_f64;
    let mut gradient_squared_sum = 0.0_f64;
    let mut clipped = 0_u64;

    for y in (1..height - 1).step_by(step) {
        let previous = (y - 1) * bytes_per_row;
        let current = y * bytes_per_row;
        let next = (y + 1) * bytes_per_row;

        for x in (1..width - 1).step_by(step) {
            let center = f64::from(plane[current + x]);
            let left = f64::from(plane[current + x - 1]);
            let right = f64::from(plane[current + x + 1]);
            let above = f64::from(plane[previous + x]);
            let below = f64::from(plane[next + x]);

            let laplacian = 4.0 * center - left - right - above - below;
            let gradient_x = right - left;
            let gradient_y = below - above;

            samples += 1;
            luma_sum += center;
            luma_squared_sum += center * center;
            laplacian_sum += laplacian;
            laplacian_squared_sum += laplacian * laplacian;
            gradient_squared_sum += gradient_x * gradient_x + gradient_y * gradient_y;
            clipped += u64::from(center <= 8.0 || center >= 247.0);
        }
    }

    if samples == 0 {
        bail!("no pixels were available for frame analysis")
    }

    let count = samples as f64;
    let mean_luma = luma_sum / count;
    let luma_variance = non_negative(luma_squared_sum / count - mean_luma * mean_luma);
    let mean_laplacian = laplacian_sum / count;
    let laplacian_variance =
        non_negative(laplacian_squared_sum / count - mean_laplacian * mean_laplacian);
    let gradient_rms = (gradient_squared_sum / count).sqrt();
    let contrast = luma_variance.sqrt();
    let clipped_fraction = clipped as f64 / count;

    // Focus dominates. The remaining terms prevent a dark/noisy transition or
    // a blown-out frame from winning on compression edges alone.
    let focus = laplacian_variance.sqrt() + 0.35 * gradient_rms;
    let centered_exposure = 1.0 - ((mean_luma - 127.5).abs() / 127.5);
    let exposure_weight = 0.35 + 0.65 * centered_exposure.clamp(0.0, 1.0);
    let clipping_weight = (1.0 - 0.75 * clipped_fraction).clamp(0.1, 1.0);
    let contrast_weight = (0.55 + 0.45 * (contrast / 48.0)).clamp(0.55, 1.0);

    Ok(FrameQuality {
        score: focus * exposure_weight * clipping_weight * contrast_weight,
        laplacian_variance,
        gradient_rms,
        mean_luma,
        contrast,
        clipped_fraction,
        sampled_pixels: samples as usize,
    })
}

fn non_negative(value: f64) -> f64 {
    if value > 0.0 { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_frame_has_no_focus_signal() {
        let image = vec![128_u8; 64 * 48];
        let quality = score_luma(&image, 64, 48, 64).unwrap();
        assert_eq!(quality.laplacian_variance, 0.0);
        assert_eq!(quality.gradient_rms, 0.0);
        assert_eq!(quality.score, 0.0);
    }

    #[test]
    fn sharp_checkerboard_beats_low_contrast_version() {
        let width = 64;
        let height = 64;
        let mut sharp = vec![0_u8; width * height];
        let mut soft = vec![0_u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let high = ((x / 4) + (y / 4)) % 2 == 0;
                sharp[y * width + x] = if high { 230 } else { 25 };
                soft[y * width + x] = if high { 165 } else { 90 };
            }
        }
        let sharp_score = score_luma(&sharp, width, height, width).unwrap().score;
        let soft_score = score_luma(&soft, width, height, width).unwrap().score;
        assert!(sharp_score > soft_score * 2.0);
    }

    #[test]
    fn respects_padded_rows() {
        let width = 8;
        let height = 8;
        let stride = 16;
        let mut image = vec![255_u8; stride * height];
        for y in 0..height {
            image[y * stride..y * stride + width].fill(128);
        }
        let quality = score_luma(&image, width, height, stride).unwrap();
        assert_eq!(quality.mean_luma, 128.0);
        assert_eq!(quality.score, 0.0);
    }

    #[test]
    fn rejects_invalid_storage() {
        assert!(score_luma(&[0; 8], 8, 8, 8).is_err());
        assert!(score_luma(&[0; 64], 8, 8, 7).is_err());
    }
}
