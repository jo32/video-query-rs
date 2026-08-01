//! Native Rust Chinese image/text embeddings powered by Candle.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::chinese_clip::{ChineseClipConfig, ChineseClipModel, div_l2_norm};
use hf_hub::Cache;
use hf_hub::api::Progress;
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::{Model, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub const MODEL_ID: &str = "OFA-Sys/chinese-clip-vit-base-patch16";
pub const MODEL_VERSION: &str = "chinese-clip-vit-b16-candle-f32-v1";
pub const EMBEDDING_DIMENSION: usize = 512;
const IMAGE_SIZE: usize = 224;
const TEXT_LENGTH: usize = 52;
const WEIGHTS_FILENAME: &str = "pytorch_model.bin";
const VOCABULARY_FILENAME: &str = "vocab.txt";

#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub weights: PathBuf,
    pub vocabulary: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatus {
    pub model: &'static str,
    pub ready: bool,
    pub cache_directory: PathBuf,
    pub weights: Option<PathBuf>,
    pub weights_bytes: Option<u64>,
    pub vocabulary: Option<PathBuf>,
    pub vocabulary_bytes: Option<u64>,
    pub missing_files: Vec<&'static str>,
}

/// Inspect the configured Hugging Face cache without making a network request.
pub fn model_status() -> ModelStatus {
    model_status_in(&Cache::from_env())
}

fn model_status_in(cache: &Cache) -> ModelStatus {
    let repository = cache.model(MODEL_ID.to_string());
    let weights = repository
        .get(WEIGHTS_FILENAME)
        .filter(|path| valid_file_size(path).is_some());
    let vocabulary = repository
        .get(VOCABULARY_FILENAME)
        .filter(|path| valid_file_size(path).is_some());
    let weights_bytes = weights.as_deref().and_then(valid_file_size);
    let vocabulary_bytes = vocabulary.as_deref().and_then(valid_file_size);
    let mut missing_files = Vec::new();
    if weights.is_none() {
        missing_files.push(WEIGHTS_FILENAME);
    }
    if vocabulary.is_none() {
        missing_files.push(VOCABULARY_FILENAME);
    }
    ModelStatus {
        model: MODEL_ID,
        ready: missing_files.is_empty(),
        cache_directory: cache.path().clone(),
        weights,
        weights_bytes,
        vocabulary,
        vocabulary_bytes,
        missing_files,
    }
}

pub fn fetch_model_files() -> Result<ModelFiles> {
    let status = model_status();
    eprintln!("[model] check id={MODEL_ID}");
    eprintln!("[model] cache={}", status.cache_directory.display());

    if let (Some(weights), Some(vocabulary)) = (status.weights.clone(), status.vocabulary.clone()) {
        eprintln!(
            "[model] status=ready weights={} vocabulary={}",
            weights.display(),
            vocabulary.display()
        );
        return Ok(ModelFiles {
            weights,
            vocabulary,
        });
    }

    eprintln!(
        "[model] status=missing files={} action=download",
        status.missing_files.join(",")
    );
    let api = ApiBuilder::from_env()
        .with_progress(false)
        .with_retries(2)
        .with_user_agent("video-query-rs", env!("CARGO_PKG_VERSION"))
        .build()
        .context("failed to initialize the Hugging Face client")?;
    let repository = api.model(MODEL_ID.to_string());
    let weights = fetch_file(
        &repository,
        WEIGHTS_FILENAME,
        status.weights,
        status.weights_bytes,
    )?;
    let vocabulary = fetch_file(
        &repository,
        VOCABULARY_FILENAME,
        status.vocabulary,
        status.vocabulary_bytes,
    )?;
    eprintln!("[model] status=ready action=continue");
    Ok(ModelFiles {
        weights,
        vocabulary,
    })
}

fn fetch_file(
    repository: &ApiRepo,
    filename: &'static str,
    cached: Option<PathBuf>,
    cached_bytes: Option<u64>,
) -> Result<PathBuf> {
    if let Some(path) = cached {
        eprintln!(
            "[model] file={filename} status=found bytes={} path={}",
            cached_bytes.unwrap_or_default(),
            path.display()
        );
        return Ok(path);
    }

    eprintln!("[model] file={filename} status=missing action=download");
    let path = repository
        .download_with_progress(filename, AgentDownloadProgress::default())
        .with_context(|| format!("failed to download {MODEL_ID}/{filename}"))?;
    let bytes = valid_file_size(&path)
        .with_context(|| format!("downloaded model file is empty: {}", path.display()))?;
    eprintln!(
        "[model] file={filename} status=stored bytes={bytes} path={}",
        path.display()
    );
    Ok(path)
}

fn valid_file_size(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() > 0)
        .map(|metadata| metadata.len())
}

#[derive(Default)]
struct AgentDownloadProgress {
    filename: String,
    total: usize,
    downloaded: usize,
    last_reported_bucket: usize,
    initialized: bool,
}

impl Progress for AgentDownloadProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.total = size;
        self.downloaded = 0;
        self.last_reported_bucket = 0;
        self.filename = filename.to_string();
        if !self.initialized {
            eprintln!(
                "[model] download-start file={} total_bytes={} total={}",
                self.filename,
                self.total,
                human_bytes(self.total)
            );
            self.initialized = true;
        }
    }

    fn update(&mut self, size: usize) {
        self.downloaded = self.downloaded.saturating_add(size).min(self.total);
        if self.total == 0 {
            return;
        }
        let percent = self.downloaded.saturating_mul(100) / self.total;
        let bucket = percent / 5;
        if bucket > self.last_reported_bucket || self.downloaded == self.total {
            self.last_reported_bucket = bucket;
            eprintln!(
                "[model] download-progress file={} percent={} downloaded_bytes={} total_bytes={}",
                self.filename, percent, self.downloaded, self.total
            );
        }
    }

    fn finish(&mut self) {
        eprintln!(
            "[model] download-complete file={} bytes={}",
            self.filename, self.total
        );
    }
}

fn human_bytes(bytes: usize) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

pub struct ChineseClipEmbedder {
    model: ChineseClipModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl ChineseClipEmbedder {
    pub fn load() -> Result<Self> {
        let files = fetch_model_files()?;
        Self::load_from_files(&files)
    }

    pub fn load_from_files(files: &ModelFiles) -> Result<Self> {
        let device = Device::Cpu;
        let configuration = ChineseClipConfig::clip_vit_base_patch16();
        let variables = VarBuilder::from_pth(&files.weights, DType::F32, &device)
            .with_context(|| format!("failed to read {}", files.weights.display()))?;
        let model = ChineseClipModel::new(variables, &configuration)
            .context("failed to construct Chinese-CLIP")?;
        let tokenizer = build_tokenizer(&files.vocabulary)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            bail!("search query must not be empty")
        }
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("Chinese tokenizer failed: {error}"))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(encoding.get_type_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;
        let features = self.model.get_text_features(
            &input_ids,
            Some(&token_type_ids),
            Some(&attention_mask),
        )?;
        normalized_vectors(features)?
            .into_iter()
            .next()
            .context("Chinese-CLIP returned no text embedding")
    }

    pub fn embed_images(&self, paths: &[PathBuf]) -> Result<Vec<Vec<f32>>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let mut images = Vec::with_capacity(paths.len());
        for path in paths {
            let image = image::open(path)
                .with_context(|| format!("failed to decode image {}", path.display()))?;
            images.push(preprocess_image(image, &self.device)?);
        }
        let references: Vec<&Tensor> = images.iter().collect();
        let batch = Tensor::cat(&references, 0)?;
        let features = self.model.get_image_features(&batch)?;
        normalized_vectors(features)
    }

    pub fn embed_image(&self, path: &Path) -> Result<Vec<f32>> {
        self.embed_images(&[path.to_path_buf()])?
            .into_iter()
            .next()
            .context("Chinese-CLIP returned no image embedding")
    }
}

fn build_tokenizer(vocabulary: &Path) -> Result<Tokenizer> {
    let vocabulary = vocabulary
        .to_str()
        .context("Chinese vocabulary path is not valid UTF-8")?;
    let wordpiece = WordPiece::from_file(vocabulary)
        .unk_token("[UNK]".to_string())
        .build()
        .map_err(|error| anyhow::anyhow!("failed to load Chinese vocabulary: {error}"))?;
    let sep = wordpiece
        .token_to_id("[SEP]")
        .context("Chinese vocabulary is missing [SEP]")?;
    let cls = wordpiece
        .token_to_id("[CLS]")
        .context("Chinese vocabulary is missing [CLS]")?;
    let pad = wordpiece
        .token_to_id("[PAD]")
        .context("Chinese vocabulary is missing [PAD]")?;

    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer
        .with_normalizer(Some(BertNormalizer::new(true, true, None, true)))
        .map_err(|error| anyhow::anyhow!("failed to configure Chinese normalization: {error}"))?
        .with_pre_tokenizer(Some(BertPreTokenizer))
        .with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_string(), sep),
            ("[CLS]".to_string(), cls),
        )));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: TEXT_LENGTH,
            ..TruncationParams::default()
        }))
        .map_err(|error| anyhow::anyhow!("failed to configure tokenizer truncation: {error}"))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(TEXT_LENGTH),
        pad_id: pad,
        ..PaddingParams::default()
    }));
    Ok(tokenizer)
}

fn preprocess_image(image: DynamicImage, device: &Device) -> Result<Tensor> {
    let (original_width, original_height) = image.dimensions();
    if original_width == 0 || original_height == 0 {
        bail!("image has invalid dimensions")
    }

    let scale = IMAGE_SIZE as f64 / f64::from(original_width.min(original_height));
    let resized_width = (f64::from(original_width) * scale).round() as u32;
    let resized_height = (f64::from(original_height) * scale).round() as u32;
    let resized = image
        .resize_exact(resized_width, resized_height, FilterType::CatmullRom)
        .to_rgb8();
    let offset_x = (resized_width - IMAGE_SIZE as u32) / 2;
    let offset_y = (resized_height - IMAGE_SIZE as u32) / 2;
    let cropped = image::imageops::crop_imm(
        &resized,
        offset_x,
        offset_y,
        IMAGE_SIZE as u32,
        IMAGE_SIZE as u32,
    )
    .to_image();

    let pixels = cropped.as_raw();
    let area = IMAGE_SIZE * IMAGE_SIZE;
    let mean = [0.481_454_66_f32, 0.457_827_5, 0.408_210_73];
    let standard_deviation = [0.268_629_54_f32, 0.261_302_6, 0.275_777_1];
    let mut planar = vec![0.0_f32; area * 3];
    for pixel_index in 0..area {
        for channel in 0..3 {
            let value = f32::from(pixels[pixel_index * 3 + channel]) / 255.0;
            planar[channel * area + pixel_index] =
                (value - mean[channel]) / standard_deviation[channel];
        }
    }
    Ok(Tensor::from_vec(
        planar,
        (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        device,
    )?)
}

fn normalized_vectors(features: Tensor) -> Result<Vec<Vec<f32>>> {
    let normalized = div_l2_norm(&features)?;
    let shape = normalized.dims2()?;
    if shape.1 != EMBEDDING_DIMENSION {
        bail!(
            "expected {EMBEDDING_DIMENSION}-dimensional embeddings, got {}",
            shape.1
        )
    }
    Ok(normalized.to_vec2::<f32>()?)
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!(
            "embedding dimensions differ: {} versus {}",
            left.len(),
            right.len()
        )
    }
    Ok(left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_dot_product_is_cosine_similarity() {
        let left = [1.0_f32, 0.0, 0.0];
        let right = [0.5_f32, 0.5, 0.0];
        assert_eq!(cosine_similarity(&left, &right).unwrap(), 0.5);
    }

    #[test]
    fn rejects_mismatched_dimensions() {
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn empty_cache_reports_both_required_files() {
        let directory = tempfile::tempdir().unwrap();
        let status = model_status_in(&Cache::new(directory.path().to_path_buf()));
        assert!(!status.ready);
        assert_eq!(
            status.missing_files,
            vec![WEIGHTS_FILENAME, VOCABULARY_FILENAME]
        );
        assert!(status.weights.is_none());
        assert!(status.vocabulary.is_none());
    }

    #[test]
    fn download_progress_tracks_resumed_and_new_bytes() {
        let mut progress = AgentDownloadProgress::default();
        progress.init(1_000, WEIGHTS_FILENAME);
        progress.init(1_000, WEIGHTS_FILENAME);
        progress.update(400);
        progress.update(100);
        assert_eq!(progress.downloaded, 500);
        assert_eq!(progress.last_reported_bucket, 10);
    }

    #[test]
    fn formats_model_sizes_for_logs() {
        assert_eq!(human_bytes(1024 * 1024), "1.0 MiB");
        assert_eq!(human_bytes(2 * 1024 * 1024 * 1024), "2.00 GiB");
    }
}
