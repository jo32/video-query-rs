//! Local speech-to-text orchestration through FFmpeg, SenseVoice, and whisper.cpp.

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use hf_hub::Cache;
use hf_hub::api::Progress;
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const SPEECH_MODEL_ID: &str = "FunAudioLLM/SenseVoiceSmall-GGUF";
pub const SPEECH_MODEL_FILE: &str = "sensevoice-small-q8.gguf";
pub const SPEECH_VAD_MODEL_ID: &str = "FunAudioLLM/fsmn-vad-GGUF";
pub const SPEECH_VAD_MODEL_FILE: &str = "fsmn-vad.gguf";
#[cfg(not(windows))]
pub const SENSEVOICE_RUNTIME_BINARY: &str = "llama-funasr-sensevoice";
#[cfg(windows)]
pub const SENSEVOICE_RUNTIME_BINARY: &str = "llama-funasr-sensevoice.exe";
pub const SENSEVOICE_RUNTIME_VERSION: &str = "runtime-llamacpp-v0.1.9";

pub const WHISPER_MODEL_ID: &str = "ggerganov/whisper.cpp";
pub const WHISPER_MODEL_FILE: &str = "ggml-large-v3-turbo-q5_0.bin";

const RUNTIME_RELEASE_BASE: &str = "https://github.com/QwenAudio/SenseVoice/releases/download";

#[derive(Clone, Debug, Serialize)]
pub struct SpeechModelStatus {
    pub model: &'static str,
    pub file: &'static str,
    pub vad_model: &'static str,
    pub vad_file: &'static str,
    pub ready: bool,
    pub models_ready: bool,
    pub runtime_ready: bool,
    pub runtime_supported: bool,
    pub cache_directory: PathBuf,
    pub weights: Option<PathBuf>,
    pub weights_bytes: Option<u64>,
    pub vad_weights: Option<PathBuf>,
    pub vad_weights_bytes: Option<u64>,
    pub runtime: Option<PathBuf>,
    pub missing_files: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpeechModelFiles {
    pub weights: PathBuf,
    pub vad_weights: PathBuf,
    pub runtime: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct WhisperModelStatus {
    pub model: &'static str,
    pub file: &'static str,
    pub ready: bool,
    pub cache_directory: PathBuf,
    pub weights: Option<PathBuf>,
    pub weights_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct SenseVoiceOptions<'a> {
    pub input: &'a Path,
    pub model: &'a Path,
    pub vad_model: &'a Path,
    pub runtime: &'a Path,
}

#[derive(Clone, Debug)]
pub struct WhisperOptions<'a> {
    pub input: &'a Path,
    pub model: &'a Path,
    pub language: &'a str,
    pub prompt: Option<&'a str>,
    pub threads: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TranscriptSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Transcript {
    pub engine: String,
    pub input: PathBuf,
    pub model: PathBuf,
    pub language: String,
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub emotions: Vec<String>,
    pub audio_events: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WhisperOutput {
    result: WhisperResult,
    transcription: Vec<WhisperSegment>,
}

#[derive(Debug, Deserialize)]
struct WhisperResult {
    language: String,
}

#[derive(Debug, Deserialize)]
struct WhisperSegment {
    offsets: WhisperOffsets,
    text: String,
}

#[derive(Debug, Deserialize)]
struct WhisperOffsets {
    from: u64,
    to: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeArchive {
    TarGz,
    Zip,
}

#[derive(Clone, Copy, Debug)]
struct RuntimeAsset {
    name: &'static str,
    sha256: &'static str,
    platform: &'static str,
    binary: &'static str,
    archive: RuntimeArchive,
}

/// Inspect the default SenseVoice and VAD caches without accessing the network.
pub fn speech_model_status() -> SpeechModelStatus {
    let cache = Cache::from_env();
    let weights = cached_model_file(&cache, SPEECH_MODEL_ID, SPEECH_MODEL_FILE);
    let vad_weights = cached_model_file(&cache, SPEECH_VAD_MODEL_ID, SPEECH_VAD_MODEL_FILE);
    let runtime_supported =
        runtime_asset_for(std::env::consts::OS, std::env::consts::ARCH).is_some();
    let runtime = sensevoice_runtime_path();
    let models_ready = weights.is_some() && vad_weights.is_some();
    let runtime_ready = runtime.is_some();
    let mut missing_files = Vec::new();
    if weights.is_none() {
        missing_files.push(SPEECH_MODEL_FILE);
    }
    if vad_weights.is_none() {
        missing_files.push(SPEECH_VAD_MODEL_FILE);
    }
    if runtime.is_none() {
        missing_files.push(SENSEVOICE_RUNTIME_BINARY);
    }
    SpeechModelStatus {
        model: SPEECH_MODEL_ID,
        file: SPEECH_MODEL_FILE,
        vad_model: SPEECH_VAD_MODEL_ID,
        vad_file: SPEECH_VAD_MODEL_FILE,
        ready: models_ready && runtime_ready,
        models_ready,
        runtime_ready,
        runtime_supported,
        cache_directory: cache.path().clone(),
        weights_bytes: weights.as_deref().and_then(valid_file_size),
        weights,
        vad_weights_bytes: vad_weights.as_deref().and_then(valid_file_size),
        vad_weights,
        runtime,
        missing_files,
    }
}

pub fn fetch_speech_components() -> Result<SpeechModelFiles> {
    Ok(SpeechModelFiles {
        weights: fetch_model_file(SPEECH_MODEL_ID, SPEECH_MODEL_FILE)?,
        vad_weights: fetch_model_file(SPEECH_VAD_MODEL_ID, SPEECH_VAD_MODEL_FILE)?,
        runtime: fetch_sensevoice_runtime()?,
    })
}

pub fn fetch_sensevoice_model() -> Result<PathBuf> {
    fetch_model_file(SPEECH_MODEL_ID, SPEECH_MODEL_FILE)
}

pub fn fetch_sensevoice_vad_model() -> Result<PathBuf> {
    fetch_model_file(SPEECH_VAD_MODEL_ID, SPEECH_VAD_MODEL_FILE)
}

pub fn whisper_model_status() -> WhisperModelStatus {
    let cache = Cache::from_env();
    let weights = cached_model_file(&cache, WHISPER_MODEL_ID, WHISPER_MODEL_FILE);
    WhisperModelStatus {
        model: WHISPER_MODEL_ID,
        file: WHISPER_MODEL_FILE,
        ready: weights.is_some(),
        cache_directory: cache.path().clone(),
        weights_bytes: weights.as_deref().and_then(valid_file_size),
        weights,
    }
}

pub fn fetch_whisper_model() -> Result<PathBuf> {
    fetch_model_file(WHISPER_MODEL_ID, WHISPER_MODEL_FILE)
}

pub fn sensevoice_runtime_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("VQ_SENSEVOICE_CLI").map(PathBuf::from)
        && valid_file_size(&path).is_some()
    {
        return Some(path);
    }
    if let Some(path) = sensevoice_runtime_cache_path()
        && valid_file_size(&path).is_some()
    {
        return Some(path);
    }
    Command::new(SENSEVOICE_RUNTIME_BINARY)
        .output()
        .ok()
        .map(|_| PathBuf::from(SENSEVOICE_RUNTIME_BINARY))
}

pub fn fetch_sensevoice_runtime() -> Result<PathBuf> {
    if let Some(runtime) = sensevoice_runtime_path() {
        eprintln!(
            "[speech-runtime] status=ready version={SENSEVOICE_RUNTIME_VERSION} path={}",
            runtime.display()
        );
        return Ok(runtime);
    }

    let asset = runtime_asset_for(std::env::consts::OS, std::env::consts::ARCH)
        .with_context(|| {
            format!(
                "no prebuilt SenseVoice runtime for {}-{}; set VQ_SENSEVOICE_CLI to a compatible binary",
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        })?;
    let destination = sensevoice_runtime_cache_path().context("unsupported speech runtime")?;
    let parent = destination
        .parent()
        .context("speech runtime cache path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;

    let url = format!(
        "{RUNTIME_RELEASE_BASE}/{SENSEVOICE_RUNTIME_VERSION}/{}",
        asset.name
    );
    eprintln!(
        "[speech-runtime] download-start version={SENSEVOICE_RUNTIME_VERSION} platform={} url={url}",
        asset.platform
    );
    let response = ureq::get(&url)
        .header(
            "User-Agent",
            concat!("video-sherlock/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let mut reader = response.into_body().into_reader();
    let mut archive_file =
        tempfile::NamedTempFile::new().context("failed to stage runtime archive")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut downloaded = 0_u64;
    loop {
        let count = reader
            .read(&mut buffer)
            .context("failed while downloading SenseVoice runtime")?;
        if count == 0 {
            break;
        }
        archive_file
            .write_all(&buffer[..count])
            .context("failed to stage SenseVoice runtime")?;
        hasher.update(&buffer[..count]);
        downloaded += count as u64;
    }
    let digest = format!("{:x}", hasher.finalize());
    if digest != asset.sha256 {
        bail!(
            "SenseVoice runtime checksum mismatch: expected {}, received {digest}",
            asset.sha256
        )
    }

    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to stage runtime in {}", parent.display()))?;
    let found = match asset.archive {
        RuntimeArchive::TarGz => {
            extract_runtime_tar_gz(archive_file.path(), asset.binary, staged.as_file_mut())?
        }
        RuntimeArchive::Zip => {
            extract_runtime_zip(archive_file.path(), asset.binary, staged.as_file_mut())?
        }
    };
    if !found {
        bail!(
            "{} was not present in the SenseVoice runtime archive",
            asset.binary
        )
    }
    set_executable(staged.path())?;
    if destination.exists() {
        fs::remove_file(&destination)
            .with_context(|| format!("failed to replace {}", destination.display()))?;
    }
    staged
        .persist(&destination)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to install {}", destination.display()))?;
    eprintln!(
        "[speech-runtime] download-complete bytes={downloaded} path={}",
        destination.display()
    );
    Ok(destination)
}

pub fn whisper_cli_available() -> bool {
    Command::new("whisper-cli")
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn recommended_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .clamp(1, 8)
}

pub fn transcribe_sensevoice(options: &SenseVoiceOptions<'_>) -> Result<Transcript> {
    validate_input_and_model(options.input, options.model)?;
    if !options.vad_model.is_file() {
        bail!(
            "speech VAD model does not exist: {}",
            options.vad_model.display()
        )
    }
    if options.runtime.as_os_str() != SENSEVOICE_RUNTIME_BINARY && !options.runtime.is_file() {
        bail!(
            "SenseVoice runtime does not exist: {}",
            options.runtime.display()
        )
    }

    let input = canonicalize_input(options.input)?;
    let model = canonicalize_model(options.model)?;
    let vad_model = options
        .vad_model
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", options.vad_model.display()))?;
    let temporary = tempfile::tempdir().context("failed to create speech work directory")?;
    let wav = temporary.path().join("audio.wav");
    convert_to_pcm_wav(&input, &wav)?;

    let output = Command::new(options.runtime)
        .arg("-m")
        .arg(&model)
        .arg("-a")
        .arg(&wav)
        .arg("--vad")
        .arg(&vad_model)
        .arg("--keep-tags")
        .output()
        .context("failed to launch llama-funasr-sensevoice")?;
    if !output.status.success() {
        bail!(
            "SenseVoice failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
    parse_sensevoice_output(&output.stdout, input, model)
}

pub fn transcribe_whisper(options: &WhisperOptions<'_>) -> Result<Transcript> {
    validate_input_and_model(options.input, options.model)?;
    if options.language.trim().is_empty() {
        bail!("speech language must not be empty")
    }
    if options.threads == 0 {
        bail!("speech recognition thread count must be greater than zero")
    }
    if !whisper_cli_available() {
        bail!(
            "whisper-cli is unavailable; install whisper.cpp (on macOS: `brew install whisper-cpp`)"
        )
    }

    let input = canonicalize_input(options.input)?;
    let model = canonicalize_model(options.model)?;
    let temporary = tempfile::tempdir().context("failed to create speech work directory")?;
    let wav = temporary.path().join("audio.wav");
    convert_to_pcm_wav(&input, &wav)?;

    let output_base = temporary.path().join("transcript");
    let mut command = Command::new("whisper-cli");
    command
        .arg("--model")
        .arg(&model)
        .arg("--file")
        .arg(&wav)
        .args(["--language", options.language, "--threads"])
        .arg(options.threads.to_string());
    if let Some(prompt) = options
        .prompt
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
    {
        command.args(["--prompt", prompt]);
    }
    let output = command
        .args(["--output-json", "--output-file"])
        .arg(&output_base)
        .arg("--no-prints")
        .output()
        .context("failed to launch whisper-cli")?;
    if !output.status.success() {
        bail!(
            "whisper-cli failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }

    let json_path = output_base.with_extension("json");
    let json = fs::read(&json_path)
        .with_context(|| format!("whisper-cli did not produce {}", json_path.display()))?;
    parse_whisper_output(&json, input, model)
}

fn parse_sensevoice_output(json: &[u8], input: PathBuf, model: PathBuf) -> Result<Transcript> {
    let raw = std::str::from_utf8(json).context("SenseVoice emitted non-UTF-8 output")?;
    let (text, tags) = strip_sensevoice_tags(raw);
    let mut languages = Vec::new();
    let mut emotions = Vec::new();
    let mut audio_events = Vec::new();
    for tag in tags {
        let normalized = tag.to_ascii_lowercase();
        match normalized.as_str() {
            "zh" | "yue" | "en" | "ja" | "ko" | "nospeech" => {
                push_unique(&mut languages, normalized)
            }
            "happy" | "sad" | "angry" | "neutral" => push_unique(&mut emotions, normalized),
            "withitn" | "woitn" => {}
            _ => push_unique(&mut audio_events, tag),
        }
    }
    let language = match languages.as_slice() {
        [] => "unknown".to_string(),
        [language] => language.clone(),
        _ => "mixed".to_string(),
    };
    Ok(Transcript {
        engine: "sensevoice".to_string(),
        input,
        model,
        language,
        text,
        segments: Vec::new(),
        emotions,
        audio_events,
    })
}

fn strip_sensevoice_tags(raw: &str) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut tags = Vec::new();
    let mut remaining = raw;
    while let Some(start) = remaining.find("<|") {
        text.push_str(&remaining[..start]);
        let tag_start = start + 2;
        let Some(relative_end) = remaining[tag_start..].find("|>") else {
            text.push_str(&remaining[start..]);
            remaining = "";
            break;
        };
        let tag_end = tag_start + relative_end;
        tags.push(remaining[tag_start..tag_end].to_string());
        remaining = &remaining[tag_end + 2..];
    }
    text.push_str(remaining);
    (text.trim().to_string(), tags)
}

fn parse_whisper_output(json: &[u8], input: PathBuf, model: PathBuf) -> Result<Transcript> {
    let raw: WhisperOutput =
        serde_json::from_slice(json).context("invalid JSON emitted by whisper-cli")?;
    let text = raw
        .transcription
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<String>()
        .trim()
        .to_string();
    let segments = raw
        .transcription
        .into_iter()
        .map(|segment| TranscriptSegment {
            start_seconds: segment.offsets.from as f64 / 1000.0,
            end_seconds: segment.offsets.to as f64 / 1000.0,
            text: segment.text.trim().to_string(),
        })
        .collect();
    Ok(Transcript {
        engine: "whisper".to_string(),
        input,
        model,
        language: raw.result.language,
        text,
        segments,
        emotions: Vec::new(),
        audio_events: Vec::new(),
    })
}

fn fetch_model_file(model_id: &'static str, filename: &'static str) -> Result<PathBuf> {
    let cache = Cache::from_env();
    if let Some(path) = cached_model_file(&cache, model_id, filename) {
        eprintln!(
            "[speech-model] status=ready id={model_id} file={filename} bytes={} path={}",
            valid_file_size(&path).unwrap_or_default(),
            path.display()
        );
        return Ok(path);
    }

    eprintln!("[speech-model] status=missing id={model_id} file={filename} action=download");
    let api = ApiBuilder::from_env()
        .with_progress(false)
        .with_retries(2)
        .with_user_agent("video-sherlock", env!("CARGO_PKG_VERSION"))
        .build()
        .context("failed to initialize the Hugging Face client")?;
    let weights = api
        .model(model_id.to_string())
        .download_with_progress(filename, SpeechDownloadProgress::new(filename))
        .with_context(|| format!("failed to download {model_id}/{filename}"))?;
    let bytes = valid_file_size(&weights)
        .with_context(|| format!("downloaded speech model is empty: {}", weights.display()))?;
    eprintln!(
        "[speech-model] status=ready id={model_id} file={filename} bytes={bytes} path={}",
        weights.display()
    );
    Ok(weights)
}

fn cached_model_file(cache: &Cache, model_id: &str, filename: &str) -> Option<PathBuf> {
    cache
        .model(model_id.to_string())
        .get(filename)
        .filter(|path| valid_file_size(path).is_some())
}

fn validate_input_and_model(input: &Path, model: &Path) -> Result<()> {
    if !input.is_file() {
        bail!("audio or video input does not exist: {}", input.display())
    }
    if !model.is_file() {
        bail!("speech model does not exist: {}", model.display())
    }
    Ok(())
}

fn canonicalize_input(input: &Path) -> Result<PathBuf> {
    input
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", input.display()))
}

fn canonicalize_model(model: &Path) -> Result<PathBuf> {
    model
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", model.display()))
}

fn convert_to_pcm_wav(input: &Path, output: &Path) -> Result<()> {
    let conversion = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
        .arg(input)
        .args([
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
        ])
        .arg(output)
        .output()
        .with_context(|| format!("failed to launch ffmpeg for {}", input.display()))?;
    if !conversion.status.success() {
        bail!(
            "ffmpeg could not extract speech audio from {}: {}",
            input.display(),
            String::from_utf8_lossy(&conversion.stderr).trim()
        )
    }
    Ok(())
}

fn sensevoice_runtime_cache_path() -> Option<PathBuf> {
    let asset = runtime_asset_for(std::env::consts::OS, std::env::consts::ARCH)?;
    let root = std::env::var_os("VQ_RUNTIME_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::home_dir().map(|home| home.join(".cache").join("video-query").join("runtime"))
        })?;
    Some(
        root.join(SENSEVOICE_RUNTIME_VERSION)
            .join(asset.platform)
            .join(asset.binary),
    )
}

fn extract_runtime_tar_gz(
    archive_path: &Path,
    binary: &str,
    output: &mut fs::File,
) -> Result<bool> {
    let archive = fs::File::open(archive_path).context("failed to reopen runtime archive")?;
    let decoder = GzDecoder::new(archive);
    let mut archive = tar::Archive::new(decoder);
    for entry in archive
        .entries()
        .context("invalid SenseVoice runtime archive")?
    {
        let mut entry = entry.context("invalid entry in SenseVoice runtime archive")?;
        let path = entry.path().context("invalid path in runtime archive")?;
        if path.file_name().and_then(|name| name.to_str()) == Some(binary) {
            std::io::copy(&mut entry, output).context("failed to extract SenseVoice runtime")?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn extract_runtime_zip(archive_path: &Path, binary: &str, output: &mut fs::File) -> Result<bool> {
    let archive = fs::File::open(archive_path).context("failed to reopen runtime archive")?;
    let mut archive = zip::ZipArchive::new(archive).context("invalid SenseVoice ZIP archive")?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .context("invalid entry in SenseVoice ZIP archive")?;
        if Path::new(entry.name())
            .file_name()
            .and_then(|name| name.to_str())
            == Some(binary)
        {
            std::io::copy(&mut entry, output).context("failed to extract SenseVoice runtime")?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn runtime_asset_for(os: &str, arch: &str) -> Option<RuntimeAsset> {
    match (os, arch) {
        ("macos", "aarch64") => Some(RuntimeAsset {
            name: "funasr-llamacpp-macos-arm64.tar.gz",
            sha256: "2d5786784ad09d8f4def1d942f678728638fe601d00acf0dad7cf094a9328363",
            platform: "macos-arm64",
            binary: "llama-funasr-sensevoice",
            archive: RuntimeArchive::TarGz,
        }),
        ("linux", "aarch64") => Some(RuntimeAsset {
            name: "funasr-llamacpp-linux-arm64.tar.gz",
            sha256: "521866e75594e56eb5023b65eb1ecf6ab7c3b5069522b71cd33aa37b8406ed4b",
            platform: "linux-arm64",
            binary: "llama-funasr-sensevoice",
            archive: RuntimeArchive::TarGz,
        }),
        ("linux", "x86_64") => Some(RuntimeAsset {
            name: "funasr-llamacpp-linux-x64.tar.gz",
            sha256: "2cd54174a3755f89c11f071dedfb935eff96007617e2e952604d90230ea9eb48",
            platform: "linux-x64",
            binary: "llama-funasr-sensevoice",
            archive: RuntimeArchive::TarGz,
        }),
        ("windows", "x86_64") => Some(RuntimeAsset {
            name: "funasr-llamacpp-windows-x64.zip",
            sha256: "6767af74e42c8b928742e12d5995c139636d9482ea151cdbb51f1b7573667772",
            platform: "windows-x64",
            binary: "llama-funasr-sensevoice.exe",
            archive: RuntimeArchive::Zip,
        }),
        _ => None,
    }
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
        .with_context(|| format!("failed to make {} executable", path.display()))
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn valid_file_size(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() > 0)
        .map(|metadata| metadata.len())
}

struct SpeechDownloadProgress {
    filename: &'static str,
    initialized: bool,
    total: usize,
    downloaded: usize,
    last_reported_bucket: usize,
}

impl SpeechDownloadProgress {
    fn new(filename: &'static str) -> Self {
        Self {
            filename,
            initialized: false,
            total: 0,
            downloaded: 0,
            last_reported_bucket: 0,
        }
    }
}

impl Progress for SpeechDownloadProgress {
    fn init(&mut self, size: usize, filename: &str) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        self.total = size;
        self.downloaded = 0;
        self.last_reported_bucket = 0;
        eprintln!("[speech-model] download-start file={filename} total_bytes={size}");
    }

    fn update(&mut self, size: usize) {
        self.downloaded = self.downloaded.saturating_add(size).min(self.total);
        if self.total == 0 {
            return;
        }
        let percent = self.downloaded.saturating_mul(100) / self.total;
        let bucket = percent / 5;
        if bucket > self.last_reported_bucket {
            self.last_reported_bucket = bucket;
            eprintln!(
                "[speech-model] download-progress file={} percent={percent} downloaded_bytes={} total_bytes={}",
                self.filename, self.downloaded, self.total
            );
        }
    }

    fn finish(&mut self) {
        eprintln!(
            "[speech-model] download-complete file={} bytes={}",
            self.filename, self.total
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_whisper_segments_and_combines_text() {
        let json = r#"{
            "result": { "language": "zh" },
            "transcription": [
                {
                    "timestamps": { "from": "00:00:00,000", "to": "00:00:01,250" },
                    "offsets": { "from": 0, "to": 1250 },
                    "text": " 你好"
                },
                {
                    "timestamps": { "from": "00:00:01,250", "to": "00:00:02,500" },
                    "offsets": { "from": 1250, "to": 2500 },
                    "text": "，世界。"
                }
            ]
        }"#;
        let transcript = parse_whisper_output(
            json.as_bytes(),
            PathBuf::from("input.m4a"),
            PathBuf::from("model.bin"),
        )
        .unwrap();
        assert_eq!(transcript.engine, "whisper");
        assert_eq!(transcript.language, "zh");
        assert_eq!(transcript.text, "你好，世界。");
        assert_eq!(transcript.segments[0].end_seconds, 1.25);
        assert_eq!(transcript.segments[1].text, "，世界。");
    }

    #[test]
    fn parses_sensevoice_text_language_emotion_and_events() {
        let output =
            "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好世界<|zh|><|HAPPY|><|BGM|><|woitn|>欢迎使用\n";
        let transcript = parse_sensevoice_output(
            output.as_bytes(),
            PathBuf::from("input.wav"),
            PathBuf::from("sensevoice.gguf"),
        )
        .unwrap();
        assert_eq!(transcript.engine, "sensevoice");
        assert_eq!(transcript.language, "zh");
        assert_eq!(transcript.text, "你好世界欢迎使用");
        assert_eq!(transcript.emotions, ["neutral", "happy"]);
        assert_eq!(transcript.audio_events, ["Speech", "BGM"]);
        assert!(transcript.segments.is_empty());
    }

    #[test]
    fn recognizes_mixed_sensevoice_languages() {
        let output =
            "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好<|en|><|NEUTRAL|><|Speech|><|woitn|>hello";
        let transcript = parse_sensevoice_output(
            output.as_bytes(),
            PathBuf::from("input.wav"),
            PathBuf::from("sensevoice.gguf"),
        )
        .unwrap();
        assert_eq!(transcript.language, "mixed");
        assert_eq!(transcript.audio_events, ["Speech"]);
    }

    #[test]
    fn runtime_assets_are_pinned_for_supported_targets() {
        assert_eq!(
            runtime_asset_for("macos", "aarch64").unwrap().name,
            "funasr-llamacpp-macos-arm64.tar.gz"
        );
        assert_eq!(
            runtime_asset_for("linux", "x86_64").unwrap().platform,
            "linux-x64"
        );
        let windows = runtime_asset_for("windows", "x86_64").unwrap();
        assert_eq!(windows.binary, "llama-funasr-sensevoice.exe");
        assert_eq!(windows.archive, RuntimeArchive::Zip);
        assert!(runtime_asset_for("macos", "x86_64").is_none());
    }

    #[test]
    fn extracts_windows_runtime_from_zip() {
        use std::io::Seek;

        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("runtime.zip");
        let archive = fs::File::create(&archive_path).unwrap();
        let mut writer = zip::ZipWriter::new(archive);
        writer
            .start_file(
                "nested/llama-funasr-sensevoice.exe",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
        writer.write_all(b"windows runtime").unwrap();
        writer.finish().unwrap();

        let mut output = tempfile::tempfile().unwrap();
        assert!(
            extract_runtime_zip(&archive_path, "llama-funasr-sensevoice.exe", &mut output,)
                .unwrap()
        );
        output.rewind().unwrap();
        let mut contents = String::new();
        output.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "windows runtime");
    }

    #[test]
    fn recommended_threads_is_bounded() {
        assert!((1..=8).contains(&recommended_thread_count()));
    }
}
