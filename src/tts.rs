//! Local text-to-speech orchestration through MLX-Audio on Apple silicon.

use anyhow::{Context, Result, bail};
use hf_hub::Cache;
use serde::Serialize;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

pub const QWEN_TTS_MODEL_ID: &str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit";
pub const MLX_AUDIO_VERSION: &str = "0.4.7";

#[derive(Clone, Debug)]
pub struct QwenTtsOptions<'a> {
    pub text: &'a str,
    pub model: &'a str,
    pub voice: &'a str,
    pub language: &'a str,
    pub output: &'a Path,
    pub reference_audio: Option<&'a Path>,
    pub reference_text: Option<&'a str>,
    pub speed: f32,
    pub play: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SynthesisOutput {
    pub engine: String,
    pub model: String,
    pub voice: String,
    pub language: String,
    pub text: String,
    pub output: PathBuf,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct QwenTtsModelStatus {
    pub model: &'static str,
    pub ready: bool,
    pub platform_supported: bool,
    pub runtime_available: bool,
    pub cache_directory: PathBuf,
    pub weights: Option<PathBuf>,
    pub weights_bytes: Option<u64>,
    pub tokenizer_weights: Option<PathBuf>,
    pub tokenizer_weights_bytes: Option<u64>,
}

pub fn tts_platform_supported() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

pub fn uv_available() -> bool {
    Command::new(uv_binary())
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

pub fn qwen_tts_model_status() -> QwenTtsModelStatus {
    let cache = Cache::from_env();
    let cache_directory = cache.path().clone();
    let repository = cache.model(QWEN_TTS_MODEL_ID.to_string());
    let weights = repository
        .get("model.safetensors")
        .filter(|path| valid_file_size(path).is_some());
    let tokenizer_weights = repository
        .get("speech_tokenizer/model.safetensors")
        .filter(|path| valid_file_size(path).is_some());
    QwenTtsModelStatus {
        model: QWEN_TTS_MODEL_ID,
        ready: weights.is_some() && tokenizer_weights.is_some(),
        platform_supported: tts_platform_supported(),
        runtime_available: uv_available(),
        cache_directory,
        weights_bytes: weights.as_deref().and_then(valid_file_size),
        weights,
        tokenizer_weights_bytes: tokenizer_weights.as_deref().and_then(valid_file_size),
        tokenizer_weights,
    }
}

pub fn qwen_tts_model_cached() -> bool {
    qwen_tts_model_status().ready
}

pub fn fetch_qwen_tts_model() -> Result<QwenTtsModelStatus> {
    if !tts_platform_supported() {
        bail!(
            "Qwen TTS through MLX-Audio requires macOS on Apple silicon; current platform is {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    }
    if !uv_available() {
        bail!("uv is unavailable; install it from https://docs.astral.sh/uv/")
    }
    let status = qwen_tts_model_status();
    if status.ready {
        eprintln!("[tts-model] status=ready id={QWEN_TTS_MODEL_ID}");
        return Ok(status);
    }

    eprintln!("[tts-model] status=missing id={QWEN_TTS_MODEL_ID} action=download");
    let package = format!("mlx-audio=={MLX_AUDIO_VERSION}");
    let script = concat!(
        "import os\n",
        "from huggingface_hub import snapshot_download\n",
        "snapshot_download(os.environ['VQ_TTS_FETCH_MODEL'])\n",
    );
    let mut command = Command::new(uv_binary());
    command
        .args(["tool", "run", "--python", "3.12", "--from"])
        .arg(package)
        .arg("python")
        .args(["-c", script])
        .env("VQ_TTS_FETCH_MODEL", QWEN_TTS_MODEL_ID);
    let exit = run_with_stdout_on_stderr(&mut command)?;
    if !exit.success() {
        bail!("Qwen TTS model download failed with status {exit}")
    }
    let status = qwen_tts_model_status();
    if !status.ready {
        bail!("Qwen TTS model download completed but its cache is incomplete")
    }
    eprintln!("[tts-model] status=ready id={QWEN_TTS_MODEL_ID}");
    Ok(status)
}

pub fn synthesize_qwen(options: &QwenTtsOptions<'_>) -> Result<SynthesisOutput> {
    validate_options(options)?;
    if !tts_platform_supported() {
        bail!(
            "Qwen TTS through MLX-Audio requires macOS on Apple silicon; current platform is {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    }
    if !uv_available() {
        bail!("uv is unavailable; install it from https://docs.astral.sh/uv/")
    }

    let output = prepare_output_path(options.output)?;
    let output_directory = output
        .parent()
        .context("TTS output path has no parent directory")?;
    let file_prefix = output
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .context("TTS output filename must have a non-empty stem")?;

    eprintln!(
        "[tts] action=synthesize engine=mlx-audio model={} language={} voice={} output={}",
        options.model,
        options.language,
        options.voice,
        output.display()
    );

    let package = format!("mlx-audio=={MLX_AUDIO_VERSION}");
    let mut command = Command::new(uv_binary());
    command
        .args(["tool", "run", "--python", "3.12", "--from"])
        .arg(package)
        .arg("mlx_audio.tts.generate")
        .arg("--model")
        .arg(options.model)
        .arg("--text")
        .arg(options.text.trim())
        .arg("--lang_code")
        .arg(options.language.trim())
        .arg("--speed")
        .arg(options.speed.to_string())
        .arg("--output_path")
        .arg(output_directory)
        .arg("--file_prefix")
        .arg(file_prefix)
        .args(["--audio_format", "wav", "--join_audio"]);

    match (options.reference_audio, options.reference_text) {
        (Some(reference_audio), Some(reference_text)) => {
            let reference_audio = reference_audio.canonicalize().with_context(|| {
                format!(
                    "failed to resolve reference audio {}",
                    reference_audio.display()
                )
            })?;
            command
                .arg("--ref_audio")
                .arg(reference_audio)
                .arg("--ref_text")
                .arg(reference_text.trim());
        }
        (None, None) => {
            command.arg("--voice").arg(options.voice.trim());
        }
        _ => unreachable!("reference audio and text are validated as a pair"),
    }

    let status = run_with_stdout_on_stderr(&mut command)?;
    if !status.success() {
        bail!("MLX-Audio TTS failed with status {status}")
    }

    let bytes = fs::metadata(&output)
        .with_context(|| format!("MLX-Audio did not produce {}", output.display()))?
        .len();
    if bytes == 0 {
        bail!("MLX-Audio produced an empty file at {}", output.display())
    }

    if options.play {
        play_audio(&output)?;
    }

    Ok(SynthesisOutput {
        engine: "mlx-audio".to_string(),
        model: options.model.to_string(),
        voice: options.voice.to_string(),
        language: options.language.to_string(),
        text: options.text.trim().to_string(),
        output,
        bytes,
    })
}

fn run_with_stdout_on_stderr(command: &mut Command) -> Result<std::process::ExitStatus> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to launch MLX-Audio through uv")?;
    let mut stdout = child
        .stdout
        .take()
        .context("failed to capture MLX-Audio output")?;
    let relay = thread::spawn(move || -> io::Result<u64> {
        let mut stderr = io::stderr().lock();
        io::copy(&mut stdout, &mut stderr)
    });
    let status = child.wait().context("failed to wait for MLX-Audio")?;
    relay
        .join()
        .map_err(|_| anyhow::anyhow!("MLX-Audio output relay panicked"))?
        .context("failed to relay MLX-Audio output")?;
    Ok(status)
}

fn validate_options(options: &QwenTtsOptions<'_>) -> Result<()> {
    if options.text.trim().is_empty() {
        bail!("TTS text must not be empty")
    }
    if options.model.trim().is_empty() {
        bail!("TTS model must not be empty")
    }
    if options.voice.trim().is_empty() {
        bail!("TTS voice must not be empty")
    }
    if options.language.trim().is_empty() {
        bail!("TTS language must not be empty")
    }
    if !options.speed.is_finite() || options.speed <= 0.0 {
        bail!("TTS speed must be a finite number greater than zero")
    }
    match (options.reference_audio, options.reference_text) {
        (Some(audio), Some(text)) => {
            if !audio.is_file() {
                bail!("reference audio does not exist: {}", audio.display())
            }
            if text.trim().is_empty() {
                bail!("reference text must not be empty")
            }
        }
        (None, None) => {}
        _ => bail!("--reference-audio and --reference-text must be supplied together"),
    }
    Ok(())
}

fn prepare_output_path(output: &Path) -> Result<PathBuf> {
    if output.extension().and_then(OsStr::to_str) != Some("wav") {
        bail!("TTS output must use the .wav extension")
    }
    let current_directory =
        std::env::current_dir().context("failed to resolve current directory")?;
    let absolute = if output.is_absolute() {
        output.to_path_buf()
    } else {
        current_directory.join(output)
    };
    let parent = absolute
        .parent()
        .context("TTS output path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let canonical_parent = parent
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", parent.display()))?;
    let filename = absolute
        .file_name()
        .context("TTS output path must include a filename")?;
    Ok(canonical_parent.join(filename))
}

fn play_audio(output: &Path) -> Result<()> {
    let status = Command::new("afplay")
        .arg(output)
        .status()
        .context("failed to launch afplay")?;
    if !status.success() {
        bail!("afplay failed with status {status}")
    }
    Ok(())
}

fn uv_binary() -> OsString {
    std::env::var_os("VQ_UV_BINARY").unwrap_or_else(|| OsString::from("uv"))
}

fn valid_file_size(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() > 0)
        .map(|metadata| metadata.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options<'a>(output: &'a Path) -> QwenTtsOptions<'a> {
        QwenTtsOptions {
            text: "你好",
            model: QWEN_TTS_MODEL_ID,
            voice: "Vivian",
            language: "Chinese",
            output,
            reference_audio: None,
            reference_text: None,
            speed: 1.0,
            play: false,
        }
    }

    #[test]
    fn validates_non_empty_text_and_positive_speed() {
        let mut value = options(Path::new("speech.wav"));
        value.text = "  ";
        assert_eq!(
            validate_options(&value).unwrap_err().to_string(),
            "TTS text must not be empty"
        );

        value.text = "hello";
        value.speed = 0.0;
        assert_eq!(
            validate_options(&value).unwrap_err().to_string(),
            "TTS speed must be a finite number greater than zero"
        );
    }

    #[test]
    fn requires_reference_audio_and_text_as_a_pair() {
        let mut value = options(Path::new("speech.wav"));
        value.reference_text = Some("reference transcript");
        assert_eq!(
            validate_options(&value).unwrap_err().to_string(),
            "--reference-audio and --reference-text must be supplied together"
        );
    }

    #[test]
    fn prepares_an_absolute_wav_output_path() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("nested").join("speech.wav");
        let prepared = prepare_output_path(&output).unwrap();
        assert!(prepared.is_absolute());
        assert_eq!(prepared.file_name(), Some(OsStr::new("speech.wav")));
        assert!(prepared.parent().unwrap().is_dir());
    }

    #[test]
    fn rejects_non_wav_output() {
        let error = prepare_output_path(Path::new("speech.mp3")).unwrap_err();
        assert_eq!(error.to_string(), "TTS output must use the .wav extension");
    }
}
