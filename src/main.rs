use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use video_sherlock::embedding::{ChineseClipEmbedder, MODEL_ID, fetch_model_files, model_status};
use video_sherlock::index::{IndexedFrame, IndexedVideo, VideoIndex, video_fingerprint};
use video_sherlock::keyframe::KeyframeCandidate;
use video_sherlock::speech::{
    SENSEVOICE_RUNTIME_VERSION, SPEECH_MODEL_FILE, SPEECH_MODEL_ID, SPEECH_VAD_MODEL_FILE,
    SPEECH_VAD_MODEL_ID, SenseVoiceOptions, Transcript, WHISPER_MODEL_FILE, WHISPER_MODEL_ID,
    WhisperOptions, fetch_sensevoice_model, fetch_sensevoice_runtime, fetch_sensevoice_vad_model,
    fetch_speech_components, fetch_whisper_model, recommended_thread_count,
    sensevoice_runtime_path, speech_model_status, transcribe_sensevoice, transcribe_whisper,
    whisper_cli_available, whisper_model_status,
};
use video_sherlock::tts::{
    QWEN_TTS_MODEL_ID, QwenTtsOptions, fetch_qwen_tts_model, qwen_tts_model_status,
    synthesize_qwen, tts_platform_supported, uv_available,
};
use video_sherlock::video::{
    ScanOptions, best_near, best_per_segment, collect_videos, ensure_ffmpeg_available,
    extract_jpeg, probe, scan_quality,
};

#[derive(Parser, Debug)]
#[command(
    name = "vq",
    version,
    about = "Evidence-backed video understanding for coding agents",
    long_about = "Power Video Sherlock with local speech transcription, high-quality keyframe extraction, Chinese/English semantic frame search, and auditable evidence. Model inference is local and does not require an API key."
)]
struct Cli {
    /// Directory containing the SQLite index and extracted keyframes (default: ~/.video-query).
    #[arg(
        long,
        global = true,
        env = "VQ_INDEX_DIR",
        default_value_os_t = default_index_dir()
    )]
    index_dir: PathBuf,

    /// Emit machine-readable JSON where supported.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Analyze or extract high-quality keyframes without loading an ML model.
    Keyframes(KeyframesArgs),
    /// Index one or more videos/directories for semantic search.
    Index(IndexArgs),
    /// Search indexed frames using a Chinese or English description.
    Search(SearchArgs),
    /// Transcribe speech and detect audio events from an audio or video file locally.
    #[command(visible_alias = "audio-to-text")]
    Transcribe(TranscribeArgs),
    /// Generate local speech from text with Qwen3-TTS on Apple silicon.
    #[command(visible_alias = "text-to-speech")]
    Speak(SpeakArgs),
    /// Download or inspect local embedding and transcription models.
    Model(ModelArgs),
    /// Print index statistics.
    Stats,
    /// Check runtime dependencies.
    Doctor,
}

#[derive(Args, Debug)]
struct KeyframesArgs {
    video: PathBuf,

    /// Find one frame near this timestamp instead of scanning fixed segments.
    #[arg(long)]
    at: Option<f64>,

    /// Search radius around --at, in seconds.
    #[arg(long, default_value_t = 3.0)]
    radius: f64,

    /// Frames per second analyzed by Rust.
    #[arg(long, default_value_t = 8.0)]
    scan_fps: f64,

    /// Select one best frame per segment of this many seconds.
    #[arg(long, default_value_t = 10.0)]
    segment_seconds: f64,

    /// Extract selected frames as JPEG files into this directory.
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct IndexArgs {
    /// Video files or directories searched recursively.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Select one keyframe per segment of this many seconds.
    #[arg(long, default_value_t = 10.0)]
    segment_seconds: f64,

    /// Frames per second evaluated during the quality scan.
    #[arg(long, default_value_t = 4.0)]
    scan_fps: f64,

    /// Chinese-CLIP image inference batch size.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Re-index videos even when their fingerprint has not changed.
    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug)]
struct SearchArgs {
    /// Natural-language image description, for example: "海边奔跑的一只狗".
    query: String,

    #[arg(short, long, default_value_t = 10)]
    limit: usize,
}

#[derive(Args, Debug)]
struct TranscribeArgs {
    /// Audio or video file containing speech.
    input: PathBuf,

    /// Recognition engine. Defaults to SenseVoice where its native runtime is supported, otherwise Whisper.
    #[arg(long, value_enum)]
    engine: Option<SpeechEngine>,

    /// Expected language code. SenseVoice detects it; Whisper uses it as a decoding constraint.
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Custom model for the selected engine. Defaults to the recommended local model.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Custom FSMN-VAD GGUF for SenseVoice.
    #[arg(long)]
    vad_model: Option<PathBuf>,

    /// CPU threads used by Whisper (SenseVoice currently uses eight internally).
    #[arg(short, long)]
    threads: Option<usize>,

    /// Vocabulary, names, or style supplied to Whisper as initial context (Whisper only).
    #[arg(long)]
    prompt: Option<String>,

    /// Include segment timestamps (Whisper only).
    #[arg(long)]
    timestamps: bool,

    /// Include detected language, emotions, and audio events in text output.
    #[arg(long)]
    metadata: bool,

    /// Also save the emitted text or JSON to this file.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct SpeakArgs {
    /// Text to synthesize.
    #[arg(required_unless_present = "text_file", conflicts_with = "text_file")]
    text: Option<String>,

    /// Read text to synthesize from a UTF-8 file.
    #[arg(long, conflicts_with = "text")]
    text_file: Option<PathBuf>,

    /// MLX-Audio model repository or local model path.
    #[arg(long, default_value = QWEN_TTS_MODEL_ID)]
    model: String,

    /// Built-in Qwen voice used when no reference audio is supplied.
    #[arg(long, default_value = "Vivian")]
    voice: String,

    /// Language hint passed to Qwen3-TTS.
    #[arg(short, long, default_value = "Chinese")]
    language: String,

    /// Generated 24 kHz PCM WAV file.
    #[arg(short, long, default_value = "speech.wav")]
    output: PathBuf,

    /// Audio sample whose voice should be cloned.
    #[arg(long, requires = "reference_text")]
    reference_audio: Option<PathBuf>,

    /// Exact transcript of --reference-audio.
    #[arg(long, requires = "reference_audio")]
    reference_text: Option<String>,

    /// Speech-rate multiplier.
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Play the generated WAV with macOS afplay.
    #[arg(long)]
    play: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SpeechEngine {
    Sensevoice,
    Whisper,
}

#[derive(Args, Debug)]
struct ModelArgs {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    /// Check every image and transcription model cache without using the network.
    Status,
    /// Download all missing image and transcription models/runtimes with progress output.
    Fetch,
    /// Check the Chinese-CLIP embedding model cache without using the network.
    StatusEmbedding,
    /// Download the Chinese-CLIP embedding model used by index and search.
    FetchEmbedding,
    /// Check the speech model cache without using the network.
    StatusSpeech,
    /// Download the recommended speech-to-text model with progress output.
    FetchSpeech,
    /// Check the optional Whisper model cache without using the network.
    StatusWhisper,
    /// Download the optional Whisper model used for timestamps and broader language support.
    FetchWhisper,
    /// Check the optional Qwen3-TTS model cache without using the network.
    StatusTts,
    /// Download the optional Qwen3-TTS model and its pinned MLX-Audio runtime.
    FetchTts,
}

fn default_index_dir() -> PathBuf {
    std::env::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".video-query")
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Keyframes(arguments) => command_keyframes(arguments, cli.json),
        Command::Index(arguments) => command_index(arguments, &cli.index_dir, cli.json),
        Command::Search(arguments) => command_search(arguments, &cli.index_dir, cli.json),
        Command::Transcribe(arguments) => command_transcribe(arguments, cli.json),
        Command::Speak(arguments) => command_speak(arguments, cli.json),
        Command::Model(arguments) => command_model(arguments, cli.json),
        Command::Stats => command_stats(&cli.index_dir, cli.json),
        Command::Doctor => command_doctor(cli.json),
    }
}

fn command_speak(arguments: SpeakArgs, json: bool) -> Result<()> {
    let text = match (arguments.text, arguments.text_file) {
        (Some(text), None) => text,
        (None, Some(path)) => fs::read_to_string(&path)
            .with_context(|| format!("failed to read TTS text from {}", path.display()))?,
        _ => unreachable!("clap requires exactly one TTS text source"),
    };
    let result = synthesize_qwen(&QwenTtsOptions {
        text: &text,
        model: &arguments.model,
        voice: &arguments.voice,
        language: &arguments.language,
        output: &arguments.output,
        reference_audio: arguments.reference_audio.as_deref(),
        reference_text: arguments.reference_text.as_deref(),
        speed: arguments.speed,
        play: arguments.play,
    })?;
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("model:  {}", result.model);
        println!("voice:  {}", result.voice);
        println!(
            "output: {} ({} bytes)",
            result.output.display(),
            result.bytes
        );
    }
    Ok(())
}

fn command_keyframes(arguments: KeyframesArgs, json: bool) -> Result<()> {
    ensure_ffmpeg_available()?;
    let video = arguments
        .video
        .canonicalize()
        .with_context(|| format!("video does not exist: {}", arguments.video.display()))?;
    let mut selected = if let Some(at) = arguments.at {
        vec![best_near(&video, at, arguments.radius, arguments.scan_fps)?]
    } else {
        let candidates = scan_quality(
            &video,
            ScanOptions {
                frames_per_second: arguments.scan_fps,
                ..ScanOptions::default()
            },
        )?;
        best_per_segment(candidates, arguments.segment_seconds)?
    };

    if let Some(output_dir) = &arguments.output_dir {
        fs::create_dir_all(output_dir)?;
        for candidate in &selected {
            let output = output_dir.join(frame_filename(candidate.timestamp_seconds));
            extract_jpeg(&video, candidate.timestamp_seconds, &output, Some(1280))?;
        }
    }
    selected.sort_by(|left, right| left.timestamp_seconds.total_cmp(&right.timestamp_seconds));
    print_candidates(&selected, json)
}

fn command_index(arguments: IndexArgs, index_dir: &Path, json: bool) -> Result<()> {
    ensure_ffmpeg_available()?;
    if arguments.batch_size == 0 {
        anyhow::bail!("batch size must be greater than zero")
    }
    let videos = collect_videos(&arguments.inputs)?;
    if videos.is_empty() {
        anyhow::bail!("no supported video files were found")
    }

    let mut index = VideoIndex::open(index_dir)?;
    let mut pending = Vec::new();
    for video in videos {
        let fingerprint = video_fingerprint(&video)?;
        if !arguments.force && index.contains_current(&video, &fingerprint)? {
            eprintln!("skip unchanged: {}", video.display());
        } else {
            pending.push((video, fingerprint));
        }
    }
    if pending.is_empty() {
        eprintln!("index is already current");
        return command_stats(index_dir, json);
    }

    eprintln!("[model] action=load purpose=image-indexing");
    let embedder = ChineseClipEmbedder::load()?;
    let progress = ProgressBar::new(pending.len() as u64);
    progress.set_style(
        ProgressStyle::with_template("{spinner:.cyan} [{bar:32.cyan/blue}] {pos}/{len} {msg}")?
            .progress_chars("=>-"),
    );

    for (video, fingerprint) in pending {
        progress.set_message(video.display().to_string());
        index_one_video(
            &mut index,
            &embedder,
            &video,
            &fingerprint,
            arguments.segment_seconds,
            arguments.scan_fps,
            arguments.batch_size,
        )?;
        progress.inc(1);
    }
    progress.finish_with_message("index complete");
    command_stats(index_dir, json)
}

fn index_one_video(
    index: &mut VideoIndex,
    embedder: &ChineseClipEmbedder,
    video: &Path,
    fingerprint: &str,
    segment_seconds: f64,
    scan_fps: f64,
    batch_size: usize,
) -> Result<()> {
    let info = probe(video)?;
    let candidates = scan_quality(
        video,
        ScanOptions {
            frames_per_second: scan_fps,
            ..ScanOptions::default()
        },
    )?;
    let selected = best_per_segment(candidates, segment_seconds)?;
    if selected.is_empty() {
        anyhow::bail!("no keyframes could be decoded from {}", video.display())
    }

    let frames_directory = index.frames_directory(fingerprint);
    fs::create_dir_all(&frames_directory)?;
    let image_paths: Vec<PathBuf> = selected
        .iter()
        .map(|candidate| frames_directory.join(frame_filename(candidate.timestamp_seconds)))
        .collect();
    for (candidate, image_path) in selected.iter().zip(&image_paths) {
        extract_jpeg(video, candidate.timestamp_seconds, image_path, Some(1280))?;
    }

    let mut embeddings = Vec::with_capacity(image_paths.len());
    for batch in image_paths.chunks(batch_size) {
        embeddings.extend(embedder.embed_images(batch)?);
    }
    let frames = selected
        .into_iter()
        .zip(image_paths)
        .zip(embeddings)
        .map(|((candidate, image_path), embedding)| IndexedFrame {
            timestamp_seconds: candidate.timestamp_seconds,
            quality: candidate.quality,
            image_path,
            embedding,
        })
        .collect();
    index.replace_video(&IndexedVideo {
        path: video.to_path_buf(),
        fingerprint: fingerprint.to_string(),
        duration_seconds: info.duration_seconds,
        width: info.width,
        height: info.height,
        frames,
    })
}

fn command_search(arguments: SearchArgs, index_dir: &Path, json: bool) -> Result<()> {
    let index = VideoIndex::open(index_dir)?;
    let stats = index.stats()?;
    if stats.frames == 0 {
        anyhow::bail!("the index contains no frames; run `vq index <video-or-directory>` first")
    }
    eprintln!("[model] action=load purpose=text-search");
    let embedder = ChineseClipEmbedder::load()?;
    let query = embedder.embed_text(&arguments.query)?;
    let matches = index.search(&query, arguments.limit)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&matches)?);
    } else {
        for (rank, result) in matches.iter().enumerate() {
            println!(
                "{:>2}. {:>7.3}  {:>9}  {}\n    {}",
                rank + 1,
                result.similarity,
                format_timestamp(result.timestamp_seconds),
                result.video_path.display(),
                result.frame_path.display()
            );
        }
    }
    Ok(())
}

fn command_transcribe(arguments: TranscribeArgs, json: bool) -> Result<()> {
    ensure_ffmpeg_available()?;
    let engine = arguments.engine.unwrap_or_else(default_speech_engine);
    let transcript = match engine {
        SpeechEngine::Sensevoice => {
            if arguments.timestamps {
                anyhow::bail!("--timestamps requires --engine whisper")
            }
            if arguments.threads.is_some() {
                anyhow::bail!("--threads is only configurable with --engine whisper")
            }
            if arguments.prompt.is_some() {
                anyhow::bail!("--prompt requires --engine whisper")
            }
            let model = match arguments.model {
                Some(path) => path,
                None => fetch_sensevoice_model()?,
            };
            let vad_model = match arguments.vad_model {
                Some(path) => path,
                None => fetch_sensevoice_vad_model()?,
            };
            let runtime = match sensevoice_runtime_path() {
                Some(path) => path,
                None => fetch_sensevoice_runtime()?,
            };
            eprintln!(
                "[speech] action=transcribe engine=sensevoice input={} expected_language={}",
                arguments.input.display(),
                arguments.language
            );
            let transcript = transcribe_sensevoice(&SenseVoiceOptions {
                input: &arguments.input,
                model: &model,
                vad_model: &vad_model,
                runtime: &runtime,
            })?;
            if arguments.language != "auto"
                && transcript.language != arguments.language
                && transcript.language != "mixed"
            {
                eprintln!(
                    "[speech] warning=language-mismatch expected={} detected={}",
                    arguments.language, transcript.language
                );
            }
            transcript
        }
        SpeechEngine::Whisper => {
            if arguments.vad_model.is_some() {
                anyhow::bail!("--vad-model is only supported with --engine sensevoice")
            }
            let model = match arguments.model {
                Some(path) => path,
                None => fetch_whisper_model()?,
            };
            let threads = arguments.threads.unwrap_or_else(recommended_thread_count);
            eprintln!(
                "[speech] action=transcribe engine=whisper input={} language={} threads={threads}",
                arguments.input.display(),
                arguments.language
            );
            transcribe_whisper(&WhisperOptions {
                input: &arguments.input,
                model: &model,
                language: &arguments.language,
                prompt: arguments.prompt.as_deref(),
                threads,
            })?
        }
    };
    let rendered = render_transcript(&transcript, json, arguments.timestamps, arguments.metadata)?;
    if let Some(output) = arguments.output {
        if let Some(parent) = output.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(&output, &rendered)
            .with_context(|| format!("failed to write {}", output.display()))?;
        eprintln!("[speech] action=save path={}", output.display());
    }
    println!("{rendered}");
    Ok(())
}

fn default_speech_engine() -> SpeechEngine {
    let status = speech_model_status();
    if status.runtime_supported || status.runtime_ready {
        SpeechEngine::Sensevoice
    } else {
        SpeechEngine::Whisper
    }
}

fn render_transcript(
    transcript: &Transcript,
    json: bool,
    timestamps: bool,
    metadata: bool,
) -> Result<String> {
    if json {
        return Ok(serde_json::to_string_pretty(transcript)?);
    }
    if timestamps {
        return Ok(transcript
            .segments
            .iter()
            .map(|segment| {
                format!(
                    "{} --> {}  {}",
                    format_timestamp(segment.start_seconds),
                    format_timestamp(segment.end_seconds),
                    segment.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n"));
    }
    if metadata {
        let emotions = if transcript.emotions.is_empty() {
            "unknown".to_string()
        } else {
            transcript.emotions.join(", ")
        };
        let events = if transcript.audio_events.is_empty() {
            "unknown".to_string()
        } else {
            transcript.audio_events.join(", ")
        };
        return Ok(format!(
            "engine: {}\nlanguage: {}\nemotions: {emotions}\naudio events: {events}\n\n{}",
            transcript.engine, transcript.language, transcript.text
        ));
    }
    Ok(transcript.text.clone())
}

fn command_model(arguments: ModelArgs, json: bool) -> Result<()> {
    match arguments.command {
        ModelCommand::Status => {
            let embedding = model_status();
            let speech = speech_model_status();
            let whisper = whisper_model_status();
            let sensevoice_required = speech.runtime_supported || speech.runtime_ready;
            let ready = embedding.ready && whisper.ready && (!sensevoice_required || speech.ready);
            if json {
                #[derive(Serialize)]
                struct Output<'a> {
                    ready: bool,
                    embedding: &'a video_sherlock::embedding::ModelStatus,
                    speech: &'a video_sherlock::speech::SpeechModelStatus,
                    whisper: &'a video_sherlock::speech::WhisperModelStatus,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Output {
                        ready,
                        embedding: &embedding,
                        speech: &speech,
                        whisper: &whisper,
                    })?
                );
            } else {
                println!("status: {}", if ready { "ready" } else { "missing" });
                println!("cache:  {}", embedding.cache_directory.display());
                println!(
                    "image:  {} ({})",
                    embedding.model,
                    if embedding.ready { "ready" } else { "missing" }
                );
                println!(
                    "speech: {} ({})",
                    speech.model,
                    if speech.ready {
                        "ready"
                    } else if sensevoice_required {
                        "missing"
                    } else {
                        "unsupported on this platform; Whisper is the default"
                    }
                );
                println!(
                    "whisper: {}/{} ({})",
                    whisper.model,
                    whisper.file,
                    if whisper.ready { "ready" } else { "missing" }
                );
                if !ready {
                    println!("next:    vq model fetch");
                }
            }
            Ok(())
        }
        ModelCommand::Fetch => {
            let embedding = fetch_model_files()?;
            let speech_status = speech_model_status();
            let speech = if speech_status.runtime_supported || speech_status.runtime_ready {
                Some(fetch_speech_components()?)
            } else {
                eprintln!(
                    "[speech-runtime] status=unsupported platform={}-{} action=skip engine=SenseVoice",
                    std::env::consts::OS,
                    std::env::consts::ARCH
                );
                None
            };
            let whisper = fetch_whisper_model()?;
            if json {
                #[derive(Serialize)]
                struct EmbeddingOutput<'a> {
                    model: &'a str,
                    weights: &'a Path,
                    vocabulary: &'a Path,
                }
                #[derive(Serialize)]
                struct SpeechOutput<'a> {
                    model: &'a str,
                    weights: &'a Path,
                    vad_model: &'a str,
                    vad_weights: &'a Path,
                    runtime_version: &'a str,
                    runtime: &'a Path,
                }
                #[derive(Serialize)]
                struct WhisperOutput<'a> {
                    model: &'a str,
                    file: &'a str,
                    weights: &'a Path,
                }
                #[derive(Serialize)]
                struct Output<'a> {
                    embedding: EmbeddingOutput<'a>,
                    speech: Option<SpeechOutput<'a>>,
                    whisper: WhisperOutput<'a>,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Output {
                        embedding: EmbeddingOutput {
                            model: MODEL_ID,
                            weights: &embedding.weights,
                            vocabulary: &embedding.vocabulary,
                        },
                        speech: speech.as_ref().map(|speech| SpeechOutput {
                            model: SPEECH_MODEL_ID,
                            weights: &speech.weights,
                            vad_model: SPEECH_VAD_MODEL_ID,
                            vad_weights: &speech.vad_weights,
                            runtime_version: SENSEVOICE_RUNTIME_VERSION,
                            runtime: &speech.runtime,
                        }),
                        whisper: WhisperOutput {
                            model: WHISPER_MODEL_ID,
                            file: WHISPER_MODEL_FILE,
                            weights: &whisper,
                        },
                    })?
                );
            } else {
                println!("image model:      {MODEL_ID}");
                println!("image weights:    {}", embedding.weights.display());
                println!("image vocabulary: {}", embedding.vocabulary.display());
                if let Some(speech) = speech {
                    println!("speech model:     {SPEECH_MODEL_ID}");
                    println!("speech weights:   {}", speech.weights.display());
                    println!("speech VAD:       {}", speech.vad_weights.display());
                    println!("speech runtime:   {}", speech.runtime.display());
                } else {
                    println!(
                        "speech runtime:   unsupported on {}-{}; using Whisper",
                        std::env::consts::OS,
                        std::env::consts::ARCH
                    );
                }
                println!("whisper model:    {WHISPER_MODEL_ID}/{WHISPER_MODEL_FILE}");
                println!("whisper weights:  {}", whisper.display());
            }
            Ok(())
        }
        ModelCommand::StatusEmbedding => {
            let status = model_status();
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!("model:  {}", status.model);
                println!("status: {}", if status.ready { "ready" } else { "missing" });
                println!("cache:  {}", status.cache_directory.display());
                if let Some(path) = status.weights {
                    println!(
                        "weights: {} ({} bytes)",
                        path.display(),
                        status.weights_bytes.unwrap_or_default()
                    );
                }
                if let Some(path) = status.vocabulary {
                    println!(
                        "vocabulary: {} ({} bytes)",
                        path.display(),
                        status.vocabulary_bytes.unwrap_or_default()
                    );
                }
                if !status.missing_files.is_empty() {
                    println!("missing: {}", status.missing_files.join(", "));
                    println!("next:    vq model fetch-embedding");
                }
            }
            Ok(())
        }
        ModelCommand::FetchEmbedding => {
            let files = fetch_model_files()?;
            if json {
                #[derive(Serialize)]
                struct Output<'a> {
                    model: &'a str,
                    weights: &'a Path,
                    vocabulary: &'a Path,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Output {
                        model: MODEL_ID,
                        weights: &files.weights,
                        vocabulary: &files.vocabulary,
                    })?
                );
            } else {
                println!("model:      {MODEL_ID}");
                println!("weights:    {}", files.weights.display());
                println!("vocabulary: {}", files.vocabulary.display());
            }
            Ok(())
        }
        ModelCommand::StatusSpeech => {
            let status = speech_model_status();
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!("engine:  SenseVoiceSmall");
                println!("model:   {}", status.model);
                println!("file:    {}", status.file);
                println!("vad:     {}/{}", status.vad_model, status.vad_file);
                println!("runtime: {SENSEVOICE_RUNTIME_VERSION}");
                println!(
                    "status:  {}",
                    if status.ready { "ready" } else { "missing" }
                );
                println!("cache:   {}", status.cache_directory.display());
                if let Some(path) = status.weights {
                    println!(
                        "weights: {} ({} bytes)",
                        path.display(),
                        status.weights_bytes.unwrap_or_default()
                    );
                }
                if let Some(path) = status.vad_weights {
                    println!(
                        "vad:     {} ({} bytes)",
                        path.display(),
                        status.vad_weights_bytes.unwrap_or_default()
                    );
                }
                if let Some(path) = status.runtime {
                    println!("binary:  {}", path.display());
                }
                if !status.missing_files.is_empty() {
                    println!("missing: {}", status.missing_files.join(", "));
                    println!("next:    vq model fetch-speech");
                }
            }
            Ok(())
        }
        ModelCommand::FetchSpeech => {
            let files = fetch_speech_components()?;
            if json {
                #[derive(Serialize)]
                struct Output<'a> {
                    model: &'a str,
                    file: &'a str,
                    weights: &'a Path,
                    vad_model: &'a str,
                    vad_file: &'a str,
                    vad_weights: &'a Path,
                    runtime_version: &'a str,
                    runtime: &'a Path,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Output {
                        model: SPEECH_MODEL_ID,
                        file: SPEECH_MODEL_FILE,
                        weights: &files.weights,
                        vad_model: SPEECH_VAD_MODEL_ID,
                        vad_file: SPEECH_VAD_MODEL_FILE,
                        vad_weights: &files.vad_weights,
                        runtime_version: SENSEVOICE_RUNTIME_VERSION,
                        runtime: &files.runtime,
                    })?
                );
            } else {
                println!("model:       {SPEECH_MODEL_ID}");
                println!("file:        {SPEECH_MODEL_FILE}");
                println!("weights:     {}", files.weights.display());
                println!("vad model:   {SPEECH_VAD_MODEL_ID}");
                println!("vad file:    {SPEECH_VAD_MODEL_FILE}");
                println!("vad weights: {}", files.vad_weights.display());
                println!("runtime:     {}", files.runtime.display());
            }
            Ok(())
        }
        ModelCommand::StatusWhisper => {
            let status = whisper_model_status();
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!("model:  {}", status.model);
                println!("file:   {}", status.file);
                println!("status: {}", if status.ready { "ready" } else { "missing" });
                println!("cache:  {}", status.cache_directory.display());
                if let Some(path) = status.weights {
                    println!(
                        "weights: {} ({} bytes)",
                        path.display(),
                        status.weights_bytes.unwrap_or_default()
                    );
                } else {
                    println!("next:    vq model fetch-whisper");
                }
            }
            Ok(())
        }
        ModelCommand::FetchWhisper => {
            let weights = fetch_whisper_model()?;
            if json {
                #[derive(Serialize)]
                struct Output<'a> {
                    model: &'a str,
                    file: &'a str,
                    weights: &'a Path,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Output {
                        model: WHISPER_MODEL_ID,
                        file: WHISPER_MODEL_FILE,
                        weights: &weights,
                    })?
                );
            } else {
                println!("model:   {WHISPER_MODEL_ID}");
                println!("file:    {WHISPER_MODEL_FILE}");
                println!("weights: {}", weights.display());
            }
            Ok(())
        }
        ModelCommand::StatusTts => {
            let status = qwen_tts_model_status();
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!("model:   {}", status.model);
                println!(
                    "status:  {}",
                    if status.ready { "ready" } else { "missing" }
                );
                println!(
                    "runtime: {}",
                    if status.runtime_available {
                        "uv ready"
                    } else {
                        "uv missing"
                    }
                );
                println!("cache:   {}", status.cache_directory.display());
                if let Some(path) = status.weights {
                    println!(
                        "weights: {} ({} bytes)",
                        path.display(),
                        status.weights_bytes.unwrap_or_default()
                    );
                } else {
                    println!("next:    vq model fetch-tts");
                }
            }
            Ok(())
        }
        ModelCommand::FetchTts => {
            let status = fetch_qwen_tts_model()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!("model:   {}", status.model);
                println!("weights: {}", status.weights.as_deref().unwrap().display());
                println!(
                    "tokenizer: {}",
                    status.tokenizer_weights.as_deref().unwrap().display()
                );
            }
            Ok(())
        }
    }
}

fn command_stats(index_dir: &Path, json: bool) -> Result<()> {
    let index = VideoIndex::open(index_dir)?;
    let stats = index.stats()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("index:  {}", index.root().display());
        println!("videos: {}", stats.videos);
        println!("frames: {}", stats.frames);
    }
    Ok(())
}

fn command_doctor(json: bool) -> Result<()> {
    let ffmpeg = ensure_ffmpeg_available().is_ok();
    let model_status = model_status();
    let whisper_cli = whisper_cli_available();
    let speech_status = speech_model_status();
    let sensevoice_runtime = speech_status.runtime_ready;
    let uv = uv_available();
    let tts_supported = tts_platform_supported();
    let tts_model_downloaded = qwen_tts_model_status().ready;
    #[derive(Serialize)]
    struct Doctor<'a> {
        rust_binary: bool,
        ffmpeg: bool,
        sensevoice_runtime: bool,
        whisper_cli: bool,
        sensevoice_runtime_supported: bool,
        default_speech_engine: &'a str,
        embedding_model: &'a str,
        embedding_model_downloaded: bool,
        speech_model: &'a str,
        speech_model_downloaded: bool,
        speech_vad_downloaded: bool,
        model_cache: &'a Path,
        uv: bool,
        tts_supported: bool,
        tts_model: &'a str,
        tts_model_downloaded: bool,
    }
    let result = Doctor {
        rust_binary: true,
        ffmpeg,
        sensevoice_runtime,
        whisper_cli,
        sensevoice_runtime_supported: speech_status.runtime_supported,
        default_speech_engine: match default_speech_engine() {
            SpeechEngine::Sensevoice => "sensevoice",
            SpeechEngine::Whisper => "whisper",
        },
        embedding_model: MODEL_ID,
        embedding_model_downloaded: model_status.ready,
        speech_model: SPEECH_MODEL_FILE,
        speech_model_downloaded: speech_status.weights.is_some(),
        speech_vad_downloaded: speech_status.vad_weights.is_some(),
        model_cache: &model_status.cache_directory,
        uv,
        tts_supported,
        tts_model: QWEN_TTS_MODEL_ID,
        tts_model_downloaded,
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Rust CLI: ok");
        println!("FFmpeg:   {}", if ffmpeg { "ok" } else { "missing" });
        println!(
            "SenseVoice runtime: {}",
            if sensevoice_runtime {
                "ok"
            } else if !speech_status.runtime_supported {
                "unsupported on this platform; Whisper is the default"
            } else {
                "missing; run `vq model fetch-speech`"
            }
        );
        println!(
            "Whisper runtime:    {}",
            if whisper_cli {
                "ok (optional)"
            } else {
                "missing (optional)"
            }
        );
        println!(
            "Qwen TTS runtime:   {}",
            if !tts_supported {
                "unsupported (requires Apple silicon)"
            } else if uv && tts_model_downloaded {
                "ok; model ready"
            } else if uv {
                "ok; model downloads on first `vq speak`"
            } else {
                "missing uv"
            }
        );
        println!(
            "Image model:  {MODEL_ID} ({})",
            if model_status.ready {
                "ready"
            } else {
                "missing; run `vq model fetch-embedding`"
            }
        );
        println!(
            "Speech model: {SPEECH_MODEL_FILE} ({})",
            if speech_status.models_ready {
                "ready"
            } else {
                "missing; run `vq model fetch-speech`"
            }
        );
        println!("Cache:    {}", model_status.cache_directory.display());
    }
    if !ffmpeg {
        anyhow::bail!("ffmpeg and ffprobe must be available on PATH")
    }
    Ok(())
}

fn print_candidates(candidates: &[KeyframeCandidate], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(candidates)?);
    } else {
        println!(" timestamp   score      sharpness   contrast  clipped");
        for candidate in candidates {
            println!(
                " {:>9}  {:>9.3}  {:>11.3}  {:>8.2}  {:>7.2}%",
                format_timestamp(candidate.timestamp_seconds),
                candidate.quality.score,
                candidate.quality.laplacian_variance,
                candidate.quality.contrast,
                candidate.quality.clipped_fraction * 100.0,
            );
        }
    }
    Ok(())
}

fn frame_filename(timestamp_seconds: f64) -> String {
    format!(
        "{:012}.jpg",
        (timestamp_seconds.max(0.0) * 1000.0).round() as u64
    )
}

fn format_timestamp(seconds: f64) -> String {
    let total_millis = (seconds.max(0.0) * 1000.0).round() as u64;
    let millis = total_millis % 1000;
    let total_seconds = total_millis / 1000;
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    format!("{minutes:02}:{seconds:02}.{millis:03}")
}
