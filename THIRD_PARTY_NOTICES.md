# Third-party notices

## Chinese-CLIP

- Project: `OFA-Sys/chinese-clip-vit-base-patch16`
- Source: <https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16>
- License: Apache License 2.0

Model files are not bundled with this repository. `vq model fetch` downloads
them into the user's Hugging Face cache. Use and redistribution of those files
remain subject to the model's license and notice requirements.

## SenseVoiceSmall, FSMN-VAD, and the FunASR GGUF runtime

- Model: `FunAudioLLM/SenseVoiceSmall-GGUF`
- Model source: <https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF>
- VAD source: <https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF>
- Runtime source: <https://github.com/QwenAudio/SenseVoice>
- Model/VAD license: Apache License 2.0
- Runtime license: MIT

The GGUF models and native runtime are not bundled. `vq model fetch` downloads
all embedding/transcription models, while `vq model fetch-speech` downloads
only these speech assets, into the user's Hugging Face and `vq` runtime caches.
The native runtime release archive is verified against a pinned SHA-256 digest
before its `llama-funasr-sensevoice` binary is installed in the cache.

## Optional Whisper and whisper.cpp

- Model: OpenAI Whisper large-v3-turbo
- Converted model source: <https://huggingface.co/ggerganov/whisper.cpp>
- Runtime source: <https://github.com/ggml-org/whisper.cpp>
- License: MIT

The GGML model is not bundled. `vq model fetch` downloads all models, while
`vq model fetch-whisper` downloads only this model, into the user's Hugging Face
cache. `vq` invokes a separately installed `whisper-cli` executable and does
not redistribute whisper.cpp. Whisper is only used when the user selects
`vq transcribe --engine whisper`.

## Optional Qwen3-TTS and MLX-Audio

- Model: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit`
- Converted model source:
  <https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit>
- Upstream model source: <https://github.com/QwenLM/Qwen3-TTS>
- Runtime source: <https://github.com/Blaizzy/mlx-audio>
- Model license: Apache License 2.0
- Runtime license: MIT

The model and Python runtime are not bundled. On Apple silicon, `vq speak` and
the explicit `vq model fetch-tts` command invoke a pinned MLX-Audio package
through the separately installed `uv` executable. `vq speak` downloads model
files into the user's Hugging Face cache on first use. Use of voice cloning is
subject to the model license and requires the reference speaker's authorization.

## FFmpeg

`vq` invokes a separately installed FFmpeg/ffprobe executable. FFmpeg is not
bundled or redistributed by this repository. The applicable FFmpeg license
depends on how that external executable was built.
