"""Voice transcription — local Whisper speech-to-text."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import config

logger = logging.getLogger(__name__)

_HAS_WHISPER = False
try:
    import whisper
    _HAS_WHISPER = True
except ImportError:
    pass


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float  # audio duration in seconds


class WhisperTranscriber:
    """Lazy-loaded Whisper model for speech-to-text."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            if not _HAS_WHISPER:
                raise RuntimeError("openai-whisper is not installed. Install with: pip install openai-whisper")
            logger.info("[Voice] Loading Whisper model '%s'...", self.model_size)
            self._model = whisper.load_model(self.model_size)
            logger.info("[Voice] Whisper model loaded")

    async def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file. Runs in thread pool since Whisper is synchronous."""
        self._ensure_loaded()

        def _run():
            options = {}
            if language:
                options["language"] = language
            result = self._model.transcribe(str(audio_path), **options)
            # Get audio duration
            import whisper
            audio = whisper.load_audio(str(audio_path))
            duration = len(audio) / whisper.audio.SAMPLE_RATE
            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language", "unknown"),
                duration=round(duration, 1),
            )

        return await asyncio.to_thread(_run)

    def unload(self):
        """Free GPU memory by unloading the model."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("[Voice] Whisper model unloaded")


# Module-level singleton
_transcriber: WhisperTranscriber | None = None


def get_transcriber() -> WhisperTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber(model_size=config.WHISPER_MODEL_SIZE)
    return _transcriber


def unload_transcriber():
    global _transcriber
    if _transcriber:
        _transcriber.unload()
        _transcriber = None


# ---------------------------------------------------------------------------
# Text-to-speech (Piper) — voice OUT
# ---------------------------------------------------------------------------

_HAS_PIPER = False
try:
    from piper.voice import PiperVoice
    _HAS_PIPER = True
except ImportError:
    pass


class PiperSynthesizer:
    """Lazy-loaded Piper TTS for local sovereign text-to-speech.

    Mirrors WhisperTranscriber's pattern: lazy load, async wrapper, unload
    method. Output is 16-bit PCM WAV at the model's native sample rate.
    """

    def __init__(self, model_path: str | None = None):
        # Default to /data/tts/en_US-amy-medium.onnx if no path given.
        # Model file must be downloaded separately — see
        # https://github.com/rhasspy/piper for the Hugging Face URL.
        self.model_path = model_path or config.TTS_MODEL_PATH
        self._voice = None

    def _ensure_loaded(self):
        if self._voice is None:
            if not _HAS_PIPER:
                raise RuntimeError(
                    "piper-tts is not installed. Install with: pip install piper-tts"
                )
            mp = Path(self.model_path)
            if not mp.exists():
                raise RuntimeError(
                    f"Piper model not found at {self.model_path}. "
                    f"Download from https://huggingface.co/rhasspy/piper-voices "
                    f"and set TTS_MODEL_PATH."
                )
            logger.info("[TTS] Loading Piper voice from %s ...", self.model_path)
            self._voice = PiperVoice.load(str(mp))
            logger.info("[TTS] Piper voice loaded (sample_rate=%d)",
                        self._voice.config.sample_rate)

    async def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesize text to a WAV byte string. Returns (wav_bytes, sample_rate).

        Runs in a thread because Piper inference is CPU-bound and synchronous.
        """
        self._ensure_loaded()
        text = (text or "").strip()
        if not text:
            return b"", 0

        def _run():
            import io
            import wave
            buf = io.BytesIO()
            sample_rate = self._voice.config.sample_rate
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(sample_rate)
                self._voice.synthesize(text, wf)
            return buf.getvalue(), sample_rate

        return await asyncio.to_thread(_run)

    def unload(self):
        if self._voice is not None:
            del self._voice
            self._voice = None
            logger.info("[TTS] Piper voice unloaded")


_synthesizer: PiperSynthesizer | None = None


def get_synthesizer() -> PiperSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = PiperSynthesizer()
    return _synthesizer


def unload_synthesizer():
    global _synthesizer
    if _synthesizer:
        _synthesizer.unload()
        _synthesizer = None
