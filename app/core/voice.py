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
