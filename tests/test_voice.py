"""Tests for Voice Interface — Whisper transcription and voice API endpoints."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# WhisperTranscriber unit tests (mocked whisper module)
# ===========================================================================


class TestTranscriptionResult:
    """Test the TranscriptionResult dataclass."""

    def test_fields(self):
        from app.core.voice import TranscriptionResult
        result = TranscriptionResult(text="hello world", language="en", duration=3.5)
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.duration == 3.5


class TestWhisperTranscriber:
    """Test WhisperTranscriber with mocked whisper module."""

    def test_lazy_loading(self):
        """Model should not be loaded until first call."""
        from app.core.voice import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size="base")
        assert transcriber._model is None

    def test_ensure_loaded_no_whisper(self):
        """Should raise RuntimeError when whisper is not installed."""
        from app.core.voice import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size="base")
        with patch("app.core.voice._HAS_WHISPER", False):
            with pytest.raises(RuntimeError, match="openai-whisper is not installed"):
                transcriber._ensure_loaded()

    def test_ensure_loaded_with_whisper(self):
        """Should load model when whisper is available."""
        from app.core.voice import WhisperTranscriber
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()

        transcriber = WhisperTranscriber(model_size="base")
        with patch("app.core.voice._HAS_WHISPER", True):
            with patch("app.core.voice.whisper", mock_whisper, create=True):
                transcriber._ensure_loaded()
                assert transcriber._model is not None
                mock_whisper.load_model.assert_called_once_with("base")

    def test_unload(self):
        """Should clear the model reference."""
        from app.core.voice import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size="base")
        transcriber._model = MagicMock()
        transcriber.unload()
        assert transcriber._model is None

    def test_unload_when_not_loaded(self):
        """Unloading when no model is loaded should be a no-op."""
        from app.core.voice import WhisperTranscriber
        transcriber = WhisperTranscriber(model_size="base")
        transcriber.unload()  # Should not raise
        assert transcriber._model is None

    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        """Transcribe should return cleaned text, language, and duration."""
        from app.core.voice import WhisperTranscriber

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": " hello world ",
            "language": "en",
        }

        mock_whisper = MagicMock()
        # Simulate audio array of 48000 samples at 16000 sample rate = 3.0s
        mock_audio = MagicMock()
        mock_audio.__len__ = lambda self: 48000
        mock_whisper.load_audio.return_value = mock_audio
        mock_whisper.audio.SAMPLE_RATE = 16000

        transcriber = WhisperTranscriber(model_size="base")
        transcriber._model = mock_model

        # _run() does `import whisper` — patch sys.modules so it resolves
        with patch("app.core.voice._HAS_WHISPER", True), \
             patch.dict(sys.modules, {"whisper": mock_whisper}):
            result = await transcriber.transcribe(Path("/tmp/test.wav"))

        assert result.text == "hello world"  # stripped
        assert result.language == "en"
        assert result.duration == 3.0
        mock_model.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_max_duration_exceeded(self):
        """Transcribe audio exceeding VOICE_MAX_DURATION (300s) still returns result.

        Note: VOICE_MAX_DURATION is defined in config but not currently
        enforced in the transcription pipeline. This test documents that
        long audio is processed without error at the transcriber level.
        """
        from app.core.voice import WhisperTranscriber

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "very long audio",
            "language": "en",
        }

        mock_whisper = MagicMock()
        # 301 seconds of audio at 16000 Hz sample rate
        mock_audio = MagicMock()
        mock_audio.__len__ = lambda self: 301 * 16000
        mock_whisper.load_audio.return_value = mock_audio
        mock_whisper.audio.SAMPLE_RATE = 16000

        transcriber = WhisperTranscriber(model_size="base")
        transcriber._model = mock_model

        # _run() does `import whisper` — patch sys.modules so it resolves
        with patch("app.core.voice._HAS_WHISPER", True), \
             patch.dict(sys.modules, {"whisper": mock_whisper}):
            result = await transcriber.transcribe(Path("/tmp/long.wav"))

        assert result.duration == 301.0
        assert result.text == "very long audio"

    @pytest.mark.asyncio
    async def test_transcribe_exception_handling(self):
        """Transcribe should propagate RuntimeError from whisper model."""
        from app.core.voice import WhisperTranscriber

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

        transcriber = WhisperTranscriber(model_size="base")
        transcriber._model = mock_model

        with patch("app.core.voice._HAS_WHISPER", True):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                await transcriber.transcribe(Path("/tmp/test.wav"))


class TestGetTranscriber:
    """Test the module-level singleton."""

    def test_singleton(self):
        """get_transcriber should return the same instance."""
        import app.core.voice as voice_mod
        voice_mod._transcriber = None
        with patch.object(voice_mod, "config") as mock_cfg:
            mock_cfg.WHISPER_MODEL_SIZE = "tiny"
            t1 = voice_mod.get_transcriber()
            t2 = voice_mod.get_transcriber()
            assert t1 is t2
            assert t1.model_size == "tiny"
        voice_mod._transcriber = None

    def test_unload_transcriber(self):
        """unload_transcriber should clear the singleton."""
        import app.core.voice as voice_mod
        voice_mod._transcriber = None
        with patch.object(voice_mod, "config") as mock_cfg:
            mock_cfg.WHISPER_MODEL_SIZE = "base"
            voice_mod.get_transcriber()
            assert voice_mod._transcriber is not None
            voice_mod.unload_transcriber()
            assert voice_mod._transcriber is None


# ===========================================================================
# Voice API endpoint validation tests
# ===========================================================================


class TestVoiceTranscribeEndpoint:
    """Test /api/voice/transcribe validation logic."""

    @pytest.fixture
    def client(self, db):
        """FastAPI test client with voice router enabled."""
        import importlib, app.config, app.auth
        importlib.reload(app.config)
        importlib.reload(app.auth)

        from fastapi import FastAPI
        from app.api.voice import router
        from app.main import _rate_limit_requests

        _rate_limit_requests.clear()

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        from fastapi.testclient import TestClient
        return TestClient(test_app)

    def test_voice_disabled(self, client):
        """Should return 400 when voice is disabled."""
        with patch("app.api.voice.config") as mock_cfg:
            mock_cfg.ENABLE_VOICE = False
            resp = client.post(
                "/api/voice/transcribe",
                files={"file": ("test.wav", io.BytesIO(b"fake audio"), "audio/wav")},
            )
            assert resp.status_code == 400
            assert "disabled" in resp.json()["detail"].lower()

    def test_empty_file(self, client):
        """Should return 400 for empty audio file."""
        with patch("app.api.voice.config") as mock_cfg:
            mock_cfg.ENABLE_VOICE = True
            resp = client.post(
                "/api/voice/transcribe",
                files={"file": ("test.wav", io.BytesIO(b""), "audio/wav")},
            )
            assert resp.status_code == 400
            assert "empty" in resp.json()["detail"].lower()

    def test_unsupported_format(self, client):
        """Should return 400 for unsupported audio format."""
        with patch("app.api.voice.config") as mock_cfg:
            mock_cfg.ENABLE_VOICE = True
            resp = client.post(
                "/api/voice/transcribe",
                files={"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")},
            )
            assert resp.status_code == 400
            assert "unsupported" in resp.json()["detail"].lower()

    def test_file_too_large(self, client):
        """Should return 400 for audio file exceeding 25MB limit."""
        with patch("app.api.voice.config") as mock_cfg:
            mock_cfg.ENABLE_VOICE = True
            # Create a payload just over the 25MB limit
            oversized = b"x" * (25 * 1024 * 1024 + 1)
            resp = client.post(
                "/api/voice/transcribe",
                files={"file": ("test.wav", io.BytesIO(oversized), "audio/wav")},
            )
            assert resp.status_code == 400
            assert "too large" in resp.json()["detail"].lower()


class TestVoiceChatEndpoint:
    """Test /api/voice/chat validation logic."""

    @pytest.fixture
    def client(self, db):
        """FastAPI test client with voice router enabled."""
        import importlib, app.config, app.auth
        importlib.reload(app.config)
        importlib.reload(app.auth)

        from fastapi import FastAPI
        from app.api.voice import router
        from app.main import _rate_limit_requests

        _rate_limit_requests.clear()

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api")

        from fastapi.testclient import TestClient
        return TestClient(test_app)

    def test_voice_disabled(self, client):
        """Should return 400 when voice is disabled."""
        with patch("app.api.voice.config") as mock_cfg:
            mock_cfg.ENABLE_VOICE = False
            resp = client.post(
                "/api/voice/chat",
                files={"file": ("test.wav", io.BytesIO(b"fake audio"), "audio/wav")},
            )
            assert resp.status_code == 400
            assert "disabled" in resp.json()["detail"].lower()

    def test_empty_file(self, client):
        """Should return 400 for empty audio file."""
        with patch("app.api.voice.config") as mock_cfg:
            mock_cfg.ENABLE_VOICE = True
            resp = client.post(
                "/api/voice/chat",
                files={"file": ("test.wav", io.BytesIO(b""), "audio/wav")},
            )
            assert resp.status_code == 400
            assert "empty" in resp.json()["detail"].lower()

    def test_voice_chat_no_speech(self, client):
        """Should return 400 when transcription yields empty text (no speech)."""
        from app.core.voice import TranscriptionResult

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="", language="en", duration=2.0)
        )

        with patch("app.api.voice.config") as mock_cfg, \
             patch("app.core.voice.get_transcriber", return_value=mock_transcriber):
            mock_cfg.ENABLE_VOICE = True
            resp = client.post(
                "/api/voice/chat",
                files={"file": ("test.wav", io.BytesIO(b"fake audio"), "audio/wav")},
            )
            assert resp.status_code == 400
            assert "no speech" in resp.json()["detail"].lower()

    def test_voice_chat_unsupported_format(self, client):
        """POST a .txt file to /voice/chat — should fail at transcription."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe = AsyncMock(
            side_effect=Exception("Failed to load audio")
        )

        with patch("app.api.voice.config") as mock_cfg, \
             patch("app.core.voice.get_transcriber", return_value=mock_transcriber):
            mock_cfg.ENABLE_VOICE = True
            resp = client.post(
                "/api/voice/chat",
                files={"file": ("test.txt", io.BytesIO(b"not audio data"), "text/plain")},
            )
            assert resp.status_code == 500
            assert "transcription failed" in resp.json()["detail"].lower()


# ===========================================================================
# Config additions
# ===========================================================================


class TestVoiceConfig:
    """Verify new config fields exist with correct defaults."""

    def test_voice_defaults(self):
        from app.config import Config
        cfg = Config()
        assert cfg.ENABLE_VOICE is False
        assert cfg.WHISPER_MODEL_SIZE == "base"
        assert cfg.VOICE_MAX_DURATION == 300
