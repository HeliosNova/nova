"""Voice API — speech-to-text transcription, text-to-speech, and voice chat."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.auth import require_auth
from app.config import config
from app.schema import EventType

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"], dependencies=[Depends(require_auth)])

# Max upload size (25MB)
MAX_AUDIO_SIZE = 25 * 1024 * 1024

# Max characters for TTS synthesis (prevent runaway requests)
MAX_TTS_CHARS = 5000

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mpeg", ".mpga", ".oga", ".opus"}


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)


@router.post("/voice/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(default="", max_length=10),
):
    """Transcribe an audio file to text using local Whisper."""
    if not config.ENABLE_VOICE:
        raise HTTPException(status_code=400, detail="Voice is disabled. Set ENABLE_VOICE=true")

    # Validate file extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {ext}")

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=400, detail=f"Audio file too large ({len(content)} bytes, max {MAX_AUDIO_SIZE})")

    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Write to temp file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp.close()

        from app.core.voice import get_transcriber
        transcriber = get_transcriber()
        result = await transcriber.transcribe(
            Path(tmp.name),
            language=language if language else None,
        )

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "model": config.WHISPER_MODEL_SIZE,
        }
    except RuntimeError as e:
        logger.error("[Voice] Transcription runtime error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed due to an internal error")
    except Exception as e:
        logger.error("[Voice] Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed due to an internal error")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@router.post("/voice/chat")
async def voice_chat(
    file: UploadFile = File(...),
    conversation_id: str = Query(default="", max_length=100),
    language: str = Query(default="", max_length=10),
):
    """Transcribe audio, then run it through the chat pipeline. Returns SSE stream."""
    if not config.ENABLE_VOICE:
        raise HTTPException(status_code=400, detail="Voice is disabled. Set ENABLE_VOICE=true")

    # Read and validate
    content = await file.read()
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=400, detail="Audio file too large")
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Transcribe
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp.close()

        from app.core.voice import get_transcriber
        transcriber = get_transcriber()
        transcription = await transcriber.transcribe(
            Path(tmp.name),
            language=language if language else None,
        )
    except Exception as e:
        logger.error("[Voice] Voice chat transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed due to an internal error")
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    if not transcription.text:
        raise HTTPException(status_code=400, detail="No speech detected in audio")

    # Stream the chat response
    from app.core.brain import think
    from app.schema import StreamEvent, EventType

    async def _stream():
        # First, emit the transcription
        transcription_event = StreamEvent(
            type=EventType.TOKEN,
            data={"type": "transcription", "text": transcription.text, "language": transcription.language, "duration": transcription.duration},
        )
        yield transcription_event.to_sse()

        # Then stream the response
        async for event in think(
            query=transcription.text,
            conversation_id=conversation_id if conversation_id else None,
        ):
            yield event.to_sse()

        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/voice/synthesize")
async def synthesize_speech(payload: SynthesizeRequest = Body(...)):
    """Synthesize text to speech using local Piper TTS. Returns audio/wav bytes.

    Sovereign — no external API. Requires Piper model file at TTS_MODEL_PATH
    (default /data/tts/en_US-amy-medium.onnx). Disabled by default.
    """
    if not getattr(config, "ENABLE_TTS", False):
        raise HTTPException(status_code=400, detail="TTS is disabled. Set ENABLE_TTS=true")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        from app.core.voice import get_synthesizer
        synthesizer = get_synthesizer()
        wav_bytes, sample_rate = await synthesizer.synthesize(text)
    except RuntimeError as e:
        # Missing model file or piper package — return clear error
        logger.error("[TTS] synthesis runtime error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("[TTS] synthesis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Synthesis failed")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-TTS-Sample-Rate": str(sample_rate)},
    )
