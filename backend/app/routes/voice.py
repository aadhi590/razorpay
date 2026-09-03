"""Retrieve a generated Hinglish TTS audio file.

    GET /api/v1/voice/{audio_id}   -> the audio file (audio/wav)

This mirrors how a Razorpay Payment Link URL is exposed: the recovery agent
generates the artifact and returns its URL; this endpoint just serves it.

IMPORTANT: serving this file is NOT a phone call and NOT delivery to a customer.
The audio exists and is retrievable; it reaches no customer through any channel
in this stage.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.services.voice import resolve_audio_file

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

_MEDIA = {".wav": "audio/wav", ".mp3": "audio/mpeg"}


@router.get("/{audio_id}")
def get_voice_audio(audio_id: str) -> FileResponse:
    path = resolve_audio_file(audio_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="audio file not found for this id",
        )
    return FileResponse(
        path,
        media_type=_MEDIA.get(path.suffix.lower(), "application/octet-stream"),
        filename=path.name,
    )
