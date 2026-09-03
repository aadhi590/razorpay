"""Fixtures + fake synthesizers for the Hinglish TTS layer.

The fakes plug into the real ``VoiceService`` via its ``synthesizer=`` hook, so
file naming, idempotency, the empty-file guard and the fail-safe wrapper are all
exercised for real -- only the audio engine is replaced. One test also runs the
real ``Pyttsx3Synthesizer`` and skips if the OS engine is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.voice import VoiceConfig, VoiceService


class FakeSynthesizer:
    """Writes a small but valid-looking file. No OS engine, no network."""

    engine = "fake"
    audio_format = "wav"

    def __init__(self, *, payload: bytes = b"RIFF....WAVEfake-audio-bytes" * 8) -> None:
        self.calls: list[tuple[str, str]] = []
        self._payload = payload

    def synthesize_to_file(self, text: str, out_path: Path, language: str) -> None:
        self.calls.append((text, language))
        out_path.write_bytes(self._payload)


class FailingSynthesizer:
    engine = "fake-failing"
    audio_format = "wav"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize_to_file(self, text: str, out_path: Path, language: str) -> None:
        self.calls += 1
        raise RuntimeError("simulated TTS engine failure")


class EmptyFileSynthesizer:
    engine = "fake-empty"
    audio_format = "wav"

    def synthesize_to_file(self, text: str, out_path: Path, language: str) -> None:
        out_path.write_bytes(b"")


@pytest.fixture
def tts_dir(tmp_path) -> Path:
    d = tmp_path / "tts"
    d.mkdir()
    return d


def make_service(
    tts_dir: Path,
    *,
    enabled: bool = True,
    synthesizer=None,
    language: str = "hi",
    engine: str = "pyttsx3",
) -> VoiceService:
    cfg = VoiceConfig(
        enabled=enabled, language=language,
        output_dir=str(tts_dir), engine=engine,
    )
    return VoiceService(cfg, synthesizer=synthesizer)
