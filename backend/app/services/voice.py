"""Hinglish text-to-speech -- DEMO SCOPE.

WHAT THIS DOES: takes the Hinglish ``customer_message`` the agent already
produces (in ``execute_recovery_action`` / ``stop_recovery``) and synthesizes it
into a real audio FILE, stored under ``TTS_OUTPUT_DIR`` and retrievable at
``GET /api/v1/voice/{audio_id}`` -- mirroring how a Razorpay Payment Link URL is
generated and returned.

WHAT THIS DOES NOT DO (restated in the API, the tool notes, and the report):
  * NO outbound phone call is placed.
  * NO real-time voice conversation happens.
  * The customer does NOT receive this audio through any channel. The file
    exists and is retrievable, exactly as the Payment Link URL exists but is not
    yet dispatched over real SMS/WhatsApp.

Engine: ``pyttsx3`` -- the local, OS-native TTS engine (Windows SAPI5), chosen
over a network-dependent option (e.g. gTTS) per the project's local-first,
minimal-dependency posture and to avoid adding a second live-demo dependency on
external connectivity. KNOWN LIMITATION: the only SAPI voices installed in this
environment are ``en-US`` (Microsoft David / Zira), so the output is a real TTS
rendering of the romanized Hinglish text spoken with an American English
accent -- not native Hindi pronunciation. Swapping ``TTS_ENGINE`` later (e.g. a
Hindi neural voice) does not change any of the wiring here.

Fail-safe: if TTS is disabled or synthesis fails for any reason, this returns a
structured ``VoiceResult(generated=False, reason=...)`` and never raises into the
recovery flow.
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.config import settings

# pyttsx3 + SAPI5/COM is not thread-safe; serialise all synthesis.
_SYNTH_LOCK = threading.Lock()

_AUDIO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# reasons a VoiceResult can carry when generated is False
REASON_DISABLED = "tts_disabled"
REASON_FAILED = "tts_generation_failed"
REASON_EMPTY = "empty_message"


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool
    language: str
    output_dir: str
    engine: str

    @classmethod
    def from_settings(cls) -> "VoiceConfig":
        return cls(
            enabled=bool(settings.TTS_ENABLED),
            language=(settings.TTS_LANGUAGE or "hi").strip(),
            output_dir=(settings.TTS_OUTPUT_DIR or "artifacts/tts").strip(),
            engine=(settings.TTS_ENGINE or "pyttsx3").strip().lower(),
        )


@dataclass(frozen=True)
class VoiceResult:
    generated: bool
    reason: str | None = None
    audio_id: str | None = None
    audio_path: str | None = None
    audio_url: str | None = None
    audio_format: str | None = None
    audio_bytes: int | None = None
    engine: str | None = None
    voice_name: str | None = None

    def as_tool_fields(self) -> dict[str, Any]:
        """Compact fields for the agent tool result. Always carries an explicit
        reason when not generated -- never a bare boolean."""
        if self.generated:
            return {
                "voice_generated": True,
                "audio_url": self.audio_url,
                "audio_format": self.audio_format,
                "audio_bytes": self.audio_bytes,
                "voice_engine": self.engine,
                "voice_note": (
                    "real TTS audio FILE generated; NOT a phone call, NOT "
                    "delivered to the customer through any channel"
                ),
            }
        return {"voice_generated": False, "voice_reason": self.reason}


class Synthesizer(Protocol):
    engine: str
    audio_format: str

    def synthesize_to_file(self, text: str, out_path: Path, language: str) -> None: ...


class Pyttsx3Synthesizer:
    """Local Windows SAPI5 TTS via pyttsx3. Output: 16-bit PCM WAV."""

    engine = "pyttsx3"
    audio_format = "wav"

    def __init__(self) -> None:
        import pyttsx3  # noqa: F401 - import here so a missing engine is a
        #                              graceful failure, not an import-time crash

        self.voice_name: str | None = None
        try:
            eng = pyttsx3.init()
            voices = eng.getProperty("voices") or []
            self.voice_name = voices[0].name if voices else None
        except Exception:  # noqa: BLE001 - introspection only
            self.voice_name = None

    def synthesize_to_file(self, text: str, out_path: Path, language: str) -> None:
        import pyttsx3

        eng = pyttsx3.init()
        eng.save_to_file(text, str(out_path))
        eng.runAndWait()


def build_synthesizer(engine: str) -> Synthesizer:
    if engine == "pyttsx3":
        return Pyttsx3Synthesizer()
    raise ValueError(f"unsupported TTS_ENGINE {engine!r} (this stage wires 'pyttsx3' only)")


class VoiceService:
    def __init__(
        self,
        config: VoiceConfig | None = None,
        *,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.config = config or VoiceConfig.from_settings()
        self._synth = synthesizer  # injected in tests; built lazily otherwise

    @classmethod
    def from_settings(cls) -> "VoiceService":
        return cls(VoiceConfig.from_settings())

    def _synthesizer(self) -> Synthesizer:
        if self._synth is None:
            self._synth = build_synthesizer(self.config.engine)
        return self._synth

    @staticmethod
    def audio_id_for(key: str, text: str) -> str:
        safe_key = re.sub(r"[^A-Za-z0-9_-]", "-", key)[:32] or "run"
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return f"{safe_key}-{digest}"

    def synthesize(self, text: str | None, *, key: str) -> VoiceResult:
        """Synthesize ``text`` to an audio file. Never raises.

        Idempotent: the file name is deterministic in ``(key, text)``, so a
        re-run reuses the existing file rather than piling up audio.
        """
        if not self.config.enabled:
            return VoiceResult(False, reason=REASON_DISABLED)
        text = (text or "").strip()
        if not text:
            return VoiceResult(False, reason=REASON_EMPTY)

        audio_id = self.audio_id_for(key, text)
        try:
            synth = self._synthesizer()
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{audio_id}.{synth.audio_format}"

            if not (path.is_file() and path.stat().st_size > 0):
                with _SYNTH_LOCK:
                    synth.synthesize_to_file(text, path, self.config.language)

            size = path.stat().st_size if path.is_file() else 0
            if size <= 0:
                raise RuntimeError("synthesizer produced an empty / missing file")

            return VoiceResult(
                generated=True,
                audio_id=audio_id,
                audio_path=str(path),
                audio_url=f"/api/v1/voice/{audio_id}",
                audio_format=synth.audio_format,
                audio_bytes=size,
                engine=synth.engine,
                voice_name=getattr(synth, "voice_name", None),
            )
        except Exception:  # noqa: BLE001 - a TTS failure must never break recovery
            return VoiceResult(False, reason=REASON_FAILED)


def resolve_audio_file(audio_id: str, output_dir: str | None = None) -> Path | None:
    """Map an audio id to an on-disk file, guarding against path traversal."""
    if not _AUDIO_ID_RE.match(audio_id):
        return None
    base = Path(output_dir or VoiceConfig.from_settings().output_dir).resolve()
    for ext in ("wav", "mp3"):
        candidate = (base / f"{audio_id}.{ext}").resolve()
        if base in candidate.parents and candidate.is_file():
            return candidate
    return None
