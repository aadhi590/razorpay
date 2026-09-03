"""
Standalone diagnostic: verify Gemini API auth and enumerate the models the key can see.

This is NOT application code. It is a read-only utility:
  - Loads GEMINI_API_KEY only from the project's .env (never printed or logged).
  - Sends the key in the `x-goog-api-key` header (never in the URL / query string).
  - Performs a single read-only call: GET v1beta/models  (models.list).
  - Makes NO generation request and nothing billable.

Usage (from the project root):
    .venv/Scripts/python.exe scripts/gemini_models_probe.py

Optionally writes the raw JSON response next to this script for later reference:
    .venv/Scripts/python.exe scripts/gemini_models_probe.py --dump
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
DUMP_PATH = SCRIPT_DIR / "gemini_models_probe.json"
BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def load_key() -> str:
    """Return GEMINI_API_KEY from .env, or exit with a clear message."""
    if not ENV_PATH.is_file():
        sys.exit(f"STOP: {ENV_PATH} does not exist. Create it with GEMINI_API_KEY=<key>.")
    try:
        from dotenv import dotenv_values

        values = dotenv_values(ENV_PATH)
    except Exception:  # pragma: no cover - fallback if python-dotenv is absent
        values = {}
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip().strip('"').strip("'")
    key = (values.get("GEMINI_API_KEY") or "").strip()
    if not key:
        sys.exit("STOP: GEMINI_API_KEY is missing from .env")
    return key


def fetch_models(key: str) -> list[dict]:
    """Read-only models.list, following pagination. Key travels only in the header."""
    models: list[dict] = []
    page_token: str | None = None
    while True:
        url = f"{BASE}?pageSize=1000"
        if page_token:
            url += f"&pageToken={urllib.parse.quote(page_token)}"
        req = urllib.request.Request(url, headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models.extend(data.get("models", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return models


def short(name: str) -> str:
    return name[len("models/"):] if name.startswith("models/") else name


def classify(m: dict) -> set[str]:
    n = short(m.get("name", "")).lower()
    desc = (m.get("description") or "").lower()
    methods = set(m.get("supportedGenerationMethods") or [])
    tags: set[str] = set()

    gen = "generateContent" in methods
    live = "bidiGenerateContent" in methods

    is_embed = "embedding" in n or "embedContent" in methods
    is_image_out = "image" in n or "imagen" in n
    is_tts = "tts" in n
    is_aqa = "aqa" in n

    if gen and n.startswith("gemini") and not (is_embed or is_image_out or is_tts or is_aqa):
        tags.add("text_agent")
        tags.add("function_calling")  # all Gemini generateContent models support tools
        if any(v in n for v in ("1.5", "2.0", "2.5", "3")):
            tags.add("translation_transcription")

    if is_tts or live or "native-audio" in n or "audio" in n or "audio" in desc:
        tags.add("audio_voice")
    if live:
        tags.add("text_agent")

    return tags


def col(s: object, w: int) -> str:
    s = "" if s is None else str(s)
    return (s[: w - 1] + "…") if len(s) > w else s.ljust(w)


def print_table(models: list[dict]) -> None:
    hw = (38, 32, 46, 9, 9)
    header = (
        col("MODEL NAME", hw[0]) + "  " + col("DISPLAY NAME", hw[1]) + "  "
        + col("GENERATION METHODS", hw[2]) + "  " + col("IN-TOK", hw[3]) + "  " + col("OUT-TOK", hw[4])
    )
    print(header)
    print("-" * len(header))
    for m in sorted(models, key=lambda x: short(x.get("name", ""))):
        methods = ",".join(m.get("supportedGenerationMethods") or []) or "-"
        print(
            col(short(m.get("name", "")), hw[0]) + "  "
            + col(m.get("displayName", "-"), hw[1]) + "  "
            + col(methods, hw[2]) + "  "
            + col(m.get("inputTokenLimit", "-"), hw[3]) + "  "
            + col(m.get("outputTokenLimit", "-"), hw[4])
        )


def print_bucket(title: str, models: list[dict], tag: str) -> None:
    print(f"\n=== {title} ===")
    rows = [m for m in models if tag in classify(m)]
    if not rows:
        print("  (none)")
        return
    for m in sorted(rows, key=lambda x: short(x.get("name", ""))):
        methods = ",".join(m.get("supportedGenerationMethods") or [])
        print(
            f"  - {short(m.get('name', '')):42s} "
            f"in={str(m.get('inputTokenLimit', '-')):>9} "
            f"out={str(m.get('outputTokenLimit', '-')):>7}  [{methods}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", action="store_true",
        help=f"write the raw JSON response to {DUMP_PATH.name} next to this script",
    )
    args = parser.parse_args()

    key = load_key()
    print("Key loaded from .env (not displayed). Calling models.list ...\n")
    try:
        models = fetch_models(key)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            body = json.dumps(json.loads(body), indent=2)
        except Exception:
            pass
        print(f"AUTH/REQUEST FAILED: HTTP {e.code} {e.reason}\n{body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"NETWORK ERROR: {e.reason}")
        sys.exit(1)

    print(f"AUTH OK. {len(models)} models returned.\n")
    print_table(models)
    print_bucket("Normal text / agent reasoning", models, "text_agent")
    print_bucket("Function calling / tool use", models, "function_calling")
    print_bucket("Audio / voice input-output", models, "audio_voice")
    print_bucket("Translation / transcription (audio-in, text-out)", models, "translation_transcription")

    if args.dump:
        DUMP_PATH.write_text(json.dumps(models, indent=2), encoding="utf-8")
        print(f"\nRaw model list written to {DUMP_PATH}")


if __name__ == "__main__":
    main()
