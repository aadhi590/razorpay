from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore[import-not-found]
else:
    try:
        from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        from pydantic import BaseSettings

        SettingsConfigDict = dict


class Settings(BaseSettings):
    DATABASE_URL: str

    # --- Gemini agent layer (app/agent/) -------------------------------
    # All optional with conservative free-tier defaults, so the core API/DB
    # behaviour is unchanged when they are absent. GEMINI_API_KEY is read from
    # the environment / .env only; it is never logged or serialised.
    GEMINI_API_KEY: str | None = None
    # Requested default was "gemini-2.5-flash"; this API key's generateContent
    # endpoint rejects it ("no longer available to new users") and points to the
    # current flash line. "gemini-flash-latest" is the stable free-tier alias
    # that always resolves to the current Flash model. Override with GEMINI_MODEL.
    GEMINI_MODEL: str = "gemini-flash-latest"
    GEMINI_MAX_TURNS: int = 4
    GEMINI_TIMEOUT_SECONDS: float = 20.0

    # --- Razorpay integration (app/integrations/razorpay/) -------------
    # TEST MODE ONLY in this stage. All optional: when unset, the live
    # execution path refuses to run and the agent degrades gracefully; the
    # rules / ML / uplift / dry-run paths are entirely unaffected. Secrets are
    # read from the environment / .env only and are never logged or serialised.
    RAZORPAY_KEY_ID: str | None = None
    RAZORPAY_KEY_SECRET: str | None = None
    RAZORPAY_WEBHOOK_SECRET: str | None = None
    RAZORPAY_BASE_URL: str = "https://api.razorpay.com/v1"
    RAZORPAY_TEST_MODE: bool = True
    RAZORPAY_TIMEOUT_SECONDS: float = 15.0
    # Razorpay requires expire_by to be > 15 minutes from now; the adapter
    # clamps anything smaller up to 16.
    RAZORPAY_PAYMENT_LINK_EXPIRY_MINUTES: int = 60

    # --- Hinglish voice / text-to-speech (app/services/voice.py) -------
    # DEMO SCOPE: converts the agent's existing Hinglish customer_message into a
    # real synthesized audio FILE (retrievable via URL). It does NOT place a
    # phone call and the customer receives nothing through any channel in this
    # stage. Off by default; failures fall back to text-only, never crash the
    # recovery flow.
    TTS_ENABLED: bool = False
    # Language code passed to the engine. "hi" = Hindi (used for Hinglish).
    TTS_LANGUAGE: str = "hi"
    TTS_OUTPUT_DIR: str = "artifacts/tts"
    # "pyttsx3" = local OS-native TTS (Windows SAPI5), no network. The only
    # engine wired in this stage.
    TTS_ENGINE: str = "pyttsx3"

    # --- Frontend / browser access (app/main.py CORS) -----------------
    # Comma-separated list of exact browser origins allowed to call this API.
    # Read-only dashboard + the single agent-run POST; no cookies, no auth
    # headers are used, so credentials are NOT allowed on CORS requests. Empty
    # string disables CORS entirely (unchanged pre-frontend behaviour).
    CORS_ALLOW_ORIGINS: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173"
    )

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    if "SettingsConfigDict" in globals() and SettingsConfigDict is not dict:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=True,
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True


settings = Settings()