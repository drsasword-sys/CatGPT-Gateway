"""
Centralized configuration — loads from .env with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

_CODE_ROOT = Path(__file__).resolve().parent.parent
_CWD = Path.cwd()

# Prefer the invocation directory as project root when running from
# a checkout (e.g. `nix run .#proxy` from repo root). Fall back to the
# code location (used for packaged/store execution).
if (_CWD / "src").exists() and (_CWD / "scripts").exists():
    _PROJECT_ROOT = _CWD
else:
    _PROJECT_ROOT = _CODE_ROOT

# Load .env from current working directory first, then from the
# resolved project root.
load_dotenv(_CWD / ".env")
load_dotenv(_PROJECT_ROOT / ".env")


def _runtime_path(env_name: str, fallback: Path) -> Path:
    """Resolve an optional absolute/relative runtime directory safely."""
    configured = os.getenv(env_name, "").strip()
    if not configured:
        return fallback
    candidate = Path(configured).expanduser()
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


_NATIVE_SIDECAR = os.getenv("NATIVE_SIDECAR", "false").lower() == "true"
_LOCAL_APP_DATA = Path(os.getenv("LOCALAPPDATA", str(_PROJECT_ROOT)))
_NATIVE_DATA_ROOT = _LOCAL_APP_DATA / "AppStran" / "CatGPT"


class Config:
    """All project settings in one place."""

    # Paths
    PROJECT_ROOT: Path = _PROJECT_ROOT
    NATIVE_SIDECAR: bool = _NATIVE_SIDECAR
    BROWSER_DATA_DIR: Path = _runtime_path(
        "BROWSER_DATA_DIR",
        (_NATIVE_DATA_ROOT / "browser-profile") if _NATIVE_SIDECAR else (_PROJECT_ROOT / "browser_data"),
    )
    LOG_DIR: Path = _runtime_path(
        "LOG_DIR",
        (_NATIVE_DATA_ROOT / "logs") if _NATIVE_SIDECAR else (_PROJECT_ROOT / "logs"),
    )
    IMAGES_DIR: Path = _runtime_path(
        "IMAGES_DIR",
        (_NATIVE_DATA_ROOT / "downloads" / "images") if _NATIVE_SIDECAR else (_PROJECT_ROOT / "downloads/images"),
    )

    # Browser
    HEADLESS: bool = os.getenv("HEADLESS", "false").lower() == "true"
    SLOW_MO: int = int(os.getenv("SLOW_MO", "25"))
    CHATGPT_URL: str = os.getenv("CHATGPT_URL", "https://chatgpt.com")
    CLAUDE_URL: str = os.getenv("CLAUDE_URL", "https://claude.ai")

    # Provider selection: "chatgpt" or "claude"
    PROVIDER: str = os.getenv("PROVIDER", "chatgpt").lower()

    @classmethod
    def provider_url(cls) -> str:
        """Return the target URL for the active provider."""
        if cls.PROVIDER == "claude":
            return cls.CLAUDE_URL
        return cls.CHATGPT_URL

    # Timeouts (ms)
    RESPONSE_TIMEOUT: int = int(os.getenv("RESPONSE_TIMEOUT", "120000"))
    SELECTOR_TIMEOUT: int = int(os.getenv("SELECTOR_TIMEOUT", "10000"))

    # Human simulation (ms)
    TYPING_SPEED_MIN: int = int(os.getenv("TYPING_SPEED_MIN", "50"))
    TYPING_SPEED_MAX: int = int(os.getenv("TYPING_SPEED_MAX", "150"))
    THINKING_PAUSE_MIN: int = int(os.getenv("THINKING_PAUSE_MIN", "500"))
    THINKING_PAUSE_MAX: int = int(os.getenv("THINKING_PAUSE_MAX", "1500"))
    # Completion poll interval — how often to check if response is ready (ms)
    POLL_INTERVAL_MS: int = int(os.getenv("POLL_INTERVAL_MS", "300"))
    # ChatGPT Web rate-limits bursts of sidebar/project navigation. Lane
    # pages remain concurrent for generation, but their initial UI setup is
    # deliberately staggered.
    LANE_BOOTSTRAP_GAP_SECONDS: float = float(os.getenv("LANE_BOOTSTRAP_GAP_SECONDS", "5"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    VERBOSE: bool = os.getenv("VERBOSE", "true").lower() == "true"

    # API (Phase 3)
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    RATE_LIMIT_SECONDS: int = int(os.getenv("RATE_LIMIT_SECONDS", "5"))
    API_TOKEN: str = os.getenv("API_TOKEN", "")  # Bearer token for API auth (empty = no auth)
    LANE_COUNT: int = max(1, min(10, int(os.getenv("CATGPT_LANE_COUNT", "2"))))
    MAX_CONCURRENCY: int = max(
        1,
        min(LANE_COUNT, int(os.getenv("CATGPT_MAX_CONCURRENCY", str(LANE_COUNT)))),
    )

    # VNC
    VNC_PASSWORD: str = os.getenv("VNC_PASSWORD", "catgpt")

    # Viewport base (will be jittered ±20px)
    VIEWPORT_WIDTH: int = 1280
    VIEWPORT_HEIGHT: int = 720

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        cls.BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
