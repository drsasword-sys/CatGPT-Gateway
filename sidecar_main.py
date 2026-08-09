"""Native Windows entry point for the App-Stran CatGPT sidecar."""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def configure_bundled_browser() -> None:
    """Point Patchright at Chromium shipped beside the onedir executable."""
    if not getattr(sys, "frozen", False):
        return
    browser_root = Path(sys.executable).resolve().parent / "browser-runtime"
    if browser_root.is_dir():
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(browser_root))


def validate_native_security() -> None:
    """Fail closed when the packaged sidecar is started without its supervisor."""
    host = os.environ.setdefault("API_HOST", "127.0.0.1").strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Native CatGPT sidecar must bind to loopback")
    if not os.getenv("API_TOKEN", "").strip():
        raise RuntimeError("Native CatGPT sidecar requires a random API token")


def main() -> None:
    configure_bundled_browser()
    os.environ.setdefault("NATIVE_SIDECAR", "true")
    validate_native_security()

    import uvicorn
    from src.api.server import app
    from src.config import Config

    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT, log_level="info")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
