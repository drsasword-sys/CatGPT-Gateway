"""
FastAPI server — serves ChatGPT as an API.

Launches the browser on startup, shuts it down on exit.

Usage:
    python -m src.api.server
    # or
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.browser.manager import BrowserManager
from src.chatgpt.client import ChatGPTClient
from src.chatgpt.lane_pool import ChatGPTLanePool
from src.claude.client import ClaudeClient
from src.config import Config
from src.api.routes import router, set_client
from src.api.openai_routes import openai_router, set_lane_pool, set_openai_client
from src.log import setup_logging

log = setup_logging("api_server")

# Global instances — needed for lifespan
_browser: BrowserManager | None = None
_client: ChatGPTClient | ClaudeClient | None = None
_lane_pool: ChatGPTLanePool | None = None


def validate_rca_runtime_security() -> None:
    """Fail startup when effective RCA-mode settings are unsafe."""
    if not Config.RCA_MODE:
        return
    if Config.PROVIDER != "chatgpt":
        raise RuntimeError("RCA mode requires the ChatGPT provider")
    provider_url = urlparse(Config.CHATGPT_URL)
    try:
        safe_provider_url = (
            provider_url.scheme == "https"
            and provider_url.hostname == "chatgpt.com"
            and provider_url.username is None
            and provider_url.password is None
            and provider_url.port is None
            and provider_url.path in {"", "/"}
            and not provider_url.params
            and not provider_url.query
            and not provider_url.fragment
        )
    except ValueError:
        safe_provider_url = False
    if not safe_provider_url:
        raise RuntimeError("RCA mode requires the canonical ChatGPT URL")
    if Config.API_HOST != "127.0.0.1":
        raise RuntimeError("RCA mode API host must be exactly 127.0.0.1")
    if not Config.API_TOKEN.strip():
        raise RuntimeError("RCA mode requires a non-empty API token")
    if Config.LANE_COUNT != 1:
        raise RuntimeError("RCA mode requires lane count 1")
    if Config.MAX_CONCURRENCY != 1:
        raise RuntimeError("RCA mode requires max concurrency 1")
    if "*" in Config.CORS_ORIGINS:
        raise RuntimeError("RCA mode forbids wildcard CORS")
    for origin in Config.CORS_ORIGINS:
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise RuntimeError("RCA mode CORS origins must be pinned to loopback")
    if Config.PROVIDER_DEADLINE_MS <= 0:
        raise RuntimeError("RCA mode requires a finite provider deadline")
    if Config.COMPLETION_STABLE_SAMPLES < 3:
        raise RuntimeError("RCA mode requires at least 3 stable samples")
    if Config.COMPLETION_STABLE_MS < 3000:
        raise RuntimeError("RCA mode requires at least 3000ms text stability")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: launch browser. Shutdown: close it."""
    global _browser, _client, _lane_pool

    validate_rca_runtime_security()
    log.info("Starting browser for API server...")
    _browser = BrowserManager()
    page = await _browser.start()

    target_url = Config.provider_url()
    provider_name = "Claude" if Config.PROVIDER == "claude" else "ChatGPT"
    log.info(f"Provider: {provider_name} ({target_url})")

    # Navigate with retries (DNS can be slow in Docker)
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Navigation attempt {attempt}/{max_retries} to {target_url}")
            await _browser.navigate(target_url)
            break
        except Exception as e:
            log.warning(f"Navigation attempt {attempt} failed: {e}")
            if attempt == max_retries:
                log.error("All navigation attempts failed")
                raise
            wait_time = attempt * 5  # 5s, 10s, 15s, 20s
            log.info(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    # Apply stealth patches AFTER the first navigation.
    # In Docker, applying stealth init scripts before navigation
    # causes Chrome's DNS resolver to fail (ERR_NAME_NOT_RESOLVED).
    await _browser.apply_stealth_patches()

    await asyncio.sleep(3)

    logged_in = await _browser.is_logged_in()
    if not logged_in:
        # A distributable sidecar must start even before first login. The API
        # remains reachable and /status reports logged_in=false while the
        # user completes authentication in the visible managed browser.
        log.warning("%s login required — API starting in recovery mode", provider_name)

    if Config.PROVIDER == "claude":
        _client = ClaudeClient(page)
    else:
        _client = ChatGPTClient(page)

    set_client(_client, _browser)
    set_openai_client(_client)
    if Config.PROVIDER == "chatgpt":
        _lane_pool = ChatGPTLanePool(
            _browser,
            _client,
            lane_count=Config.LANE_COUNT,
            max_concurrency=Config.MAX_CONCURRENCY,
        )
        set_lane_pool(_lane_pool)
    log.info(
        "API server ready — browser launched, provider=%s, logged_in=%s, lanes=%s, concurrency=%s",
        provider_name,
        logged_in,
        Config.LANE_COUNT,
        Config.MAX_CONCURRENCY,
    )

    yield  # Server is running

    log.info("Shutting down — closing browser...")
    if _lane_pool is not None:
        await _lane_pool.close()
        _lane_pool = None
        set_lane_pool(None)
    await _browser.close()
    log.info("Browser closed")


app = FastAPI(
    title="CatGPT Gateway API",
    description=(
        "Browser automation API for ChatGPT and Claude. "
        "Sends messages via browser and returns responses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class RcaRouteGuardMiddleware:
    """Expose only the approved RCA surface when dedicated mode is enabled."""

    ALLOWED_PATHS = {
        "/docs",
        "/redoc",
        "/openapi.json",
        "/healthz",
        "/status",
        "/control/show-login",
        "/v1/models",
        "/v1/chat/completions",
    }

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            Config.RCA_MODE
            and scope["type"] == "http"
            and scope.get("method") != "OPTIONS"
            and scope.get("path") not in self.ALLOWED_PATHS
        ):
            response = JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "RCA_ROUTE_DISABLED",
                        "message": "This route is disabled in RCA mode.",
                    }
                },
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class RequestBodyLimitMiddleware:
    """Enforce the raw request-body cap before JSON parsing or dispatch."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length", b"")
        try:
            declared_size = int(content_length) if content_length else None
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > Config.MAX_REQUEST_BYTES:
            await self._reject(scope, receive, send)
            return

        messages: list[dict] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.request":
                total += len(message.get("body", b""))
                if total > Config.MAX_REQUEST_BYTES:
                    await self._reject(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message.get("type") == "http.disconnect":
                break

        index = 0

        async def replay_receive():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "REQUEST_TOO_LARGE",
                    "message": "Request body exceeded the hard input budget.",
                    "request_id": "",
                    "completion_status": "not_dispatched",
                    "provider_outcome": "not_dispatched",
                    "manual_action_required": False,
                }
            },
        )
        await response(scope, receive, send)


# ── Bearer Token Auth Middleware ────────────────────────────────
class BearerTokenMiddleware:
    """
    Pure ASGI middleware for Bearer token auth.

    Uses raw ASGI protocol instead of BaseHTTPMiddleware to avoid the
    Python 3.9 event-loop mismatch bug that corrupts asyncio.Lock
    when exceptions propagate through BaseHTTPMiddleware's task group.

    Skips auth for /docs, /openapi.json, and health-check paths.
    """

    OPEN_PATHS = {b"/docs", b"/redoc", b"/openapi.json", b"/healthz"}

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        token = Config.API_TOKEN
        if not token:
            await self.app(scope, receive, send)
            return

        path = (
            scope.get("path", "").encode()
            if isinstance(scope.get("path"), str)
            else scope.get("raw_path", b"")
        )
        # Also check the string path for comparison
        path_str = scope.get("path", "")
        if path_str in {"/docs", "/redoc", "/openapi.json", "/healthz"}:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth_value = headers.get(b"authorization", b"").decode()

        provided = ""
        if auth_value.startswith("Bearer "):
            provided = auth_value[7:]

        if provided != token:
            response = JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API token. Set Authorization: Bearer <API_TOKEN>",
                        "type": "auth_error",
                    }
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


app.add_middleware(BearerTokenMiddleware)
app.add_middleware(RequestBodyLimitMiddleware)
app.add_middleware(RcaRouteGuardMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(Config.CORS_ORIGINS),
    allow_credentials="*" not in Config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(router)
app.include_router(openai_router)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Unauthenticated health-check for Docker / load-balancers."""
    return {
        "status": "ok",
        "rca_mode": Config.RCA_MODE,
        "api_host": Config.API_HOST,
        "auth_required": bool(Config.API_TOKEN),
        "lane_count": Config.LANE_COUNT,
        "max_concurrency": Config.MAX_CONCURRENCY,
        "cors_origins": list(Config.CORS_ORIGINS),
        "provider_deadline_ms": Config.PROVIDER_DEADLINE_MS,
        "max_request_bytes": Config.MAX_REQUEST_BYTES,
        "max_output_bytes": Config.MAX_OUTPUT_BYTES,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info",
    )
