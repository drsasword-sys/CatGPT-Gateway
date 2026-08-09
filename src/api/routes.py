"""
API routes — FastAPI router for ChatGPT interaction.

Endpoints:
  POST /chat              Send a message in the current/new thread
  POST /thread/{id}/chat  Send a message in a specific thread
  POST /thread/new        Start a new conversation
  GET  /threads           List recent threads
  GET  /status            Health check + login status
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import re
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ImageInfoResponse,
    StatusResponse,
    ControlResponse,
    ThreadInfo,
    ThreadListResponse,
    ProjectCreateRequest,
    ProjectCreateResponse,
)
from src.browser.manager import BrowserManager
from src.chatgpt.client import ChatGPTClient
from src.chatgpt.project_client import ChatGPTProjectClient, ProjectSetupError
from src.claude.client import ClaudeClient
from src.log import setup_logging
from src.config import Config

log = setup_logging("api_routes")

router = APIRouter()

# Serialize browser access — single page, not thread-safe
_lock = asyncio.Lock()

# Global reference — set by the server on startup
_client: ChatGPTClient | ClaudeClient | None = None
_browser: BrowserManager | None = None


def set_client(client: ChatGPTClient | ClaudeClient, browser: BrowserManager) -> None:
    """Called by server.py to inject the client instance."""
    global _client, _browser
    _client = client
    _browser = browser


def _get_client():
    if _client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    return _client


def _build_response(result) -> ChatResponse:
    """Convert internal ChatResponse to API ChatResponse with image data."""
    images = [
        ImageInfoResponse(
            url=img.url,
            alt=img.alt,
            local_path=img.local_path,
            prompt_title=img.prompt_title,
        )
        for img in (result.images or [])
    ]
    return ChatResponse(
        message=result.message,
        thread_id=result.thread_id,
        response_time_ms=result.response_time_ms,
        images=images,
        has_images=result.has_images,
    )


# ── Chat ────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message in the current conversation."""
    client = _get_client()
    log.info(f"POST /chat — {len(req.message)} chars")

    async with _lock:
        try:
            result = await client.send_message(req.message)
            return _build_response(result)
        except Exception as e:
            log.error(f"Chat error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/thread/{thread_id}/chat", response_model=ChatResponse)
async def chat_in_thread(thread_id: str, req: ChatRequest) -> ChatResponse:
    """Send a message in a specific thread. Navigates to it first."""
    client = _get_client()
    log.info(f"POST /thread/{thread_id}/chat — {len(req.message)} chars")

    async with _lock:
        try:
            # Navigate to the thread if not already there
            current_tid = client._extract_thread_id()
            if current_tid != thread_id:
                await client.navigate_to_thread(thread_id)

            result = await client.send_message(req.message)
            return _build_response(result)
        except Exception as e:
            log.error(f"Thread chat error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/thread/new", response_model=ChatResponse)
async def new_thread(req: ChatRequest) -> ChatResponse:
    """Start a new conversation and send the first message."""
    client = _get_client()
    log.info(f"POST /thread/new — {len(req.message)} chars")

    async with _lock:
        try:
            await client.new_chat()
            result = await client.send_message(req.message)
            return _build_response(result)
        except Exception as e:
            log.error(f"New thread error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


# ── Threads ─────────────────────────────────────────────────────


@router.get("/threads", response_model=ThreadListResponse)
async def list_threads() -> ThreadListResponse:
    """List recent conversation threads from the sidebar."""
    client = _get_client()
    log.info("GET /threads")

    async with _lock:
        try:
            raw_threads = await client.list_threads()
            threads = [
                ThreadInfo(id=t["id"], title=t["title"], url=t["url"])
                for t in raw_threads
            ]
            return ThreadListResponse(threads=threads)
        except Exception as e:
            log.error(f"Threads list error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects", response_model=ProjectCreateResponse)
async def create_project(req: ProjectCreateRequest) -> ProjectCreateResponse:
    """Create a ChatGPT Web Project and optionally seed its visible context."""
    if Config.PROVIDER == "claude":
        raise HTTPException(status_code=501, detail="Projects are available only for ChatGPT Web")
    client = _get_client()
    upload_paths: list[str] = []
    async with _lock:
        try:
            upload_dir = Path(tempfile.gettempdir()) / "catgpt-project-files"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for file_info in req.files[:25]:
                filename = re.sub(r"[^A-Za-z0-9._-]", "_", str(file_info.get("filename") or "reference.txt"))
                encoded = file_info.get("data")
                if not encoded:
                    continue
                if len(str(encoded)) > 15 * 1024 * 1024:
                    raise ValueError("Project file payload is too large")
                path = upload_dir / f"{uuid4().hex}_{filename}"
                path.write_bytes(base64.b64decode(str(encoded), validate=True))
                upload_paths.append(str(path))
            result = await ChatGPTProjectClient(client).create_project(
                req.name,
                instructions=req.instructions,
                file_paths=upload_paths,
            )
            return ProjectCreateResponse(project_ref=result.project_ref, warnings=list(result.warnings))
        except ProjectSetupError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail="Invalid Project file payload") from exc
        except Exception as exc:
            log.error("Project creation error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            for path in upload_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass


# ── Status ──────────────────────────────────────────────────────


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Health check — returns login status and current thread."""
    try:
        client = _get_client()
        logged_in = await _browser.is_logged_in()
        tid = client._extract_thread_id()
        return StatusResponse(status="ok", logged_in=logged_in, current_thread=tid)
    except Exception:
        return StatusResponse(status="ok", logged_in=False, current_thread="")


@router.post("/control/show-login", response_model=ControlResponse)
async def show_login() -> ControlResponse:
    """Show the managed browser so the local user can sign in or recover."""
    if _browser is None:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    try:
        logged_in = await _browser.show_login()
        return ControlResponse(status="ok", logged_in=logged_in)
    except Exception as exc:
        log.error("Show-login control failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Managed browser is unavailable") from exc
