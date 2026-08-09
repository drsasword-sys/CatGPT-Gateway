"""Per-lane ChatGPT browser sessions for safe project fan-out.

The ChatGPT web UI is not safe to drive concurrently on one page.  This pool
uses one persistent browser context (shared login) with one page and lock per
lane.  Requests in the same lane remain ordered; different lanes may run at
the same time, subject to the configured pool limit.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from patchright.async_api import Page

from src.browser.manager import BrowserManager
from src.chatgpt.client import ChatGPTClient
from src.chatgpt.models import ChatResponse
from src.config import Config
from src.log import setup_logging

log = setup_logging("chatgpt_lane_pool")


@dataclass
class LaneSession:
    lane_number: int
    client: ChatGPTClient
    lock: asyncio.Lock
    thread_id: str = ""
    project_ref: str = ""
    last_response_at: float = 0.0
    message_count: int = 0


class ChatGPTLanePool:
    """Lazily create up to ten isolated pages in one logged-in context."""

    def __init__(
        self,
        browser: BrowserManager,
        initial_client: ChatGPTClient,
        *,
        lane_count: int = 10,
        max_concurrency: int = 10,
        min_message_gap_seconds: float = 3.0,
        max_messages_per_chat: int = 8,
    ) -> None:
        self.browser = browser
        self.lane_count = max(1, min(10, int(lane_count)))
        self.max_concurrency = max(1, min(self.lane_count, int(max_concurrency)))
        self.min_message_gap_seconds = max(0.0, float(min_message_gap_seconds))
        self.max_messages_per_chat = max(1, int(max_messages_per_chat))
        self._startup_lock = asyncio.Lock()
        self._navigation_lock = asyncio.Lock()
        self._last_navigation_at = 0.0
        self._sidebar_guard_pages: set[int] = set()
        self._capacity = asyncio.Semaphore(self.max_concurrency)
        self._sessions: dict[int, LaneSession] = {
            1: LaneSession(1, initial_client, asyncio.Lock())
        }

    @property
    def sessions(self) -> dict[int, LaneSession]:
        return dict(self._sessions)

    async def configure(self, lane_count: int) -> None:
        """Set the requested pool size before pages are materialized."""
        requested = max(1, min(10, int(lane_count)))
        async with self._startup_lock:
            if len(self._sessions) > 1 and requested != self.lane_count:
                raise ValueError("lane_count cannot change after the lane pool has started")
            if requested == self.lane_count:
                return
            self.lane_count = requested
            self.max_concurrency = min(self.max_concurrency, requested)
            self._capacity = asyncio.Semaphore(self.max_concurrency)

    async def start(self) -> None:
        """Create missing pages only once; the initial page is lane 1."""
        async with self._startup_lock:
            await self._install_sidebar_history_guard(self._sessions[1].client.page)
            while len(self._sessions) < self.lane_count:
                lane_number = len(self._sessions) + 1
                page: Page = await self.browser.new_page()
                await self._install_sidebar_history_guard(page)
                await page.goto(Config.CHATGPT_URL, wait_until="domcontentloaded")
                client = ChatGPTClient(page)
                self._sessions[lane_number] = LaneSession(lane_number, client, asyncio.Lock())
                log.info("Lane %s ready (%s/%s)", lane_number, len(self._sessions), self.lane_count)
                if lane_number < self.lane_count and hasattr(page, "route"):
                    await asyncio.sleep(Config.LANE_BOOTSTRAP_GAP_SECONDS)

    async def send(
        self,
        lane_number: int,
        prompt: str,
        *,
        project_ref: str = "",
        chat_ref: str = "",
        lane_count: int | None = None,
        image_paths: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> ChatResponse:
        """Send one prompt to a lane while preserving its conversation order."""
        lane_number = int(lane_number)
        if lane_number < 1 or lane_number > self.lane_count:
            raise ValueError(f"lane must be between 1 and {self.lane_count}")
        if lane_count is not None:
            await self.configure(lane_count)
            if lane_number > self.lane_count:
                raise ValueError(f"lane must be between 1 and {self.lane_count}")
        await self.start()
        session = self._sessions[lane_number]

        async with session.lock:
            async with self._capacity:
                if project_ref and session.project_ref != project_ref:
                    await self._navigate_with_gap(session.client.navigate_to_project, project_ref)
                    session.project_ref = project_ref
                    session.thread_id = ""
                    session.message_count = 0

                if chat_ref and session.thread_id != chat_ref:
                    await self._navigate_with_gap(session.client.navigate_to_thread, chat_ref)
                    session.thread_id = chat_ref
                    session.message_count = 0
                elif not session.thread_id:
                    # Sending from the Project landing page creates a new chat
                    # in that Project, avoiding the global New Chat route.
                    if not project_ref:
                        await session.client.new_chat()

                if session.last_response_at:
                    elapsed = time.monotonic() - session.last_response_at
                    if elapsed < self.min_message_gap_seconds:
                        await asyncio.sleep(self.min_message_gap_seconds - elapsed)

                if session.message_count >= self.max_messages_per_chat:
                    if project_ref:
                        await self._navigate_with_gap(session.client.navigate_to_project, project_ref)
                    else:
                        await session.client.new_chat()
                    session.thread_id = ""
                    session.message_count = 0

                result = await session.client.send_message(
                    prompt,
                    image_paths=image_paths,
                    file_paths=file_paths,
                )
                session.thread_id = result.thread_id or session.thread_id
                session.message_count += 1
                session.last_response_at = time.monotonic()
                return result

    async def _navigate_with_gap(self, navigate, target: str) -> None:
        """Serialize UI navigations so sidebar requests do not burst."""
        async with self._navigation_lock:
            elapsed = time.monotonic() - self._last_navigation_at
            gap = Config.LANE_BOOTSTRAP_GAP_SECONDS
            if self._last_navigation_at and elapsed < gap:
                await asyncio.sleep(gap - elapsed)
            await navigate(target)
            self._last_navigation_at = time.monotonic()

    async def _install_sidebar_history_guard(self, page: Page) -> None:
        """Abort noisy global history refreshes; lane prompts use direct URLs."""
        if not hasattr(page, "route"):
            return
        page_key = id(page)
        if page_key in self._sidebar_guard_pages:
            return

        async def abort_history(route) -> None:
            await route.abort()

        try:
            await page.route(
                re.compile(r"/backend-api/conversations\?offset="),
                abort_history,
            )
            self._sidebar_guard_pages.add(page_key)
        except Exception as exc:
            log.debug("Could not install sidebar history guard: %s", exc)

    async def close(self) -> None:
        """Close only extra pages; BrowserManager owns the context lifecycle."""
        for lane_number, session in list(self._sessions.items()):
            if lane_number == 1:
                continue
            try:
                await session.client.page.close()
            except Exception as exc:
                log.warning("Could not close lane %s page: %s", lane_number, exc)
