"""Content-free network completion tracker for one ChatGPT browser page."""

from __future__ import annotations

import time
from urllib.parse import urlsplit

from patchright.async_api import Page, Request, Response

from src.chatgpt.models import NetworkStreamStatus
from src.log import setup_logging

log = setup_logging("network")


class NetworkRecorder:
    """Correlate terminal transport events without retaining private payloads."""

    _WINDOW_SECONDS = 300.0

    def __init__(self, page: Page) -> None:
        self._page = page
        self._active = False
        self._armed_request_id: str | None = None
        self._armed_at = 0.0
        self._correlated_object_id: int | None = None
        self._endpoint_category: str | None = None
        self._stream_status = NetworkStreamStatus.UNAVAILABLE
        self._events: list[dict[str, object]] = []

    @property
    def stream_status(self) -> NetworkStreamStatus:
        """Current terminal state for the currently armed logical request."""
        return self._stream_status

    def start(self) -> None:
        """Register page listeners once."""
        if self._active:
            return
        self._page.on("request", self._on_request)
        self._page.on("response", self._on_response)
        self._page.on("requestfinished", self._on_request_finished)
        self._page.on("requestfailed", self._on_request_failed)
        self._active = True
        log.info("Network completion tracker started")

    def stop(self) -> None:
        """Stop accepting events; registered callbacks become no-ops."""
        self._active = False
        log.info("Network completion tracker stopped")

    def arm(self, request_id: str) -> None:
        """Begin a fresh correlation window immediately before browser send."""
        self._armed_request_id = request_id
        self._armed_at = time.monotonic()
        self._correlated_object_id = None
        self._endpoint_category = None
        self._stream_status = NetworkStreamStatus.UNAVAILABLE
        self._events = []

    def _is_in_window(self) -> bool:
        return bool(
            self._armed_request_id
            and (time.monotonic() - self._armed_at) <= self._WINDOW_SECONDS
        )

    @staticmethod
    def _category(request: Request) -> str | None:
        """Map a sanitized path to a stable category; never retain URL/query."""
        if str(getattr(request, "method", "")).upper() != "POST":
            return None
        path = urlsplit(str(getattr(request, "url", ""))).path.casefold()
        if path.endswith("/f/conversation") or path.endswith("/conversation"):
            return "conversation"
        return None

    def _record(self, event: str) -> None:
        self._events.append(
            {
                "request_id": self._armed_request_id,
                "endpoint_category": self._endpoint_category,
                "method": "POST",
                "url": self._endpoint_category,
                "event": event,
                "at_ms": int((time.monotonic() - self._armed_at) * 1000),
                "stream_status": self._stream_status.value,
            }
        )

    def _on_request(self, request: Request) -> None:
        if not self._active or not self._is_in_window():
            return
        category = self._category(request)
        if category is None:
            return
        if self._correlated_object_id is not None:
            self._stream_status = NetworkStreamStatus.CONFLICT
            self._record("conflicting_request")
            return
        self._correlated_object_id = id(request)
        self._endpoint_category = category
        self._stream_status = NetworkStreamStatus.OPEN
        self._record("request")

    def _on_response(self, response: Response) -> None:
        if not self._active or not self._is_in_window():
            return
        request = getattr(response, "request", None)
        if request is None or self._category(request) is None:
            return
        if id(request) != self._correlated_object_id:
            self._stream_status = NetworkStreamStatus.CONFLICT
            self._record("stale_response")
            return
        # Headers/status are not a terminal stream event.
        self._record("response_headers")

    def _on_request_finished(self, request: Request) -> None:
        self._on_terminal(request, NetworkStreamStatus.CLOSED, "requestfinished")

    def _on_request_failed(self, request: Request) -> None:
        self._on_terminal(request, NetworkStreamStatus.FAILED, "requestfailed")

    def _on_terminal(
        self,
        request: Request,
        terminal: NetworkStreamStatus,
        event: str,
    ) -> None:
        if not self._active or not self._is_in_window():
            return
        if self._category(request) is None:
            return
        if id(request) != self._correlated_object_id:
            self._stream_status = NetworkStreamStatus.CONFLICT
            self._record(f"stale_{event}")
            return
        self._stream_status = terminal
        self._record(event)

    def get_captured(self) -> list[dict[str, object]]:
        """Return sanitized metadata for tests/diagnostics."""
        return [dict(event) for event in self._events]

    def clear(self) -> None:
        """Clear all correlation state and sanitized events."""
        self._armed_request_id = None
        self._correlated_object_id = None
        self._endpoint_category = None
        self._stream_status = NetworkStreamStatus.UNAVAILABLE
        self._events = []
