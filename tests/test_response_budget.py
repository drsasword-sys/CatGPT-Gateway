from __future__ import annotations

import asyncio
import hashlib
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import openai_routes
from src.api.openai_schemas import ChatCompletionRequest
from src.api.server import RequestBodyLimitMiddleware
from src.chatgpt.models import (
    ChatResponse,
    CompletionEvidence,
    CompletionResult,
    CompletionStatus,
    NetworkStreamStatus,
)
from src.config import Config


def _payload(content: str, **extra) -> dict:
    body = {
        "model": "catgpt-browser",
        "messages": [{"role": "user", "content": content}],
        "metadata": {
            "request_id": f"rca-{uuid.uuid4()}",
            "conversation_mode": "fresh",
            "consumer": "rca-engine-local-v1",
        },
    }
    body.update(extra)
    return body


def _completion(message: str) -> ChatResponse:
    return ChatResponse(
        message=message,
        thread_id="thread-budget",
        completion=CompletionResult(
            status=CompletionStatus.COMPLETE,
            evidence=CompletionEvidence(
                evidence="new_turn_final_action_stop_absent_text_stable",
                turn_signature="2:budget-turn",
                stable_samples=3,
                stable_for_ms=3000,
                network_stream=NetworkStreamStatus.CLOSED,
                output_chars=len(message),
                output_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
                final_action_present=True,
                stop_button_visible=False,
            ),
        ),
    )


class _BudgetClient:
    def __init__(
        self, outcome: ChatResponse | None = None, *, block: bool = False
    ) -> None:
        self.outcome = outcome or _completion("short output")
        self.block = block
        self.events: list[str] = []

    async def new_chat(self) -> None:
        self.events.append("new_chat")

    async def send_message(self, _prompt: str, **_kwargs) -> ChatResponse:
        self.events.append("send")
        if self.block:
            await asyncio.Event().wait()
        return self.outcome


class ResponseBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        test_app = FastAPI()
        test_app.include_router(openai_routes.openai_router)
        self.http = TestClient(test_app)
        self.previous_client = openai_routes._client
        self.previous_pool = openai_routes._lane_pool
        openai_routes._lane_pool = None
        openai_routes._lock = None
        openai_routes._thread_message_count = 0
        openai_routes._last_response_time = 0.0
        self.provider_patch = patch.object(openai_routes.Config, "PROVIDER", "chatgpt")
        self.provider_patch.start()

    def tearDown(self) -> None:
        self.provider_patch.stop()
        openai_routes._client = self.previous_client
        openai_routes._lane_pool = self.previous_pool
        openai_routes._lock = None

    def test_input_below_cap_succeeds_and_above_cap_never_dispatches(self) -> None:
        below = _BudgetClient()
        openai_routes.set_openai_client(below)
        with patch.object(openai_routes.Config, "MAX_REQUEST_BYTES", 512, create=True):
            response = self.http.post("/v1/chat/completions", json=_payload("small"))
        self.assertEqual(200, response.status_code)
        self.assertEqual(["new_chat", "send"], below.events)

        above = _BudgetClient()
        openai_routes.set_openai_client(above)
        with patch.object(openai_routes.Config, "MAX_REQUEST_BYTES", 512, create=True):
            response = self.http.post("/v1/chat/completions", json=_payload("x" * 1000))
        self.assertEqual(413, response.status_code)
        self.assertEqual("REQUEST_TOO_LARGE", response.json()["error"]["code"])
        self.assertEqual([], above.events)

    def test_output_over_cap_has_no_partial_success_content(self) -> None:
        secret_output = "OUTPUT_CANARY_" + ("x" * 128)
        fake = _BudgetClient(_completion(secret_output))
        openai_routes.set_openai_client(fake)

        with patch.object(openai_routes.Config, "MAX_OUTPUT_BYTES", 64, create=True):
            response = self.http.post("/v1/chat/completions", json=_payload("small"))

        self.assertEqual(502, response.status_code)
        self.assertEqual("PROVIDER_OUTPUT_TOO_LARGE", response.json()["error"]["code"])
        self.assertNotIn(secret_output, response.text)
        self.assertNotIn("choices", response.json())

    def test_deadline_returns_504_without_partial_success(self) -> None:
        fake = _BudgetClient(block=True)
        openai_routes.set_openai_client(fake)

        with patch.object(openai_routes.Config, "PROVIDER_DEADLINE_MS", 5, create=True):
            response = self.http.post("/v1/chat/completions", json=_payload("small"))

        self.assertEqual(504, response.status_code)
        self.assertEqual("PROVIDER_TIMEOUT", response.json()["error"]["code"])
        self.assertNotIn("choices", response.json())

    def test_max_tokens_is_rejected_before_fresh_chat(self) -> None:
        fake = _BudgetClient()
        openai_routes.set_openai_client(fake)

        response = self.http.post(
            "/v1/chat/completions",
            json=_payload("small", max_tokens=100),
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual("UNSUPPORTED_PARAMETER", response.json()["error"]["code"])
        self.assertEqual([], fake.events)

    def test_tools_are_rejected_for_rca_before_fresh_chat(self) -> None:
        fake = _BudgetClient()
        openai_routes.set_openai_client(fake)
        payload = _payload("small")
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "unsafe_tool_path",
                    "description": "must not dispatch",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response = self.http.post("/v1/chat/completions", json=payload)

        self.assertEqual(400, response.status_code)
        self.assertEqual("UNSUPPORTED_PARAMETER", response.json()["error"]["code"])
        self.assertEqual([], fake.events)

    def test_multimodal_attachment_is_rejected_before_download_or_dispatch(
        self,
    ) -> None:
        fake = _BudgetClient()
        openai_routes.set_openai_client(fake)
        payload = _payload("small")
        payload["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "synthetic"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://127.0.0.1/private"},
                    },
                ],
            }
        ]

        with patch.object(
            openai_routes,
            "_download_file",
            new=AsyncMock(),
        ) as download:
            response = self.http.post("/v1/chat/completions", json=payload)

        self.assertEqual(400, response.status_code)
        self.assertEqual("UNSUPPORTED_PARAMETER", response.json()["error"]["code"])
        download.assert_not_awaited()
        self.assertEqual([], fake.events)

    def test_raw_body_middleware_enforces_exact_byte_boundary(self) -> None:
        app = FastAPI()
        app.add_middleware(RequestBodyLimitMiddleware)

        @app.post("/echo")
        async def echo():
            return {"status": "accepted"}

        http = TestClient(app)
        with patch.object(Config, "MAX_REQUEST_BYTES", 8):
            below = http.post("/echo", content=b"12345678")
            above = http.post("/echo", content=b"123456789")

        self.assertEqual(200, below.status_code)
        self.assertEqual(413, above.status_code)
        self.assertEqual("REQUEST_TOO_LARGE", above.json()["error"]["code"])

    def test_deadline_includes_time_waiting_for_browser_lock(self) -> None:
        async def scenario():
            openai_routes.set_openai_client(_BudgetClient())
            lock = asyncio.Lock()
            await lock.acquire()
            openai_routes._lock = lock
            request = ChatCompletionRequest.model_validate(_payload("small"))
            with patch.object(
                openai_routes.Config,
                "PROVIDER_DEADLINE_MS",
                5,
                create=True,
            ):
                try:
                    return await asyncio.wait_for(
                        openai_routes.create_chat_completion(request),
                        timeout=0.1,
                    )
                finally:
                    lock.release()

        response = asyncio.run(scenario())
        self.assertEqual(504, response.status_code)
        self.assertIn("PROVIDER_TIMEOUT", response.body.decode())


if __name__ == "__main__":
    unittest.main()
