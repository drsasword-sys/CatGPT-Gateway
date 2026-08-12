from __future__ import annotations

import hashlib
import unittest
import uuid
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import openai_routes
from src.chatgpt.client import (
    ChatGPTCompletionError,
    ChatGPTProviderStateError,
)
from src.chatgpt.models import (
    ChatResponse,
    CompletionEvidence,
    CompletionResult,
    CompletionStatus,
    NetworkStreamStatus,
)


def _request_id() -> str:
    return f"rca-{uuid.uuid4()}"


def _payload(request_id: str) -> dict:
    return {
        "model": "catgpt-browser",
        "messages": [{"role": "user", "content": "synthetic input"}],
        "metadata": {
            "request_id": request_id,
            "conversation_mode": "fresh",
            "consumer": "rca-engine-local-v1",
        },
    }


def _complete_response() -> ChatResponse:
    message = "synthetic completed output"
    return ChatResponse(
        message=message,
        thread_id="thread-verified",
        completion=CompletionResult(
            status=CompletionStatus.COMPLETE,
            evidence=CompletionEvidence(
                evidence="new_turn_final_action_stop_absent_text_stable",
                turn_signature="2:verified-turn",
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


class _FakeClient:
    def __init__(self, outcome=None, *, fresh_error: Exception | None = None) -> None:
        self.outcome = outcome or _complete_response()
        self.fresh_error = fresh_error
        self.events: list[tuple[str, str | None]] = []

    async def new_chat(self) -> None:
        self.events.append(("new_chat", None))
        if self.fresh_error:
            raise self.fresh_error

    async def send_message(self, _prompt: str, **kwargs) -> ChatResponse:
        self.events.append(("send", kwargs.get("request_id")))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class OpenAiCompletionContractTests(unittest.TestCase):
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

    def test_complete_response_has_stop_request_id_evidence_and_estimated_usage(
        self,
    ) -> None:
        request_id = _request_id()
        fake = _FakeClient()
        openai_routes.set_openai_client(fake)

        response = self.http.post("/v1/chat/completions", json=_payload(request_id))

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("stop", body["choices"][0]["finish_reason"])
        self.assertEqual(request_id, body["metadata"]["request_id"])
        self.assertEqual("estimated", body["metadata"]["usage_source"])
        self.assertEqual(
            "complete",
            body["metadata"]["provider_completion"]["status"],
        )
        self.assertEqual(
            hashlib.sha256("synthetic completed output".encode("utf-8")).hexdigest(),
            body["metadata"]["provider_completion"]["output_sha256"],
        )
        self.assertEqual(
            [("new_chat", None), ("send", request_id)],
            fake.events,
        )

    def test_fresh_chat_failure_never_sends_prompt(self) -> None:
        fake = _FakeClient(fresh_error=RuntimeError("secret browser detail"))
        openai_routes.set_openai_client(fake)

        response = self.http.post("/v1/chat/completions", json=_payload(_request_id()))

        self.assertEqual(503, response.status_code)
        self.assertEqual("CHATGPT_FRESH_CHAT_FAILED", response.json()["error"]["code"])
        self.assertEqual([("new_chat", None)], fake.events)
        self.assertNotIn("secret browser detail", response.text)

    def test_timeout_and_stale_are_structured_non_success(self) -> None:
        cases = (
            (CompletionStatus.TIMEOUT, 504, "CHATGPT_RESPONSE_TIMEOUT"),
            (CompletionStatus.STALE, 409, "CHATGPT_STALE_RESPONSE"),
            (CompletionStatus.INCOMPLETE, 422, "CHATGPT_OUTPUT_INCOMPLETE"),
            (CompletionStatus.SELECTOR_DRIFT, 503, "CHATGPT_SELECTOR_DRIFT"),
        )
        for status, http_status, code in cases:
            with self.subTest(status=status):
                error = ChatGPTCompletionError(CompletionResult(status=status))
                openai_routes.set_openai_client(_FakeClient(outcome=error))

                response = self.http.post(
                    "/v1/chat/completions",
                    json=_payload(_request_id()),
                )

                self.assertEqual(http_status, response.status_code)
                body = response.json()
                self.assertEqual(code, body["error"]["code"])
                self.assertNotIn("choices", body)

    def test_login_and_rate_limit_are_typed_non_success(self) -> None:
        cases = (
            ("login_required", 401, "CHATGPT_LOGIN_REQUIRED"),
            ("rate_limited", 429, "CHATGPT_RATE_LIMITED"),
        )
        for state, http_status, code in cases:
            with self.subTest(state=state):
                error = ChatGPTProviderStateError(state)
                openai_routes.set_openai_client(_FakeClient(outcome=error))

                response = self.http.post(
                    "/v1/chat/completions",
                    json=_payload(_request_id()),
                )

                self.assertEqual(http_status, response.status_code)
                error_body = response.json()["error"]
                self.assertEqual(code, error_body["code"])
                self.assertEqual(
                    "not_dispatched" if state == "login_required" else "rejected",
                    error_body["provider_outcome"],
                )


if __name__ == "__main__":
    unittest.main()
