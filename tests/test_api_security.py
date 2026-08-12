from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import openai_routes
from src.api.openai_schemas import ChatCompletionRequest
from src.api.server import (
    BearerTokenMiddleware,
    RcaRouteGuardMiddleware,
    healthz,
    validate_rca_runtime_security,
)
from src.chatgpt.client import ChatGPTClient
from src.chatgpt.models import (
    CompletionEvidence,
    CompletionResult,
    CompletionStatus,
)
from src.config import Config


class ApiSecurityTests(unittest.TestCase):
    def _security_config(self, **overrides):
        values = {
            "RCA_MODE": True,
            "PROVIDER": "chatgpt",
            "CHATGPT_URL": "https://chatgpt.com",
            "API_HOST": "127.0.0.1",
            "API_TOKEN": "test-token",
            "LANE_COUNT": 1,
            "MAX_CONCURRENCY": 1,
            "CORS_ORIGINS": (),
            "PROVIDER_DEADLINE_MS": 1000,
            "COMPLETION_STABLE_SAMPLES": 3,
            "COMPLETION_STABLE_MS": 3000,
        }
        values.update(overrides)
        stack = []
        for key, value in values.items():
            current = patch.object(Config, key, value, create=True)
            current.start()
            stack.append(current)
        return stack

    def test_rca_runtime_rejects_unsafe_auth_host_cors_and_concurrency(self) -> None:
        cases = (
            ({"API_TOKEN": ""}, "API token"),
            ({"API_HOST": "0.0.0.0"}, "127.0.0.1"),
            ({"CORS_ORIGINS": ("*",)}, "CORS"),
            ({"LANE_COUNT": 2}, "lane"),
            ({"MAX_CONCURRENCY": 2}, "concurrency"),
            ({"PROVIDER": "claude"}, "ChatGPT provider"),
            (
                {"CHATGPT_URL": "https://chatgpt.com.evil.example"},
                "ChatGPT URL",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                patches = self._security_config(**overrides)
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        validate_rca_runtime_security()
                finally:
                    for current in reversed(patches):
                        current.stop()

    def test_rca_runtime_accepts_safe_effective_values(self) -> None:
        patches = self._security_config()
        try:
            validate_rca_runtime_security()
            metadata = asyncio.run(healthz())
        finally:
            for current in reversed(patches):
                current.stop()

        self.assertTrue(metadata["rca_mode"])
        self.assertTrue(metadata["auth_required"])
        self.assertNotIn("test-token", repr(metadata))
        self.assertEqual(1, metadata["lane_count"])
        self.assertEqual(1, metadata["max_concurrency"])

    def test_rca_runtime_disables_alternate_provider_routes(self) -> None:
        app = FastAPI()
        app.add_middleware(RcaRouteGuardMiddleware)

        @app.post("/chat")
        async def legacy_chat():
            return {"status": "must-not-run"}

        @app.post("/v1/responses")
        async def responses_api():
            return {"status": "must-not-run"}

        @app.post("/v1/chat/completions")
        async def rca_chat():
            return {"status": "allowed"}

        http = TestClient(app)
        with patch.object(Config, "RCA_MODE", True):
            legacy = http.post("/chat")
            responses = http.post("/v1/responses")
            allowed = http.post("/v1/chat/completions")

        self.assertEqual(404, legacy.status_code)
        self.assertEqual("RCA_ROUTE_DISABLED", legacy.json()["error"]["code"])
        self.assertEqual(404, responses.status_code)
        self.assertEqual(200, allowed.status_code)

    def test_rca_runtime_requires_strict_consumer_metadata(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "catgpt-browser",
                "messages": [{"role": "user", "content": "synthetic"}],
                "metadata": {},
            }
        )
        with patch.object(Config, "RCA_MODE", True):
            response = asyncio.run(openai_routes.create_chat_completion(request))

        self.assertEqual(400, response.status_code)
        self.assertIn("INVALID_RCA_METADATA", response.body.decode())

    def test_bearer_middleware_rejects_missing_and_wrong_tokens(self) -> None:
        app = FastAPI()
        app.add_middleware(BearerTokenMiddleware)

        @app.post("/protected")
        async def protected():
            return {"status": "ok"}

        http = TestClient(app)
        with patch.object(Config, "API_TOKEN", "gateway-test-secret"):
            missing = http.post("/protected")
            wrong = http.post(
                "/protected",
                headers={"Authorization": "Bearer wrong"},
            )
            accepted = http.post(
                "/protected",
                headers={"Authorization": "Bearer gateway-test-secret"},
            )

        self.assertEqual(401, missing.status_code)
        self.assertEqual(401, wrong.status_code)
        self.assertEqual(200, accepted.status_code)
        self.assertNotIn("gateway-test-secret", missing.text + wrong.text)

    def test_canary_prompt_output_and_exception_do_not_reach_logs_or_error(
        self,
    ) -> None:
        prompt_canary = "PROMPT_CANARY_DO_NOT_LOG"
        output_canary = "OUTPUT_CANARY_DO_NOT_LOG"
        exception_canary = "EXCEPTION_CANARY_DO_NOT_LOG"
        request_id = f"rca-{uuid.uuid4()}"

        test_app = FastAPI()
        test_app.include_router(openai_routes.openai_router)
        http = TestClient(test_app)
        previous_client = openai_routes._client
        previous_pool = openai_routes._lane_pool
        openai_routes._lane_pool = None
        openai_routes._lock = None
        openai_routes._last_response_time = 0.0

        class FailingClient:
            async def new_chat(self) -> None:
                return None

            async def send_message(self, _prompt: str, **_kwargs):
                raise RuntimeError(exception_canary)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        openai_routes.log.addHandler(handler)
        try:
            openai_routes.set_openai_client(FailingClient())
            with patch.object(openai_routes.Config, "PROVIDER", "chatgpt"):
                response = http.post(
                    "/v1/chat/completions",
                    json={
                        "model": "catgpt-browser",
                        "messages": [{"role": "user", "content": prompt_canary}],
                        "metadata": {
                            "request_id": request_id,
                            "conversation_mode": "fresh",
                            "consumer": "rca-engine-local-v1",
                        },
                    },
                )
        finally:
            openai_routes.log.removeHandler(handler)
            openai_routes._client = previous_client
            openai_routes._lane_pool = previous_pool
            openai_routes._lock = None

        captured = stream.getvalue() + response.text
        self.assertNotIn(prompt_canary, captured)
        self.assertNotIn(output_canary, captured)
        self.assertNotIn(exception_canary, captured)

    def test_client_does_not_log_prompt_or_output_prefixes(self) -> None:
        prompt_canary = "PROMPT_CANARY_CLIENT"
        output_canary = "OUTPUT_CANARY_CLIENT"
        client = object.__new__(ChatGPTClient)
        page = AsyncMock()
        page.keyboard = AsyncMock()
        client._page = page
        client._network_recorder = None
        client._detect_page_error = AsyncMock(return_value=None)
        client._dismiss_overlays = AsyncMock()
        client._find_selector = AsyncMock(return_value="#prompt")
        client._click_send = AsyncMock(return_value=True)
        client._upload_files = AsyncMock()
        client._extract_thread_id = lambda: "thread-safe-log"
        completion = CompletionResult(
            status=CompletionStatus.COMPLETE,
            evidence=CompletionEvidence(
                evidence="new_turn_final_action_stop_absent_text_stable",
                turn_signature="2:safe-log-turn",
                stable_samples=3,
                stable_for_ms=3000,
                output_chars=len(output_canary),
                output_sha256=hashlib.sha256(output_canary.encode("utf-8")).hexdigest(),
                final_action_present=True,
                stop_button_visible=False,
            ),
        )

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        from src.chatgpt import client as client_module

        client_module.log.addHandler(handler)
        try:
            with (
                patch("src.chatgpt.client.random_delay", new=AsyncMock()),
                patch("src.chatgpt.client.human_type", new=AsyncMock()),
                patch(
                    "src.chatgpt.client.count_assistant_messages",
                    new=AsyncMock(side_effect=[0, 1]),
                ),
                patch(
                    "src.chatgpt.client.get_latest_assistant_turn_signature",
                    new=AsyncMock(return_value="1:old-turn"),
                ),
                patch(
                    "src.chatgpt.client.wait_for_response_complete",
                    new=AsyncMock(return_value=completion),
                ),
                patch(
                    "src.chatgpt.client.extract_images_from_response",
                    new=AsyncMock(return_value=[]),
                ),
                patch(
                    "src.chatgpt.client.extract_last_response_via_copy",
                    new=AsyncMock(return_value=output_canary),
                ),
                patch(
                    "src.chatgpt.client.revalidate_completion_evidence",
                    new=AsyncMock(return_value=completion),
                ),
                patch("src.chatgpt.client.asyncio.sleep", new=AsyncMock()),
            ):
                asyncio.run(client.send_message(prompt_canary))
        finally:
            client_module.log.removeHandler(handler)

        captured = stream.getvalue()
        self.assertNotIn(prompt_canary, captured)
        self.assertNotIn(output_canary, captured)


if __name__ == "__main__":
    unittest.main()
