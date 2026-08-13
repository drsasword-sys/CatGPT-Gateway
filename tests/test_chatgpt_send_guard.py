from __future__ import annotations

import asyncio
from contextlib import nullcontext
import hashlib
import unittest
from unittest.mock import AsyncMock, patch

from src.chatgpt import detector
from src.chatgpt.client import ChatGPTClient, ChatGPTCompletionError
from src.chatgpt.models import (
    CompletionEvidence,
    CompletionResult,
    CompletionStatus,
    NetworkStreamStatus,
)
from src.network_recorder import NetworkRecorder


class _StreamingPage:
    def __init__(self) -> None:
        self.screenshot = AsyncMock()

    async def evaluate(self, script: str):
        if "conversation-turn-" in script:
            return {
                "found": True,
                "index": 2,
                "signature": "2:new-turn",
                "hasCopyButton": True,
                "hasImage": False,
                "text": "Response is still growing",
            }
        if "stop-button" in script:
            return True
        return None


class _EventPage:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}

    def on(self, event: str, handler) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, payload) -> None:
        for handler in self.handlers.get(event, []):
            handler(payload)


class _Request:
    def __init__(self, url: str, method: str = "POST") -> None:
        self.url = url
        self.method = method
        self.resource_type = "fetch"
        self.failure = None


class _Response:
    def __init__(self, request: _Request) -> None:
        self.request = request
        self.url = request.url
        self.status = 200


class ChatGptSendGuardTests(unittest.TestCase):
    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _client_ready_to_send(self) -> ChatGPTClient:
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
        client._extract_thread_id = lambda: "thread-test"
        return client

    def test_timeout_never_returns_chat_response(self) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()
            timeout = CompletionResult(status=CompletionStatus.TIMEOUT)
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
                    new=AsyncMock(return_value=timeout),
                ),
            ):
                with self.assertRaises(ChatGPTCompletionError) as raised:
                    await client.send_message("private prompt")
            self.assertEqual(CompletionStatus.TIMEOUT, raised.exception.result.status)

        asyncio.run(scenario())

    def test_output_budget_breach_stops_visible_generation(self) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()
            stop_button = AsyncMock()
            stop_button.is_visible.return_value = True
            client._page.query_selector.return_value = stop_button
            over_budget = CompletionResult(
                status=CompletionStatus.OUTPUT_TOO_LARGE,
                evidence=CompletionEvidence(
                    turn_signature="2:large-turn",
                    output_chars=999,
                    stop_button_visible=True,
                ),
            )
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
                    new=AsyncMock(return_value=over_budget),
                ),
            ):
                with self.assertRaises(ChatGPTCompletionError):
                    await client.send_message("synthetic prompt")
            stop_button.click.assert_awaited_once()

        asyncio.run(scenario())

    def test_fresh_chat_verification_rejects_existing_turns(self) -> None:
        async def scenario() -> None:
            client = object.__new__(ChatGPTClient)
            client._page = AsyncMock()
            client._page.evaluate.return_value = 1
            client._wait_for_chat_input = AsyncMock()
            with self.assertRaisesRegex(RuntimeError, "not fresh"):
                await client._verify_fresh_chat()
            client._wait_for_chat_input.assert_not_awaited()

        asyncio.run(scenario())

    def test_new_chat_does_not_skip_when_legacy_dom_hides_message_roles(self) -> None:
        async def scenario() -> None:
            client = object.__new__(ChatGPTClient)
            page = AsyncMock()
            page.url = "https://chatgpt.com/"

            async def evaluate(script: str, *args):
                # Simulate the UI drift seen in the live canary: the old
                # conversation-turn selector is empty, while role markers
                # still expose the existing conversation.
                if "window.location.href" in script:
                    return None
                if "data-message-author-role" in script:
                    return 1
                if "conversation-turn-" in script:
                    return 0
                return None

            page.evaluate.side_effect = evaluate
            page.query_selector.return_value = None
            client._page = page
            client._detect_page_error = AsyncMock(return_value=None)
            client._verify_fresh_chat = AsyncMock()

            await client.new_chat()

            client._verify_fresh_chat.assert_awaited_once()
            self.assertTrue(
                any("window.location.href" in call.args[0] for call in page.evaluate.call_args_list)
            )

        asyncio.run(scenario())

    def test_conversation_marker_count_uses_role_fallbacks(self) -> None:
        async def scenario() -> None:
            client = object.__new__(ChatGPTClient)
            client._page = AsyncMock()
            client._page.evaluate.return_value = 2

            self.assertEqual(2, await client._conversation_marker_count())
            script = client._page.evaluate.call_args.args[0]
            self.assertIn('data-message-author-role="assistant"', script)
            self.assertIn('data-turn="assistant"', script)
            self.assertIn('article', script)

        asyncio.run(scenario())

    def test_extracted_turn_must_match_completion_evidence(self) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()
            complete = CompletionResult(
                status=CompletionStatus.COMPLETE,
                evidence=CompletionEvidence(
                    evidence="new_turn_final_action_stop_absent_text_stable",
                    turn_signature="2:new-turn",
                    stable_samples=3,
                    stable_for_ms=3000,
                    output_chars=14,
                    output_sha256=self._sha256("Finished reply"),
                    final_action_present=True,
                    stop_button_visible=False,
                ),
            )
            with (
                patch("src.chatgpt.client.random_delay", new=AsyncMock()),
                patch("src.chatgpt.client.human_type", new=AsyncMock()),
                patch(
                    "src.chatgpt.client.count_assistant_messages",
                    new=AsyncMock(side_effect=[0, 1]),
                ),
                patch(
                    "src.chatgpt.client.get_latest_assistant_turn_signature",
                    new=AsyncMock(side_effect=["1:old-turn", "3:wrong-turn"]),
                ),
                patch(
                    "src.chatgpt.client.wait_for_response_complete",
                    new=AsyncMock(return_value=complete),
                ),
                patch(
                    "src.chatgpt.client.extract_images_from_response",
                    new=AsyncMock(return_value=[]),
                ),
                patch(
                    "src.chatgpt.client.extract_last_response_via_copy",
                    new=AsyncMock(return_value="Finished reply"),
                ),
            ):
                with self.assertRaises(ChatGPTCompletionError) as raised:
                    await client.send_message("private prompt")
            self.assertEqual(CompletionStatus.STALE, raised.exception.result.status)

        asyncio.run(scenario())

    def test_empty_extraction_is_incomplete_not_success(self) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()
            complete = CompletionResult(
                status=CompletionStatus.COMPLETE,
                evidence=CompletionEvidence(
                    evidence="new_turn_final_action_stop_absent_text_stable",
                    turn_signature="2:new-turn",
                    stable_samples=3,
                    stable_for_ms=3000,
                    output_chars=14,
                    output_sha256=self._sha256("Finished reply"),
                    final_action_present=True,
                    stop_button_visible=False,
                ),
            )
            with (
                patch("src.chatgpt.client.random_delay", new=AsyncMock()),
                patch("src.chatgpt.client.human_type", new=AsyncMock()),
                patch(
                    "src.chatgpt.client.count_assistant_messages",
                    new=AsyncMock(side_effect=[0, 1]),
                ),
                patch(
                    "src.chatgpt.client.get_latest_assistant_turn_signature",
                    new=AsyncMock(side_effect=["1:old-turn", "2:new-turn"]),
                ),
                patch(
                    "src.chatgpt.client.wait_for_response_complete",
                    new=AsyncMock(return_value=complete),
                ),
                patch(
                    "src.chatgpt.client.extract_images_from_response",
                    new=AsyncMock(return_value=[]),
                ),
                patch(
                    "src.chatgpt.client.extract_last_response_via_copy",
                    new=AsyncMock(return_value=""),
                ),
                patch("src.chatgpt.client.asyncio.sleep", new=AsyncMock()),
            ):
                with self.assertRaises(ChatGPTCompletionError) as raised:
                    await client.send_message("private prompt")
            self.assertEqual(
                CompletionStatus.INCOMPLETE, raised.exception.result.status
            )

        asyncio.run(scenario())

    def test_network_response_header_alone_remains_open(self) -> None:
        page = _EventPage()
        recorder = NetworkRecorder(page)
        recorder.start()
        recorder.arm("rca-request-1")
        request = _Request(
            "https://chatgpt.com/backend-api/f/conversation?secret=do-not-store"
        )

        page.emit("request", request)
        page.emit("response", _Response(request))

        self.assertEqual(NetworkStreamStatus.OPEN, recorder.stream_status)
        self.assertNotIn("do-not-store", repr(recorder.get_captured()))

    def test_client_rejects_dom_complete_when_network_request_remains_open(
        self,
    ) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()

            class OpenRecorder:
                stream_status = NetworkStreamStatus.OPEN

                def arm(self, _request_id: str) -> None:
                    return None

            client._network_recorder = OpenRecorder()
            complete = CompletionResult(
                status=CompletionStatus.COMPLETE,
                evidence=CompletionEvidence(
                    evidence="new_turn_final_action_stop_absent_text_stable",
                    turn_signature="2:new-turn",
                    stable_samples=3,
                    stable_for_ms=3000,
                    output_chars=14,
                    output_sha256=self._sha256("Finished reply"),
                    final_action_present=True,
                    stop_button_visible=False,
                ),
            )
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
                    new=AsyncMock(return_value=complete),
                ),
                patch(
                    "src.chatgpt.client.extract_images_from_response",
                    new=AsyncMock(return_value=[]),
                ),
                patch(
                    "src.chatgpt.client.extract_last_response_via_copy",
                    new=AsyncMock(return_value="Finished reply"),
                ),
                patch(
                    "src.chatgpt.client.revalidate_completion_evidence",
                    new=AsyncMock(return_value=complete),
                ),
            ):
                with self.assertRaises(ChatGPTCompletionError) as raised:
                    await client.send_message("synthetic prompt")
            self.assertEqual(
                NetworkStreamStatus.OPEN,
                raised.exception.result.evidence.network_stream,
            )

        asyncio.run(scenario())

    def test_correlated_requestfinished_is_closed(self) -> None:
        page = _EventPage()
        recorder = NetworkRecorder(page)
        recorder.start()
        recorder.arm("rca-request-2")
        request = _Request("https://chatgpt.com/backend-api/f/conversation")

        page.emit("request", request)
        page.emit("requestfinished", request)

        self.assertEqual(NetworkStreamStatus.CLOSED, recorder.stream_status)

    def test_failed_open_and_stale_terminal_never_become_closed(self) -> None:
        page = _EventPage()
        recorder = NetworkRecorder(page)
        recorder.start()
        recorder.arm("rca-request-3")
        current = _Request("https://chatgpt.com/backend-api/f/conversation")
        stale = _Request("https://chatgpt.com/backend-api/f/conversation")

        page.emit("request", current)
        self.assertEqual(NetworkStreamStatus.OPEN, recorder.stream_status)
        page.emit("requestfinished", stale)
        self.assertEqual(NetworkStreamStatus.CONFLICT, recorder.stream_status)

        recorder.arm("rca-request-4")
        failed = _Request("https://chatgpt.com/backend-api/f/conversation")
        page.emit("request", failed)
        page.emit("requestfailed", failed)
        self.assertEqual(NetworkStreamStatus.FAILED, recorder.stream_status)

    def _wait_for_completion(
        self,
        snapshots: list[dict],
        *,
        stop_visible: bool,
        previous_turn_signature: str = "1:old-turn",
        timeout_ms: int = 8,
        stable_ms: int = 0,
        clock_step: float | None = None,
    ):
        async def scenario():
            snapshot = AsyncMock(side_effect=[*snapshots, *([snapshots[-1]] * 20)])
            stop = AsyncMock(return_value=stop_visible)
            clock_value = 0.0

            def advance_clock() -> float:
                nonlocal clock_value
                clock_value += clock_step or 0.0
                return clock_value

            clock_patch = (
                patch.object(detector, "_monotonic", side_effect=advance_clock)
                if clock_step is not None
                else nullcontext()
            )
            with (
                patch.object(detector, "_latest_assistant_turn_snapshot", snapshot),
                patch.object(detector, "_is_stop_button_visible", stop),
                patch.object(detector.Config, "POLL_INTERVAL_MS", 1),
                patch.object(
                    detector.Config, "COMPLETION_STABLE_SAMPLES", 3, create=True
                ),
                patch.object(
                    detector.Config,
                    "COMPLETION_STABLE_MS",
                    stable_ms,
                    create=True,
                ),
                clock_patch,
            ):
                return await detector.wait_for_response_complete(
                    AsyncMock(),
                    timeout_ms=timeout_ms,
                    previous_turn_signature=previous_turn_signature,
                )

        return asyncio.run(scenario())

    def test_stale_previous_turn_does_not_complete(self) -> None:
        result = self._wait_for_completion(
            [
                {
                    "found": True,
                    "signature": "1:old-turn",
                    "hasCopyButton": True,
                    "hasImage": False,
                    "text": "Old complete answer",
                }
            ],
            stop_visible=False,
        )

        self.assertEqual("stale", getattr(result.status, "value", result.status))

    def test_transient_new_turn_does_not_complete(self) -> None:
        result = self._wait_for_completion(
            [
                {
                    "found": True,
                    "signature": "2:new-turn",
                    "hasCopyButton": True,
                    "hasImage": False,
                    "text": "Thinking...",
                }
            ],
            stop_visible=False,
        )

        self.assertNotEqual("complete", getattr(result.status, "value", result.status))

    def test_copy_button_does_not_complete_while_stop_remains_visible(self) -> None:
        result = self._wait_for_completion(
            [
                {
                    "found": True,
                    "signature": "2:new-turn",
                    "hasCopyButton": True,
                    "hasImage": False,
                    "text": "A response that paused mid-stream",
                }
            ],
            stop_visible=True,
        )

        self.assertNotEqual("complete", getattr(result.status, "value", result.status))

    def test_new_turn_stop_absent_and_three_stable_samples_complete(self) -> None:
        result = self._wait_for_completion(
            [
                {
                    "found": True,
                    "signature": "2:new-turn",
                    "hasCopyButton": True,
                    "hasImage": False,
                    "text": "Stable final response",
                }
            ]
            * 3,
            stop_visible=False,
            timeout_ms=10000,
            stable_ms=3000,
            clock_step=0.6,
        )

        self.assertEqual("complete", getattr(result.status, "value", result.status))
        self.assertTrue(result.verified)
        self.assertEqual("2:new-turn", result.evidence.turn_signature)
        self.assertGreaterEqual(result.evidence.stable_samples, 3)

    def test_response_headers_cannot_override_five_visible_stop_samples(self) -> None:
        async def scenario():
            snapshot = AsyncMock(
                return_value={
                    "found": True,
                    "signature": "2:new-turn",
                    "hasCopyButton": True,
                    "hasImage": False,
                    "text": "Stable final response",
                }
            )
            stop = AsyncMock(side_effect=([True] * 5) + ([False] * 20))
            clock_value = 0.0

            def advance_clock() -> float:
                nonlocal clock_value
                clock_value += 0.6
                return clock_value

            with (
                patch.object(detector, "_latest_assistant_turn_snapshot", snapshot),
                patch.object(detector, "_is_stop_button_visible", stop),
                patch.object(detector.Config, "POLL_INTERVAL_MS", 1),
                patch.object(detector.Config, "COMPLETION_STABLE_SAMPLES", 3),
                patch.object(detector.Config, "COMPLETION_STABLE_MS", 3000),
                patch.object(detector, "_monotonic", side_effect=advance_clock),
            ):
                result = await detector.wait_for_response_complete(
                    AsyncMock(),
                    timeout_ms=30000,
                    previous_turn_signature="1:old-turn",
                )
            return result, stop.await_count

        result, stop_checks = asyncio.run(scenario())
        self.assertEqual(CompletionStatus.COMPLETE, result.status)
        self.assertGreaterEqual(stop_checks, 8)

    def test_rate_limit_during_generation_is_a_typed_provider_state(self) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()
            rate_limited = CompletionResult(status=CompletionStatus.RATE_LIMITED)
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
                    new=AsyncMock(return_value=rate_limited),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "not ready") as raised:
                    await client.send_message("synthetic prompt")
            self.assertEqual("rate_limited", raised.exception.state)
            self.assertEqual("rejected", raised.exception.provider_outcome)

        asyncio.run(scenario())

    def test_login_during_generation_preserves_unknown_provider_outcome(self) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()
            login_required = CompletionResult(status=CompletionStatus.LOGIN_REQUIRED)
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
                    new=AsyncMock(return_value=login_required),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "not ready") as raised:
                    await client.send_message("synthetic prompt")
            self.assertEqual("login_required", raised.exception.state)
            self.assertEqual("unknown", raised.exception.provider_outcome)

        asyncio.run(scenario())

    def test_detector_reports_login_required_during_generation(self) -> None:
        async def scenario():
            with (
                patch.object(
                    detector,
                    "_check_page_error",
                    new=AsyncMock(return_value="login_required"),
                ),
                patch.object(detector.Config, "POLL_INTERVAL_MS", 1),
            ):
                return await detector.wait_for_response_complete(
                    AsyncMock(),
                    timeout_ms=10,
                    previous_turn_signature="1:old-turn",
                )

        result = asyncio.run(scenario())
        self.assertEqual(CompletionStatus.LOGIN_REQUIRED, result.status)

    def test_revalidation_rejects_a_different_latest_turn(self) -> None:
        async def scenario():
            complete = CompletionResult(
                status=CompletionStatus.COMPLETE,
                evidence=CompletionEvidence(
                    evidence="new_turn_final_action_stop_absent_text_stable",
                    turn_signature="2:expected-turn",
                    stable_samples=3,
                    stable_for_ms=3000,
                    output_chars=14,
                    output_sha256=self._sha256("Finished reply"),
                    final_action_present=True,
                    stop_button_visible=False,
                ),
            )
            with (
                patch.object(
                    detector,
                    "_latest_assistant_turn_snapshot",
                    new=AsyncMock(
                        return_value={
                            "signature": "3:different-turn",
                            "hasCopyButton": True,
                            "hasImage": False,
                            "text": "Finished reply",
                        }
                    ),
                ),
                patch.object(
                    detector,
                    "_is_stop_button_visible",
                    new=AsyncMock(return_value=False),
                ),
            ):
                return await detector.revalidate_completion_evidence(
                    AsyncMock(),
                    complete,
                    "Finished reply",
                )

        result = asyncio.run(scenario())
        self.assertEqual(CompletionStatus.STALE, result.status)

    def test_revalidation_rejects_text_changed_after_stability(self) -> None:
        async def scenario():
            stable_text = "Original stable reply"
            changed_text = "Changed after the stable window"
            complete = CompletionResult(
                status=CompletionStatus.COMPLETE,
                evidence=CompletionEvidence(
                    evidence="new_turn_final_action_stop_absent_text_stable",
                    turn_signature="2:expected-turn",
                    stable_samples=3,
                    stable_for_ms=3000,
                    output_chars=len(stable_text),
                    output_sha256=self._sha256(stable_text),
                    final_action_present=True,
                    stop_button_visible=False,
                ),
            )
            with (
                patch.object(
                    detector,
                    "_latest_assistant_turn_snapshot",
                    new=AsyncMock(
                        return_value={
                            "signature": "2:expected-turn",
                            "hasCopyButton": True,
                            "hasImage": False,
                            "text": changed_text,
                        }
                    ),
                ),
                patch.object(
                    detector,
                    "_is_stop_button_visible",
                    new=AsyncMock(return_value=False),
                ),
            ):
                return await detector.revalidate_completion_evidence(
                    AsyncMock(),
                    complete,
                    changed_text,
                )

        result = asyncio.run(scenario())
        self.assertEqual(CompletionStatus.INCOMPLETE, result.status)

    def test_complete_status_without_full_evidence_is_not_verified(self) -> None:
        result = CompletionResult(status=CompletionStatus.COMPLETE)
        self.assertFalse(result.verified)

    def test_client_maps_complete_status_with_missing_evidence_to_incomplete(
        self,
    ) -> None:
        async def scenario() -> None:
            client = self._client_ready_to_send()
            invalid_complete = CompletionResult(status=CompletionStatus.COMPLETE)
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
                    new=AsyncMock(return_value=invalid_complete),
                ),
            ):
                with self.assertRaises(ChatGPTCompletionError) as raised:
                    await client.send_message("synthetic prompt")
            self.assertEqual(
                CompletionStatus.INCOMPLETE,
                raised.exception.result.status,
            )

        asyncio.run(scenario())

    def test_copy_button_does_not_complete_while_stop_button_is_visible(self) -> None:
        page = _StreamingPage()

        with patch.object(detector.Config, "POLL_INTERVAL_MS", 1):
            result = asyncio.run(
                detector._wait_for_copy_button_or_image(
                    page,
                    timeout_ms=4,
                    previous_turn_signature="1:old-turn",
                )
            )

        self.assertIsNone(result)

    def test_text_stability_does_not_complete_while_stop_button_is_visible(
        self,
    ) -> None:
        page = _StreamingPage()

        with patch.object(detector.Config, "POLL_INTERVAL_MS", 1):
            result = asyncio.run(
                detector._wait_via_text_stability(
                    page,
                    timeout_ms=5,
                    previous_turn_signature="1:old-turn",
                )
            )

        self.assertFalse(result)

    def test_click_send_rejects_composer_button_when_it_is_stop(self) -> None:
        async def scenario() -> None:
            client = object.__new__(ChatGPTClient)
            page = AsyncMock()
            page.evaluate.return_value = {
                "selector": "#composer-submit-button",
                "disabled": False,
                "ariaDisabled": None,
                "visible": True,
                "dataTestId": "stop-button",
                "ariaLabel": "Stop generating",
                "text": "Stop",
                "classes": "composer-submit-button",
            }
            client._page = page
            client._find_selector = AsyncMock(return_value="#composer-submit-button")

            with patch("src.chatgpt.client.human_click", new=AsyncMock()) as click:
                with self.assertRaisesRegex(RuntimeError, "still generating"):
                    await client._click_send()
                click.assert_not_awaited()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
