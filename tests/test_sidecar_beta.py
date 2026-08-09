from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.api import routes
from src.api.server import app
from src.browser.manager import BrowserManager
from src.chatgpt.client import ChatGPTClient
from src.config import Config
from sidecar_main import validate_native_security


class SidecarBetaTests(unittest.TestCase):
    def test_gateway_defaults_to_loopback_and_safe_concurrency(self) -> None:
        self.assertEqual("127.0.0.1", Config.API_HOST)
        self.assertGreaterEqual(Config.LANE_COUNT, 1)
        self.assertLessEqual(Config.MAX_CONCURRENCY, 3)
        self.assertLessEqual(Config.MAX_CONCURRENCY, Config.LANE_COUNT)

    def test_native_sidecar_fails_closed_without_token_or_loopback(self) -> None:
        with patch.dict(os.environ, {"API_HOST": "127.0.0.1", "API_TOKEN": ""}):
            with self.assertRaisesRegex(RuntimeError, "random API token"):
                validate_native_security()
        with patch.dict(os.environ, {"API_HOST": "0.0.0.0", "API_TOKEN": "secret"}):
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                validate_native_security()
        with patch.dict(os.environ, {"API_HOST": "127.0.0.1", "API_TOKEN": "secret"}):
            validate_native_security()

    def test_show_login_brings_browser_forward(self) -> None:
        async def scenario() -> None:
            browser = BrowserManager()
            page = AsyncMock()
            page.url = "https://chatgpt.com/"
            browser._page = page
            with patch.object(browser, "is_logged_in", AsyncMock(return_value=False)):
                logged_in = await browser.show_login()
            self.assertFalse(logged_in)
            page.bring_to_front.assert_awaited_once()

        asyncio.run(scenario())

    def test_control_endpoint_works_before_login(self) -> None:
        class FakeBrowser:
            async def show_login(self) -> bool:
                return False

        async def scenario() -> None:
            previous = routes._browser
            routes._browser = FakeBrowser()
            try:
                response = await routes.show_login()
                self.assertEqual("ok", response.status)
                self.assertFalse(response.logged_in)
            finally:
                routes._browser = previous

        asyncio.run(scenario())

    def test_project_route_is_present_in_public_contract(self) -> None:
        project_contract = app.openapi()["paths"]["/projects"]
        self.assertEqual(["post"], sorted(project_contract))

    def test_project_navigation_rejects_lookalike_hosts(self) -> None:
        async def scenario() -> None:
            client = object.__new__(ChatGPTClient)
            page = AsyncMock()
            client._page = page
            for project_ref in (
                "https://chatgpt.com.evil.example/project/demo",
                "https://chatgpt.com@evil.example/project/demo",
                "http://chatgpt.com/project/demo",
            ):
                with self.subTest(project_ref=project_ref):
                    with self.assertRaises(ValueError):
                        await client.navigate_to_project(project_ref)
            page.goto.assert_not_awaited()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
