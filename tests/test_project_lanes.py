from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from src.chatgpt.lane_pool import ChatGPTLanePool
from src.chatgpt.models import ChatResponse


class _FakePage:
    def __init__(self, page_number: int) -> None:
        self.page_number = page_number
        self.url = "https://chatgpt.com"
        self.closed = False

    async def goto(self, url: str, **_kwargs) -> None:
        self.url = url

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self) -> None:
        self.next_page = 2

    async def new_page(self) -> _FakePage:
        page = _FakePage(self.next_page)
        self.next_page += 1
        return page


class _FakeClient:
    active = 0
    max_active = 0
    events: list[tuple[int, str]] = []

    def __init__(self, page: _FakePage) -> None:
        self.page = page

    async def new_chat(self) -> None:
        self.events.append((self.page.page_number, "new"))

    async def navigate_to_project(self, _project_ref: str) -> None:
        return None

    async def navigate_to_thread(self, _thread_ref: str) -> None:
        return None

    async def send_message(self, prompt: str, **_kwargs) -> ChatResponse:
        self.__class__.active += 1
        self.__class__.max_active = max(self.__class__.max_active, self.__class__.active)
        self.__class__.events.append((self.page.page_number, prompt))
        await asyncio.sleep(0.03)
        self.__class__.active -= 1
        return ChatResponse(message=prompt, thread_id=f"thread-{self.page.page_number}")


class ProjectLanePoolTests(unittest.TestCase):
    def test_same_lane_is_ordered_and_different_lanes_overlap(self) -> None:
        async def scenario() -> None:
            _FakeClient.active = 0
            _FakeClient.max_active = 0
            _FakeClient.events = []
            browser = _FakeBrowser()
            initial = _FakeClient(_FakePage(1))
            with patch("src.chatgpt.lane_pool.ChatGPTClient", _FakeClient):
                pool = ChatGPTLanePool(browser, initial, lane_count=2, max_concurrency=2, min_message_gap_seconds=0)
                results = await asyncio.gather(
                    pool.send(1, "lane-1-chapter-1"),
                    pool.send(1, "lane-1-chapter-11"),
                    pool.send(2, "lane-2-chapter-2"),
                )
                await pool.close()

            lane_one_events = [prompt for page, prompt in _FakeClient.events if page == 1 and prompt != "new"]
            self.assertEqual(["lane-1-chapter-1", "lane-1-chapter-11"], lane_one_events)
            self.assertGreaterEqual(_FakeClient.max_active, 2)
            self.assertEqual(
                ["lane-1-chapter-1", "lane-1-chapter-11", "lane-2-chapter-2"],
                [result.message for result in results],
            )

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
