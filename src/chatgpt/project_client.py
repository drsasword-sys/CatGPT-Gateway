"""Best-effort UI adapter for ChatGPT Projects.

Projects are a ChatGPT Web feature, not a public OpenAI API resource.  This
adapter therefore uses visible UI controls and returns a project URL that can
be persisted by the caller.  It deliberately fails loudly when the UI changes
instead of silently falling back to a normal chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from patchright.async_api import Page

from src.browser.human import random_delay
from src.chatgpt.client import ChatGPTClient
from src.config import Config
from src.log import setup_logging

log = setup_logging("chatgpt_project_client")


class ProjectSetupError(RuntimeError):
    """Project UI could not be completed safely."""


@dataclass(frozen=True)
class ProjectSetupResult:
    project_ref: str
    warnings: tuple[str, ...] = ()


class ChatGPTProjectClient:
    """Create/find a Project using the visible ChatGPT web interface."""

    def __init__(self, client: ChatGPTClient) -> None:
        self.client = client
        self.page: Page = client.page

    async def create_project(
        self,
        name: str,
        *,
        instructions: str = "",
        file_paths: list[str] | None = None,
    ) -> ProjectSetupResult:
        name = str(name or "").strip()
        if not name:
            raise ProjectSetupError("Project name cannot be empty")

        await self.page.goto(Config.CHATGPT_URL, wait_until="domcontentloaded")
        # Current ChatGPT renders this as an icon-only button with
        # aria-label="New project" beside the Projects section.
        await self._click_button(("New project",))
        name_input = self.page.locator(
            "input[placeholder*='project name' i], input[name='name'], input[type='text']"
        ).first
        try:
            await name_input.wait_for(state="visible", timeout=5000)
            await name_input.fill(name)
        except Exception as exc:
            raise ProjectSetupError("ChatGPT Project name field was not found") from exc

        project_ref = ""
        await self._click_button(("Create project", "Create", "Continue"))
        # The project is created asynchronously and the browser may update
        # the URL/sidebar several seconds after the confirmation click. Poll
        # for the resulting project instead of clicking Create again, which
        # can create duplicates or fail after the modal has disappeared.
        for _ in range(12):
            project_ref = await self._find_project_ref(name)
            if project_ref:
                break
            await random_delay(500, 900)
        if not project_ref:
            raise ProjectSetupError(
                "Project was not created or its URL could not be detected; UI may have changed"
            )

        warnings: list[str] = []
        if instructions:
            try:
                await self._set_instructions(instructions)
            except ProjectSetupError as exc:
                warnings.append(str(exc))
                log.warning("Project instructions were not configured: %s", exc)
        if file_paths:
            try:
                await self._upload_project_files(file_paths)
            except ProjectSetupError as exc:
                warnings.append(str(exc))
                log.warning("Project files were not uploaded: %s", exc)
        return ProjectSetupResult(project_ref=project_ref, warnings=tuple(warnings))

    async def _click_text(self, text: str) -> None:
        candidates = [
            self.page.get_by_text(text, exact=True),
            self.page.locator(f"button:has-text('{text}')"),
            self.page.locator(f"a:has-text('{text}')"),
        ]
        for candidate in candidates:
            try:
                if await candidate.count() and await candidate.first.is_visible():
                    await candidate.first.click()
                    return
            except Exception:
                continue
        raise ProjectSetupError(f"ChatGPT control '{text}' was not found")

    async def _click_button(self, names: tuple[str, ...]) -> None:
        for name in names:
            candidates = [
                self.page.locator(f"button[aria-label='{name}']").first,
                self.page.get_by_role("button", name=re.compile(re.escape(name), re.I)).first,
                self.page.locator(f"button:has-text('{name}')").first,
            ]
            for candidate in candidates:
                try:
                    await candidate.wait_for(state="visible", timeout=6000)
                    await candidate.click()
                    return
                except Exception:
                    continue
        raise ProjectSetupError("ChatGPT Project confirmation button was not found")

    async def _find_project_ref(self, name: str) -> str:
        def is_project_href(href: str) -> bool:
            path = urlparse(href).path if href.startswith(("http://", "https://")) else href
            return bool(re.search(r"/project(?:/|$)", path, re.I))

        links = self.page.locator("a[href]")
        for link in await links.all():
            try:
                href = await link.get_attribute("href") or ""
                text = (await link.inner_text()).strip()
            except Exception:
                continue
            if is_project_href(href) and (name.casefold() in text.casefold() or text == name):
                return f"{Config.CHATGPT_URL.rstrip('/')}{href}" if href.startswith("/") else href
        current_url = self.page.url if hasattr(self.page, "url") else ""
        if is_project_href(current_url):
            return current_url
        return ""

    async def _set_instructions(self, instructions: str) -> None:
        # Project settings labels have changed; accept only visible, explicit
        # controls and report a warning instead of writing to an arbitrary box.
        try:
            await self._click_button(("Project settings", "Customize project", "Edit project instructions"))
        except ProjectSetupError as exc:
            raise ProjectSetupError("Project settings control was not found") from exc
        editor = self.page.locator("textarea, [contenteditable='true']").last()
        try:
            await editor.wait_for(state="visible", timeout=5000)
            await editor.fill(instructions)
        except Exception as exc:
            raise ProjectSetupError("Project instructions editor was not found") from exc
        await self._click_button(("Save", "Done"))

    async def _upload_project_files(self, file_paths: list[str]) -> None:
        valid_paths = [str(Path(path)) for path in file_paths if Path(path).is_file()]
        if not valid_paths:
            raise ProjectSetupError("No readable Project files were provided")
        try:
            await self._click_text("Project settings")
        except ProjectSetupError:
            # Some versions expose file upload directly on the Project page.
            pass
        upload = self.page.locator("input[type='file']").first
        try:
            await upload.set_input_files(valid_paths)
        except Exception as exc:
            raise ProjectSetupError("Project file upload control was not found") from exc
