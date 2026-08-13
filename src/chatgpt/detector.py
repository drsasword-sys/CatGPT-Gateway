"""
Response completion detector.

Primary strategy: detect completion on the newest assistant turn only,
then extract from that same turn. This avoids one-turn lag where a
previous assistant response is returned for the current request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time

from patchright.async_api import Page
from patchright._impl._errors import TargetClosedError

from src.selectors import Selectors
from src.browser.human import idle_mouse_movement
from src.log import setup_logging
from src.config import Config
from src.chatgpt.models import (
    CompletionEvidence,
    CompletionResult,
    CompletionStatus,
)

log = setup_logging("detector")
_monotonic = time.monotonic


# ChatGPT has shipped both section-based turns and role-marked message nodes.
# Keep the discovery policy in one place so completion, counting and extraction
# cannot drift apart when the web UI changes again. Role nodes are preferred;
# the legacy section selector remains the fallback for older conversations.
_CONVERSATION_TURNS_JS = """
const legacyTurns = Array.from(
    document.querySelectorAll('section[data-testid^="conversation-turn-"]')
);
const roleTurns = Array.from(document.querySelectorAll(
    '[data-message-author-role="user"], [data-message-author-role="assistant"]'
));
const turns = roleTurns.length > 0 ? roleTurns : legacyTurns;
const getTurnRoot = (turn) => (
    turn.closest(
        'section[data-testid^="conversation-turn-"], [data-testid^="conversation-turn-"]'
    ) || turn
);
const getCopyButton = (turn) => (
    turn.querySelector(
        'button[data-testid="copy-turn-action-button"], button[aria-label="Copy message"], button[aria-label="Copy"]'
    ) || getTurnRoot(turn).querySelector(
        'button[data-testid="copy-turn-action-button"], button[aria-label="Copy message"], button[aria-label="Copy"]'
    )
);
"""


def _conversation_script(body: str, *, argument: bool = False) -> str:
    """Build one DOM probe with the shared conversation discovery policy."""
    prefix = "(previousSignature) => {" if argument else "() => {"
    return f"{prefix}\n{_CONVERSATION_TURNS_JS}\n{body}\n}}"


def normalize_assistant_text(text: str | None) -> str:
    """Normalize extracted assistant text for validation and comparisons."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^ChatGPT said:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^You said:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _assistant_text_sha256(text: str) -> str:
    """Hash normalized text without retaining another content copy."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_incomplete_response_text(text: str | None) -> bool:
    """
    Heuristic: true when text looks like transient "thinking/searching" UI status.
    """
    cleaned = normalize_assistant_text(text)
    if not cleaned:
        return True

    lower = cleaned.lower()
    markers = [
        "pro thinking",
        "thinking",
        "searching for",
        "searching the web",
        "analyzing",
        "working on",
        "please wait",
        "gathering",
    ]

    if any(marker in lower for marker in markers):
        if len(cleaned) < 240:
            return True
        if lower.startswith(("pro thinking", "thinking", "searching", "analyzing", "working on", "gathering")):
            return True

    return False


def _empty_snapshot() -> dict:
    return {
        "found": False,
        "index": -1,
        "signature": None,
        "hasCopyButton": False,
        "hasImage": False,
        "text": "",
    }


async def _dump_all_turns(page: Page) -> list[dict]:
    """Dump info about all conversation turns for debugging."""
    try:
        turns = await page.evaluate(
            _conversation_script(
                """
                return turns.map((turn, idx) => {
                    const root = getTurnRoot(turn);
                    const role = turn.getAttribute('data-message-author-role') ||
                        turn.getAttribute('data-turn') ||
                        root.getAttribute('data-turn') ||
                        'unknown';
                    const textLength = (turn.innerText || '').trim().length;
                    const buttons = root.querySelectorAll('button').length;
                    const copyBtn = Boolean(getCopyButton(turn));
                    const hasArticle = Boolean(root.querySelector('article')) ||
                        root.matches('article');
                    return { idx, role, textLength, buttons, copyBtn, hasArticle };
                });
                """
            )
        )
        return turns or []
    except Exception as e:
        log.debug(f"_dump_all_turns failed: {e}")
        return []


async def _latest_assistant_turn_snapshot(page: Page) -> dict:
    """
    Return metadata for the latest assistant turn (article ordered).

    signature format: "<article-index>:<stable-id>"
    where stable-id is best-effort from DOM attributes.
    """
    snapshot = await page.evaluate(
        _conversation_script(
            """
            for (let idx = turns.length - 1; idx >= 0; idx--) {
                const turn = turns[idx];
                const root = getTurnRoot(turn);
                const turnRole = turn.getAttribute('data-message-author-role') ||
                    turn.getAttribute('data-turn') ||
                    root.getAttribute('data-turn');
                const hasAssistantRole = turnRole === 'assistant' ||
                    Boolean(turn.querySelector('[data-message-author-role="assistant"]')) ||
                    Boolean(root.querySelector('[data-message-author-role="assistant"]'));
                if (!hasAssistantRole) continue;

                const stableId =
                    turn.getAttribute('data-message-id') ||
                    turn.getAttribute('data-turn-id') ||
                    turn.getAttribute('data-testid') ||
                    turn.id ||
                    root.getAttribute('data-message-id') ||
                    root.getAttribute('data-turn-id') ||
                    root.getAttribute('data-testid') ||
                    root.id ||
                    '';

                const hasCopyButton = Boolean(getCopyButton(turn));

                const hasImage = Boolean(
                    root.querySelector('img[alt="Generated image"], div[id^="image-"] img, div[id^="image-"]')
                );

                const text = (turn.innerText || '').trim();

                return {
                    found: true,
                    index: idx,
                    signature: `${idx}:${stableId}`,
                    hasCopyButton,
                    hasImage,
                    text,
                };
            }

            return {
                found: false,
                index: -1,
                signature: null,
                hasCopyButton: false,
                hasImage: false,
                text: '',
            };
            """,
        )
    )

    if not isinstance(snapshot, dict):
        return _empty_snapshot()

    normalized = _empty_snapshot()
    normalized.update(snapshot)
    return normalized


async def get_latest_assistant_turn_signature(page: Page) -> str | None:
    """Return signature for the latest assistant turn, if available."""
    snapshot = await _latest_assistant_turn_snapshot(page)
    signature = snapshot.get("signature")
    return signature if isinstance(signature, str) and signature else None


async def count_assistant_messages(page: Page) -> int:
    """Count assistant turns (article based, newest-UI friendly)."""
    count = await page.evaluate(
        _conversation_script(
            """
            let total = 0;
            for (const turn of turns) {
                const root = getTurnRoot(turn);
                const turnRole = turn.getAttribute('data-message-author-role') ||
                    turn.getAttribute('data-turn') ||
                    root.getAttribute('data-turn');
                const hasAssistantRole = turnRole === 'assistant' ||
                    Boolean(turn.querySelector('[data-message-author-role="assistant"]')) ||
                    Boolean(root.querySelector('[data-message-author-role="assistant"]'));
                if (hasAssistantRole) total++;
            }
            return total;
            """,
        )
    )
    return int(count or 0)


async def _detect_image_in_latest_turn(page: Page, previous_turn_signature: str | None = None) -> bool:
    """Check if the newest assistant turn (not previous turn) contains an image."""
    snapshot = await _latest_assistant_turn_snapshot(page)
    signature = snapshot.get("signature")
    is_new_turn = previous_turn_signature is None or (
        isinstance(signature, str) and signature != previous_turn_signature
    )
    return bool(is_new_turn and snapshot.get("hasImage"))


async def _count_copy_buttons(page: Page) -> int:
    """Count assistant turns that currently expose a copy button."""
    count = await page.evaluate(
        _conversation_script(
            """
            let total = 0;
            for (const turn of turns) {
                const root = getTurnRoot(turn);
                const turnRole = turn.getAttribute('data-message-author-role') ||
                    turn.getAttribute('data-turn') ||
                    root.getAttribute('data-turn');
                const hasAssistantRole = turnRole === 'assistant' ||
                    Boolean(turn.querySelector('[data-message-author-role="assistant"]')) ||
                    Boolean(root.querySelector('[data-message-author-role="assistant"]'));
                if (!hasAssistantRole) continue;
                const hasCopyButton = getCopyButton(turn);
                if (hasCopyButton) total++;
            }
            return total;
            """,
        )
    )
    return int(count or 0)


async def _wait_for_new_turn_signature(
    page: Page,
    previous_turn_signature: str,
    timeout_ms: int,
) -> bool:
    """Wait until latest assistant-turn signature differs from previous one."""
    elapsed = 0
    poll_interval = Config.POLL_INTERVAL_MS / 1000
    heartbeat = 10

    while elapsed * 1000 < timeout_ms:
        snapshot = await _latest_assistant_turn_snapshot(page)
        signature = snapshot.get("signature")
        if isinstance(signature, str) and signature and signature != previous_turn_signature:
            log.debug(f"New assistant turn detected: {signature} (prev: {previous_turn_signature})")
            return True

        if elapsed > 0 and elapsed % heartbeat == 0:
            log.debug(f"Still waiting for new assistant turn... ({int(elapsed)}s)")

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    log.debug("Timed out waiting for a new assistant-turn signature")
    return False


async def wait_for_response_complete(
    page: Page,
    expected_msg_count: int | None = None,
    timeout_ms: int | None = None,
    previous_turn_signature: str | None = None,
    max_output_bytes: int | None = None,
) -> CompletionResult:
    """
    Wait until ChatGPT finishes generating the current response.

    A result is complete only when the newest assistant turn has a final action,
    the stop control is absent, non-transient text is stable for the configured
    sample count and duration, and all evidence points to that same turn.
    """
    timeout = timeout_ms or Config.RESPONSE_TIMEOUT
    log.info(f"Waiting for response (timeout: {timeout}ms)...")
    if timeout <= 0:
        return CompletionResult(status=CompletionStatus.TIMEOUT)

    required_samples = max(3, Config.COMPLETION_STABLE_SAMPLES)
    required_stable_ms = max(3000, Config.COMPLETION_STABLE_MS)
    output_limit = max_output_bytes or Config.MAX_OUTPUT_BYTES
    poll_interval = max(0.001, Config.POLL_INTERVAL_MS / 1000)
    deadline = _monotonic() + (timeout / 1000)

    last_text = ""
    stable_samples = 0
    stable_started_at: float | None = None
    saw_stale_turn = False
    saw_new_turn = False
    saw_transient = False
    selector_failures = 0
    latest_evidence = CompletionEvidence()

    while _monotonic() < deadline:
        page_error = await _check_page_error(page)
        if page_error == "login_required":
            return CompletionResult(
                status=CompletionStatus.LOGIN_REQUIRED,
                evidence=latest_evidence,
            )
        if page_error == "rate_limited":
            return CompletionResult(
                status=CompletionStatus.RATE_LIMITED,
                evidence=latest_evidence,
            )
        try:
            snapshot = await _latest_assistant_turn_snapshot(page)
        except TargetClosedError:
            return CompletionResult(
                status=CompletionStatus.SELECTOR_DRIFT,
                evidence=latest_evidence,
            )
        except Exception as exc:
            selector_failures += 1
            log.warning("Could not inspect the latest assistant turn: %s", type(exc).__name__)
            remaining = deadline - _monotonic()
            if remaining > 0:
                await asyncio.sleep(min(poll_interval, remaining))
            continue

        signature = snapshot.get("signature")
        signature = signature if isinstance(signature, str) and signature else None
        is_new_turn = signature is not None and (
            previous_turn_signature is None or signature != previous_turn_signature
        )
        if not is_new_turn:
            saw_stale_turn = saw_stale_turn or (
                signature is not None and signature == previous_turn_signature
            )
            remaining = deadline - _monotonic()
            if remaining > 0:
                await asyncio.sleep(min(poll_interval, remaining))
            continue

        saw_new_turn = True
        text = normalize_assistant_text(
            snapshot.get("text") if isinstance(snapshot.get("text"), str) else ""
        )
        transient = is_incomplete_response_text(text)
        saw_transient = saw_transient or transient
        final_action_present = bool(
            snapshot.get("hasCopyButton") or snapshot.get("hasImage")
        )
        stop_visible = await _is_stop_button_visible(page)

        if len(text.encode("utf-8")) > output_limit:
            return CompletionResult(
                status=CompletionStatus.OUTPUT_TOO_LARGE,
                evidence=CompletionEvidence(
                    turn_signature=signature,
                    output_chars=len(text),
                    final_action_present=final_action_present,
                    stop_button_visible=stop_visible,
                ),
            )

        now = _monotonic()
        if stop_visible:
            # Streaming text is not final evidence. Start a fresh stability
            # window only after ChatGPT's stop control has disappeared.
            last_text = ""
            stable_samples = 0
            stable_started_at = None
        elif text and not transient:
            if text == last_text:
                stable_samples += 1
            else:
                last_text = text
                stable_samples = 1
                stable_started_at = now
        else:
            last_text = ""
            stable_samples = 0
            stable_started_at = None

        stable_for_ms = (
            int((now - stable_started_at) * 1000)
            if stable_started_at is not None
            else 0
        )
        latest_evidence = CompletionEvidence(
            turn_signature=signature,
            stable_samples=stable_samples,
            stable_for_ms=stable_for_ms,
            output_chars=len(text),
            output_sha256=_assistant_text_sha256(text) if text else None,
            final_action_present=final_action_present,
            stop_button_visible=stop_visible,
        )

        if (
            final_action_present
            and not stop_visible
            and text
            and not transient
            and stable_samples >= required_samples
            and stable_for_ms >= required_stable_ms
        ):
            latest_evidence.evidence = (
                "new_turn_final_action_stop_absent_text_stable"
            )
            log.info(
                "Verified completion for turn %s (%s stable samples)",
                signature,
                stable_samples,
            )
            return CompletionResult(
                status=CompletionStatus.COMPLETE,
                evidence=latest_evidence,
            )

        remaining = deadline - _monotonic()
        if remaining > 0:
            await asyncio.sleep(min(poll_interval, remaining))

    if selector_failures and not saw_new_turn:
        status = CompletionStatus.SELECTOR_DRIFT
    elif saw_stale_turn and not saw_new_turn:
        status = CompletionStatus.STALE
    elif saw_new_turn and (
        saw_transient
        or not latest_evidence.final_action_present
        or latest_evidence.output_chars == 0
    ):
        status = CompletionStatus.INCOMPLETE
    else:
        status = CompletionStatus.TIMEOUT

    log.warning("Response completion was not verified (status=%s)", status.value)
    return CompletionResult(status=status, evidence=latest_evidence)


async def revalidate_completion_evidence(
    page: Page,
    completion: CompletionResult,
    extracted_text: str,
) -> CompletionResult:
    """Re-check the exact final turn immediately before returning content."""
    if not completion.verified:
        if completion.status is CompletionStatus.COMPLETE:
            return CompletionResult(
                status=CompletionStatus.INCOMPLETE,
                evidence=completion.evidence,
            )
        return completion

    try:
        snapshot = await _latest_assistant_turn_snapshot(page)
        stop_visible = await _is_stop_button_visible(page)
    except Exception as exc:
        log.warning("Could not revalidate completion evidence: %s", type(exc).__name__)
        return CompletionResult(
            status=CompletionStatus.SELECTOR_DRIFT,
            evidence=completion.evidence,
        )

    signature = snapshot.get("signature")
    if signature != completion.evidence.turn_signature:
        return CompletionResult(
            status=CompletionStatus.STALE,
            evidence=completion.evidence,
        )

    dom_text = normalize_assistant_text(
        snapshot.get("text") if isinstance(snapshot.get("text"), str) else ""
    )
    extracted = normalize_assistant_text(extracted_text)
    extracted_hash = _assistant_text_sha256(extracted) if extracted else None
    final_action_present = bool(
        snapshot.get("hasCopyButton") or snapshot.get("hasImage")
    )
    evidence = completion.evidence.model_copy(
        update={
            "final_action_present": final_action_present,
            "stop_button_visible": stop_visible,
        }
    )

    if stop_visible:
        return CompletionResult(status=CompletionStatus.TIMEOUT, evidence=evidence)
    if (
        not final_action_present
        or not extracted
        or is_incomplete_response_text(extracted)
        or dom_text != extracted
        or extracted_hash != completion.evidence.output_sha256
        or len(extracted) != completion.evidence.output_chars
        or completion.evidence.stable_samples < 3
        or completion.evidence.stable_for_ms < Config.COMPLETION_STABLE_MS
    ):
        return CompletionResult(
            status=CompletionStatus.INCOMPLETE,
            evidence=evidence,
        )
    return CompletionResult(status=CompletionStatus.COMPLETE, evidence=evidence)


async def _check_page_error(page: Page) -> str | None:
    """Check if the page is showing an error state (DNS failure, crash, etc.).

    Returns error description string if an error is detected, None otherwise.
    """
    try:
        error = await page.evaluate(
            """
            () => {
                // Chrome error pages
                const body = document.body ? document.body.innerText : '';
                if (body.includes('DNS_PROBE_FINISHED_NXDOMAIN')) return 'DNS_PROBE_FINISHED_NXDOMAIN';
                if (body.includes('ERR_NAME_NOT_RESOLVED')) return 'ERR_NAME_NOT_RESOLVED';
                if (body.includes('ERR_CONNECTION_REFUSED')) return 'ERR_CONNECTION_REFUSED';
                if (body.includes('ERR_INTERNET_DISCONNECTED')) return 'ERR_INTERNET_DISCONNECTED';
                if (body.includes('ERR_CONNECTION_TIMED_OUT')) return 'ERR_CONNECTION_TIMED_OUT';
                // ChatGPT error states
                if (body.includes('Something went wrong')) return 'ChatGPT_something_went_wrong';
                if (
                    body.includes("We're experiencing high demand") ||
                    body.includes('You have reached the current usage cap') ||
                    body.includes('rate limit')
                ) return 'rate_limited';
                const loginLink = document.querySelector(
                    'a[href*="/auth/login"], button[data-testid="login-button"]'
                );
                if (loginLink || body.includes('Log in to continue')) return 'login_required';
                if (document.title && document.title.includes('is not available')) return 'page_not_available';
                return null;
            }
            """
        )
        return error
    except Exception:
        return None


async def _is_stop_button_visible(page: Page) -> bool:
    """Return whether ChatGPT is still streaming the current response."""
    selectors = json.dumps(Selectors.STOP_BUTTON)
    try:
        return bool(
            await page.evaluate(
                f"""
                () => {{
                    const selectors = {selectors};
                    for (const selector of selectors) {{
                        let button = null;
                        try {{
                            button = document.querySelector(selector);
                        }} catch (_error) {{
                            continue;
                        }}
                        if (!button) continue;
                        const style = window.getComputedStyle(button);
                        const rect = button.getBoundingClientRect();
                        if (
                            style.display !== 'none' &&
                            style.visibility !== 'hidden' &&
                            rect.width > 0 &&
                            rect.height > 0
                        ) {{
                            return true;
                        }}
                    }}
                    return false;
                }}
                """
            )
        )
    except Exception as exc:
        log.warning(f"Could not verify stop-button state: {exc}")
        return True


async def _wait_for_copy_button_or_image(
    page: Page,
    timeout_ms: int,
    previous_turn_signature: str | None = None,
) -> str | None:
    """
    Wait for either copy-button readiness or generated image on the latest turn.

    Returns "copy", "image", or None if timed out.
    """
    elapsed = 0
    poll_interval = Config.POLL_INTERVAL_MS / 1000
    heartbeat = 10
    next_heartbeat = heartbeat
    first_snapshot_logged = False

    while elapsed * 1000 < timeout_ms:
        try:
            snapshot = await _latest_assistant_turn_snapshot(page)
        except TargetClosedError:
            log.error("Page/browser closed while waiting for response")
            return None
        except Exception as e:
            log.warning(f"Snapshot failed ({type(e).__name__}): {e}")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            continue

        signature = snapshot.get("signature")
        is_new_turn = previous_turn_signature is None or (
            isinstance(signature, str) and signature != previous_turn_signature
        )

        # Log the first snapshot for diagnostics
        if not first_snapshot_logged and elapsed >= 2:
            first_snapshot_logged = True
            turn_length = len(snapshot.get("text") or "")
            all_turns = await _dump_all_turns(page)
            log.debug(
                f"First snapshot at {int(elapsed)}s | "
                f"prev_sig={previous_turn_signature} cur_sig={signature} "
                f"is_new={is_new_turn} copy={snapshot.get('hasCopyButton')} "
                f"text_length={turn_length}"
            )
            log.debug(f"All turns ({len(all_turns)}): {all_turns}")

        has_copy_button = bool(snapshot.get("hasCopyButton"))
        has_image = bool(snapshot.get("hasImage"))
        if is_new_turn and (has_copy_button or has_image):
            if await _is_stop_button_visible(page):
                log.debug(f"Latest turn {signature} still streaming; completion is blocked")
            elif has_copy_button:
                log.info(f"Copy button detected on completed latest turn {signature}")
                return "copy"
            else:
                await asyncio.sleep(0.5)
                if not await _is_stop_button_visible(page):
                    log.info(f"Generated image detected on completed latest turn {signature}")
                    return "image"

        if elapsed >= next_heartbeat:
            next_heartbeat = elapsed + heartbeat
            # Diagnostic: log what we see on the latest assistant turn
            turn_length = len(snapshot.get("text") or "")
            log.debug(
                f"Still waiting for copy/image... ({int(elapsed)}s) | "
                f"sig={signature} is_new={is_new_turn} "
                f"copy={snapshot.get('hasCopyButton')} "
                f"text_length={turn_length}"
            )

            # Dump all turn info for debugging
            all_turns = await _dump_all_turns(page)
            log.debug(f"All turns: {all_turns}")

            await idle_mouse_movement(page)

            # Check for page-level errors every heartbeat to fail fast
            page_error = await _check_page_error(page)
            if page_error:
                log.error(f"Page error detected while waiting: {page_error}")
                return None

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Diagnostic: save screenshot on timeout
    try:
        await page.screenshot(path="logs/detector_timeout.png")
        log.info("Saved timeout screenshot to logs/detector_timeout.png")
    except Exception as e:
        log.debug(f"Could not save timeout screenshot: {e}")

    log.warning(f"Neither copy button nor image found after {int(elapsed)}s")
    return None


async def _wait_via_stop_button(page: Page, timeout_ms: int) -> bool:
    """Wait for stop button appear -> disappear cycle."""
    stop_selector = ", ".join(Selectors.STOP_BUTTON)
    log.debug("Waiting for stop button to appear...")

    try:
        await page.wait_for_selector(stop_selector, state="visible", timeout=15000)
        log.info("Stop button appeared — response is streaming")
    except Exception:
        log.debug("Stop button never appeared (short response or selector changed)")
        return False

    log.debug("Waiting for stop button to disappear...")
    heartbeat_interval = 10
    elapsed = 0

    while elapsed * 1000 < timeout_ms:
        try:
            await page.wait_for_selector(stop_selector, state="hidden", timeout=heartbeat_interval * 1000)
            log.info("Stop button disappeared — streaming done")
            return True
        except Exception:
            elapsed += heartbeat_interval
            log.debug(f"Still streaming... ({elapsed}s elapsed)")
            await idle_mouse_movement(page)

    log.warning(f"Timed out after {elapsed}s waiting for stop button")
    return False


async def _wait_via_text_stability(
    page: Page,
    timeout_ms: int,
    previous_turn_signature: str | None = None,
) -> bool:
    """
    Last resort: poll latest assistant-turn text and wait until stable.

    If previous_turn_signature is provided, ignores stabilization on that old turn.
    """
    stable_count = 0
    required_stable = 3
    last_text = ""
    elapsed = 0
    poll_interval = Config.POLL_INTERVAL_MS / 1000

    while elapsed * 1000 < timeout_ms:
        snapshot = await _latest_assistant_turn_snapshot(page)
        signature = snapshot.get("signature")
        text = snapshot.get("text") if isinstance(snapshot.get("text"), str) else ""

        if previous_turn_signature and signature == previous_turn_signature:
            stable_count = 0
            last_text = ""
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            continue

        if text and text == last_text:
            stable_count += 1
            log.debug(f"Text stable ({stable_count}/{required_stable})")
            if stable_count >= required_stable:
                if await _is_stop_button_visible(page):
                    log.debug("Text is stable but response is still streaming; continuing wait")
                    stable_count = 0
                    last_text = text
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                    continue
                if is_incomplete_response_text(text) and not bool(snapshot.get("hasCopyButton")):
                    log.debug("Stable text looks like transient thinking status; continuing wait")
                    stable_count = 0
                    last_text = text
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                    continue
                log.info("Response text stabilized — complete")
                return True
        else:
            stable_count = 0
            last_text = text

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    log.warning(f"Text stability timed out after {int(elapsed)}s")
    return False


async def extract_last_response_via_copy(
    page: Page,
    previous_turn_signature: str | None = None,
) -> str:
    """
    Extract latest assistant response by clicking copy on the latest turn.

    Never intentionally copies from previous_turn_signature when provided.
    """
    log.debug("Attempting extraction via latest-turn copy button...")

    try:
        await page.context.grant_permissions(["clipboard-read", "clipboard-write"])

        if previous_turn_signature:
            await _wait_for_new_turn_signature(page, previous_turn_signature, timeout_ms=8000)

        pre_clipboard = await page.evaluate("navigator.clipboard.readText().catch(() => '')")
        await page.evaluate("navigator.clipboard.writeText('').catch(() => {})")

        click_result = await page.evaluate(
            _conversation_script(
                """
                for (let idx = turns.length - 1; idx >= 0; idx--) {
                    const turn = turns[idx];
                    const root = getTurnRoot(turn);
                    const turnRole = turn.getAttribute('data-message-author-role') ||
                        turn.getAttribute('data-turn') ||
                        root.getAttribute('data-turn');
                    const hasAssistantRole = turnRole === 'assistant' ||
                        Boolean(turn.querySelector('[data-message-author-role="assistant"]')) ||
                        Boolean(root.querySelector('[data-message-author-role="assistant"]'));
                    if (!hasAssistantRole) continue;

                    const stableId =
                        turn.getAttribute('data-message-id') ||
                        turn.getAttribute('data-turn-id') ||
                        turn.getAttribute('data-testid') ||
                        turn.id ||
                        root.getAttribute('data-message-id') ||
                        root.getAttribute('data-turn-id') ||
                        root.getAttribute('data-testid') ||
                        root.id ||
                        '';
                    const signature = `${idx}:${stableId}`;

                    if (previousSignature && signature === previousSignature) {
                        return { clicked: false, reason: 'stale-turn', signature };
                    }

                    const btn = getCopyButton(turn);
                    if (!btn) {
                        return { clicked: false, reason: 'no-copy-button', signature };
                    }

                    btn.click();
                    return { clicked: true, reason: 'ok', signature };
                }

                return { clicked: false, reason: 'no-assistant-turn', signature: null };
                """,
                argument=True,
            ),
            previous_turn_signature,
        )

        if isinstance(click_result, dict) and click_result.get("clicked"):
            await asyncio.sleep(0.3)
            content = await page.evaluate("navigator.clipboard.readText().catch(() => '')")
            if content and content.strip() and content.strip() != str(pre_clipboard).strip():
                log.info(
                    "Extracted via copy button (latest-turn): "
                    f"{len(content)} chars, turn={click_result.get('signature')}"
                )
                return content.strip()
            log.debug("Clipboard unchanged/empty after latest-turn copy click")
        else:
            reason = click_result.get("reason") if isinstance(click_result, dict) else "unknown"
            log.debug(f"Latest-turn copy click not used: {reason}")

    except Exception as e:
        log.warning(f"Copy button extraction failed: {e}")

    log.info("Falling back to latest-turn DOM extraction...")
    return await _extract_via_dom(page, previous_turn_signature)


async def _extract_via_dom(
    page: Page,
    previous_turn_signature: str | None = None,
) -> str:
    """Fallback extraction: innerText from latest assistant turn only."""
    text = await page.evaluate(
        _conversation_script(
            """
            for (let idx = turns.length - 1; idx >= 0; idx--) {
                const turn = turns[idx];
                const root = getTurnRoot(turn);
                const turnRole = turn.getAttribute('data-message-author-role') ||
                    turn.getAttribute('data-turn') ||
                    root.getAttribute('data-turn');
                const hasAssistantRole = turnRole === 'assistant' ||
                    Boolean(turn.querySelector('[data-message-author-role="assistant"]')) ||
                    Boolean(root.querySelector('[data-message-author-role="assistant"]'));
                if (!hasAssistantRole) continue;

                const stableId =
                    turn.getAttribute('data-message-id') ||
                    turn.getAttribute('data-turn-id') ||
                    turn.getAttribute('data-testid') ||
                    turn.id ||
                    root.getAttribute('data-message-id') ||
                    root.getAttribute('data-turn-id') ||
                    root.getAttribute('data-testid') ||
                    root.id ||
                    '';
                const signature = `${idx}:${stableId}`;

                if (previousSignature && signature === previousSignature) {
                    return '';
                }

                return (turn.innerText || '').trim();
            }

            return '';
            """,
            argument=True,
        ),
        previous_turn_signature,
    )

    if text and str(text).strip():
        cleaned = normalize_assistant_text(str(text))
        if is_incomplete_response_text(cleaned):
            log.debug("Latest-turn DOM text looks incomplete/transient; waiting for a fuller reply")
            return ""
        log.debug(f"Extracted via DOM (latest-turn): {len(cleaned)} chars")
        return cleaned

    log.error("Could not extract any latest assistant response")
    return ""


# Keep old name as alias for backward compat
extract_last_response = extract_last_response_via_copy
