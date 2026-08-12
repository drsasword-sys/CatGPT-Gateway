"""
Data models for ChatGPT interactions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import string

from pydantic import BaseModel, Field


class CompletionStatus(str, Enum):
    """Outcome of the browser-side completion detector."""

    COMPLETE = "complete"
    TIMEOUT = "timeout"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    SELECTOR_DRIFT = "selector_drift"
    OUTPUT_TOO_LARGE = "output_too_large"
    LOGIN_REQUIRED = "login_required"
    RATE_LIMITED = "rate_limited"


class NetworkStreamStatus(str, Enum):
    """Minimal terminal state for the correlated provider request."""

    UNAVAILABLE = "unavailable"
    OPEN = "open"
    CLOSED = "closed"
    FAILED = "failed"
    CONFLICT = "conflict"


class CompletionEvidence(BaseModel):
    """Content-free facts used to prove one assistant turn is final."""

    evidence: str = "unverified"
    turn_signature: str | None = None
    stable_samples: int = 0
    stable_for_ms: int = 0
    network_stream: NetworkStreamStatus = NetworkStreamStatus.UNAVAILABLE
    output_chars: int = 0
    output_sha256: str | None = None
    final_action_present: bool = False
    stop_button_visible: bool | None = None


class CompletionResult(BaseModel):
    """Structured completion result; callers must check status explicitly."""

    status: CompletionStatus
    evidence: CompletionEvidence = Field(default_factory=CompletionEvidence)

    @property
    def verified(self) -> bool:
        output_hash = self.evidence.output_sha256 or ""
        return bool(
            self.status is CompletionStatus.COMPLETE
            and self.evidence.evidence
            == "new_turn_final_action_stop_absent_text_stable"
            and self.evidence.turn_signature
            and self.evidence.stable_samples >= 3
            and self.evidence.stable_for_ms >= 3000
            and self.evidence.output_chars > 0
            and len(output_hash) == 64
            and all(character in string.hexdigits for character in output_hash)
            and self.evidence.final_action_present
            and self.evidence.stop_button_visible is False
            and self.evidence.network_stream
            not in {
                NetworkStreamStatus.OPEN,
                NetworkStreamStatus.FAILED,
                NetworkStreamStatus.CONFLICT,
            }
        )


class ImageInfo(BaseModel):
    """Metadata for a generated image."""

    url: str = Field(description="Original image URL from ChatGPT/DALL-E")
    alt: str = Field(default="", description="Alt text / image description")
    local_path: str = Field(default="", description="Local file path after download")
    prompt_title: str = Field(
        default="", description="Image generation title shown by ChatGPT"
    )


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message text content")
    timestamp: datetime = Field(default_factory=datetime.now)
    images: list[ImageInfo] = Field(
        default_factory=list, description="Images in this message"
    )


class ChatResponse(BaseModel):
    """Response from a chat interaction."""

    message: str = Field(description="Assistant's response text")
    thread_id: str = Field(default="", description="Conversation thread ID from URL")
    response_time_ms: int = Field(
        default=0, description="Time taken for response in ms"
    )
    images: list[ImageInfo] = Field(
        default_factory=list, description="Generated images"
    )
    has_images: bool = Field(
        default=False, description="Whether the response contains images"
    )
    completion: CompletionResult | None = Field(
        default=None,
        description="Browser completion evidence for this exact assistant turn",
    )


class Thread(BaseModel):
    """A conversation thread."""

    id: str = Field(description="Thread ID (from URL /c/{id})")
    title: str = Field(default="", description="Thread title from sidebar")
    url: str = Field(default="", description="Full URL")
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
