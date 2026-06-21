from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChannelDeliveryIssue(Exception):
    """Base class for channel-specific delivery failures with structured handling."""


@dataclass
class ChannelEditRateLimited(ChannelDeliveryIssue):
    """A channel temporarily rejected an edit due to rate limiting."""

    retry_after_seconds: float

    def __str__(self) -> str:
        return f"edit rate-limited, retry after {self.retry_after_seconds:.1f}s"


@dataclass
class ChannelEditUnavailable(ChannelDeliveryIssue):
    """Editing is unavailable for the target message and a fresh send is required."""

    reason: str = "edit unavailable"


@dataclass
class IncomingMessage:
    """Normalized inbound message from any channel."""

    text: str
    chat_id: str
    user_id: str
    username: str | None = None
    source: str = ""  # "telegram", "tui", etc.


@dataclass
class OutgoingMessage:
    """Response to send back through a channel."""

    text: str
    chat_id: str


class Channel(ABC):
    """Abstract base for all messaging channels."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send(self, chat_id: str, text: str, *, render_markdown: bool = False) -> str:
        """Send a text message. Returns a message_id for later editing."""
        ...

    async def send_media(self, chat_id: str, path: str, caption: str | None = None) -> str:
        """Send a local media file when the channel supports native attachments."""
        fallback = caption or f"MEDIA:{path}"
        return await self.send(chat_id, fallback)

    @abstractmethod
    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        render_markdown: bool = False,
    ) -> None:
        """Edit a previously sent message. Silently ignore if unsupported."""
        ...

    @abstractmethod
    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator."""
        ...

    @property
    def supports_edit(self) -> bool:
        """Whether this channel supports message editing for streaming."""
        return False

    @property
    def requires_final_reformat_edit(self) -> bool:
        """Whether final delivery should force one last edit for rendering upgrades."""
        return False

    @property
    def message_limit(self) -> int:
        """Max message length for this channel."""
        return 4096
