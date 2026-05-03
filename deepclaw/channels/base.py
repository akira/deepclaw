from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    async def send(self, chat_id: str, text: str) -> str:
        """Send a text message. Returns a message_id for later editing."""
        ...

    async def send_media(self, chat_id: str, path: str, caption: str | None = None) -> str:
        """Send a local media file when the channel supports native attachments."""
        fallback = caption or f"MEDIA:{path}"
        return await self.send(chat_id, fallback)

    @abstractmethod
    async def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
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
    def message_limit(self) -> int:
        """Max message length for this channel."""
        return 4096
