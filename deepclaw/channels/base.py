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
