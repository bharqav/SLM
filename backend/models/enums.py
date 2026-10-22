from enum import StrEnum


class MessageRole(StrEnum):
    """Who produced the message (PRD: user | assistant; system reserved for tooling)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
