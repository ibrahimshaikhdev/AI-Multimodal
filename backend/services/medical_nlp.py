from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Protocol


class AssertionStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    POSSIBLE = "possible"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MedicalEntityMention:
    text: str
    label: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None
    assertion: AssertionStatus = AssertionStatus.UNKNOWN

    def validate_for(self, source_text: str) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Entity text is required")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Entity label is required")
        if not isinstance(self.assertion, AssertionStatus):
            raise ValueError("Entity assertion must use a supported AssertionStatus")

        if (self.start is None) != (self.end is None):
            raise ValueError("Entity start and end offsets must both be provided or both be omitted")
        if self.start is not None and self.end is not None:
            if any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in (self.start, self.end)
            ):
                raise ValueError("Entity offsets must be integers")
            if self.start < 0 or self.end <= self.start or self.end > len(source_text):
                raise ValueError("Entity offsets are outside the source text")
            if source_text[self.start : self.end] != self.text:
                raise ValueError("Entity offsets must identify the entity text in the source")

        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not isfinite(self.confidence)
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise ValueError("Entity confidence must be a finite number between 0 and 1")


@dataclass(frozen=True)
class MedicalNLPResult:
    mentions: tuple[MedicalEntityMention, ...]


class MedicalNLPProvider(Protocol):
    def analyze(self, text: str) -> Iterable[MedicalEntityMention]:
        """Return standardized medical entity mentions for the supplied text."""


class MedicalNLPService:
    def __init__(self, provider: MedicalNLPProvider):
        if not callable(getattr(provider, "analyze", None)):
            raise TypeError("Medical NLP provider must implement analyze(text)")
        self._provider = provider

    def analyze(self, text: str) -> MedicalNLPResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text content is required")

        provider_output = self._provider.analyze(text)
        try:
            mentions = tuple(provider_output)
        except TypeError as exc:
            raise TypeError("Medical NLP provider must return an iterable of entity mentions") from exc

        for mention in mentions:
            if not isinstance(mention, MedicalEntityMention):
                raise TypeError("Medical NLP providers must return MedicalEntityMention values")
            mention.validate_for(text)

        return MedicalNLPResult(mentions=mentions)


__all__ = [
    "AssertionStatus",
    "MedicalEntityMention",
    "MedicalNLPProvider",
    "MedicalNLPResult",
    "MedicalNLPService",
]