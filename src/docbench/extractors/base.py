"""Extractor protocol — the contract every OCR/VLM backend implements.

An extractor turns a document image into a dict of predicted field values plus
the cost/latency/token telemetry needed by the results contract. Keeping this a
``Protocol`` means OCR and VLM backends are interchangeable and new ones need no
base-class inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class Extraction:
    """Result of running one extractor on one document."""

    fields: dict[str, str]  # canonical field -> predicted value
    latency_ms: float
    cost_usd: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class Extractor(Protocol):
    """A document-extraction backend."""

    name: str  # concrete model id, e.g. "tesseract", "pixtral-12b"
    system: str  # "ocr" | "vlm"

    def extract(self, image: Image) -> Extraction:
        """Extract canonical field values from a single document image."""
        ...
