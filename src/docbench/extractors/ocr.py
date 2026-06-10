"""Classical OCR baselines: an OCR engine + a rule-based field parser.

A pure OCR engine returns a blob of text, not structured fields, so to compare it
fairly against a VLM that emits JSON we add a deliberately-simple keyword/regex
parser (:func:`parse_fields_from_text`). That simplicity is intentional — it is
the honest "what you'd get from Tesseract + 50 lines of rules" baseline the VLM
must beat. The parser is pure and unit-tested; the engines (pytesseract/easyocr)
are imported lazily so the module stays importable without them.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Protocol

from docbench.extractors.base import Extraction

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

# Field → keywords, ordered most-specific first so "subtotal" is not stolen by "total".
_FIELD_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sub_total.subtotal_price", ("subtotal", "sub total", "sub-total")),
    ("sub_total.tax_price", ("tax", "vat", "ppn", "gst")),
    ("sub_total.service_price", ("service", "svc")),
    ("sub_total.discount_price", ("discount", "disc")),
    ("total.cashprice", ("cash", "tendered", "paid")),
    ("total.changeprice", ("change", "kembali")),
    ("total.total_price", ("total", "grand total", "amount due")),  # generic last
)

_MONEY = re.compile(r"\d[\d.,]*")


def _last_money(line: str) -> str | None:
    """Return the rightmost money-like token on a line (receipts put amounts right)."""
    matches = [m.group(0) for m in _MONEY.finditer(line) if any(c.isdigit() for c in m.group(0))]
    return matches[-1] if matches else None


def parse_fields_from_text(text: str) -> dict[str, str]:
    """Extract canonical scalar fields from raw OCR text via keyword + amount rules.

    For each canonical field (most-specific keyword first) we find candidate lines
    containing one of its keywords and take the amount from the *last* such line
    (grand totals tend to appear near the bottom). A line is claimed by at most one
    field. ``total.total_price`` explicitly skips lines that are sub-totals.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    claimed: set[int] = set()
    out: dict[str, str] = {}

    for field, keywords in _FIELD_KEYWORDS:
        chosen: str | None = None
        for idx, line in enumerate(lines):
            if idx in claimed:
                continue
            low = line.lower()
            if not any(kw in low for kw in keywords):
                continue
            if field == "total.total_price" and ("subtotal" in low or "sub total" in low):
                continue  # don't let "subtotal" satisfy the generic "total"
            amount = _last_money(line)
            if amount is not None:
                chosen = amount
                claimed.add(idx)  # later (lower) lines override earlier for this field
        if chosen is not None:
            out[field] = chosen
    return out


class OCREngine(Protocol):
    """Turns an image into raw text."""

    name: str

    def image_to_text(self, image: Image) -> str: ...


class TesseractEngine:
    name = "tesseract"

    def image_to_text(self, image: Image) -> str:
        import pytesseract

        return pytesseract.image_to_string(image)


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, languages: tuple[str, ...] = ("en",)) -> None:
        self._languages = list(languages)
        self._reader = None

    def image_to_text(self, image: Image) -> str:
        import numpy as np

        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self._languages, gpu=False)
        # EasyOCR works on arrays; preserve reading order with detail=0.
        lines = self._reader.readtext(np.asarray(image), detail=0)
        return "\n".join(lines)


class OCRExtractor:
    """OCR engine + field parser; implements the Extractor protocol."""

    system = "ocr"

    def __init__(self, engine: OCREngine) -> None:
        self._engine = engine
        self.name = engine.name

    def extract(self, image: Image) -> Extraction:
        start = time.perf_counter()
        text = self._engine.image_to_text(image)
        fields = parse_fields_from_text(text)
        latency_ms = (time.perf_counter() - start) * 1000
        return Extraction(fields=fields, latency_ms=latency_ms, cost_usd=0.0)
