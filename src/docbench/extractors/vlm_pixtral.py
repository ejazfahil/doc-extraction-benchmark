"""Pixtral (Mistral) vision extractor.

Sends the receipt image to a Mistral vision model and asks for the canonical
scalar fields as JSON. Implements the :class:`docbench.extractors.base.Extractor`
protocol.

API shape verified against https://docs.mistral.ai/capabilities/vision/ (chat
completions; image passed as an ``image_url`` content part that accepts a base64
data URI). NOTE (rule: no hallucinated APIs): the model id and per-token pricing
change over time and the vision doc does not pin `response_format`/`usage` — these
are therefore **configurable** and the code degrades gracefully if a field is
absent. Verify the current id at https://docs.mistral.ai/getting-started/models/
and pricing at https://mistral.ai/pricing before quoting cost numbers.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docbench.extractors.base import Extraction
from docbench.scoring.match import CANONICAL_FIELDS

if TYPE_CHECKING:
    from PIL.Image import Image

_DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
# Default id at time of writing; override via constructor. Verify against docs.
_DEFAULT_MODEL = "pixtral-12b-2409"

_PROMPT = (
    "You are extracting fields from a receipt image. Return ONLY a JSON object "
    "whose keys are EXACTLY these field names and whose values are the extracted "
    "string (a number like \"12.50\"), or null if the field is not present:\n"
    + "\n".join(f"  - {f}" for f in CANONICAL_FIELDS)
    + "\nDo not invent values. Do not add other keys. Output JSON only."
)


@dataclass
class PixtralPricing:
    """USD per 1M tokens. Defaults are placeholders — verify before reporting cost."""

    input_per_mtok: float = 0.15
    output_per_mtok: float = 0.15


class PixtralExtractor:
    """Mistral vision-model extractor (implements the Extractor protocol)."""

    system = "vlm"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        pricing: PixtralPricing | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.name = model
        self.model = model
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError("MISTRAL_API_KEY not set (pass api_key= or set the env var)")
        self._base_url = base_url.rstrip("/")
        self._pricing = pricing or PixtralPricing()
        self._timeout = timeout

    @staticmethod
    def _to_data_uri(image: Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _parse_json(text: str) -> dict[str, object]:
        """Best-effort JSON parse: handles bare JSON and ```-fenced blocks."""
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if not brace:
                return {}
            try:
                obj = json.loads(brace.group(0))
            except json.JSONDecodeError:
                return {}
        return obj if isinstance(obj, dict) else {}

    def extract(self, image: Image) -> Extraction:
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": self._to_data_uri(image)},
                    ],
                }
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        start = time.perf_counter()
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        parsed = self._parse_json(content)
        fields = {
            f: str(parsed[f])
            for f in CANONICAL_FIELDS
            if parsed.get(f) is not None and str(parsed[f]).strip() != ""
        }

        usage = data.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        cost = 0.0
        if prompt_tokens is not None:
            cost += prompt_tokens / 1_000_000 * self._pricing.input_per_mtok
        if completion_tokens is not None:
            cost += completion_tokens / 1_000_000 * self._pricing.output_per_mtok

        return Extraction(
            fields=fields,
            latency_ms=latency_ms,
            cost_usd=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
