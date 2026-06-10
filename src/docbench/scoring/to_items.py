"""Assemble item-level contract rows from an extraction + gold.

This is the single place where the binary ``correct`` outcome and the per-field
rows are produced. Output rows conform to ``docbench.schema.ResultsSchema``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from docbench.datasets.cord import Document
from docbench.extractors.base import Extraction
from docbench.scoring.match import is_correct

# Single source of truth for the harness version stamped on every row.
try:  # pragma: no cover - trivial
    from importlib.metadata import version

    HARNESS_VERSION = version("doc-extraction-benchmark")
except Exception:  # pragma: no cover
    HARNESS_VERSION = "0.0.0+local"


def build_rows(
    doc: Document,
    extraction: Extraction,
    *,
    model: str,
    system: str,
    run_id: int,
) -> list[dict[str, object]]:
    """Produce one contract row per gold-present field for this (doc, model, run).

    Only fields present in ``doc.gold`` are scored (recall-oriented), matching the
    contract. Cost/latency are repeated on each row and must be de-duplicated on
    ``doc_id`` by consumers before aggregation.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for field_key, gold_value in doc.gold.items():
        pred = extraction.fields.get(field_key)
        rows.append(
            {
                "doc_id": doc.doc_id,
                "doc_class": doc.doc_class,
                "layout_type": doc.layout_type,
                "field": field_key,
                "system": system,
                "model": model,
                "run_id": run_id,
                "correct": is_correct(pred, gold_value),
                "pred_value": pred,
                "gold_value": gold_value,
                "cost_usd": extraction.cost_usd,
                "latency_ms": extraction.latency_ms,
                "prompt_tokens": extraction.prompt_tokens,
                "completion_tokens": extraction.completion_tokens,
                "dataset_version": doc.meta.get("dataset_version", "unknown"),
                "harness_version": HARNESS_VERSION,
                "timestamp": now,
            }
        )
    return rows
