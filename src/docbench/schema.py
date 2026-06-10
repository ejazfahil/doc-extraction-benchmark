"""Pandas/pandera schema implementing the item-level results contract.

The authoritative description lives in ``docs/CONTRACT.md``; this module is the
machine-checkable mirror. ``validate_results`` is called before any export so a
malformed frame never reaches a downstream consumer.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

CONTRACT_VERSION = "v1"


class ResultsSchema(pa.DataFrameModel):
    """One row per scored field instance. See docs/CONTRACT.md."""

    doc_id: Series[str]
    doc_class: Series[str]
    layout_type: Series[str] = pa.Field(isin=["printed", "scanned", "photographed"])
    field: Series[str]
    system: Series[str] = pa.Field(isin=["ocr", "vlm"])
    model: Series[str]
    run_id: Series[int] = pa.Field(ge=0)
    correct: Series[int] = pa.Field(isin=[0, 1])
    pred_value: Series[str] = pa.Field(nullable=True)
    gold_value: Series[str]
    cost_usd: Series[float] = pa.Field(ge=0.0)
    latency_ms: Series[float] = pa.Field(ge=0.0)
    prompt_tokens: Series[pa.Int64] = pa.Field(nullable=True, ge=0)
    completion_tokens: Series[pa.Int64] = pa.Field(nullable=True, ge=0)
    dataset_version: Series[str]
    harness_version: Series[str]
    timestamp: Series[str]

    class Config:
        strict = True  # reject unexpected columns — the contract is exact
        coerce = True


# Column order for stable on-disk parquet/jsonl.
COLUMNS: tuple[str, ...] = tuple(ResultsSchema.to_schema().columns.keys())
