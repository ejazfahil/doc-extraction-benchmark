"""Spine test: extraction + gold -> contract rows that pass schema validation.

Uses synthetic objects only (no network, no datasets dependency), so it proves
the producer end of the contract in isolation.
"""

import pandas as pd

from docbench.datasets.cord import Document
from docbench.extractors.base import Extraction
from docbench.scoring.to_items import build_rows


def _doc() -> Document:
    return Document(
        doc_id="cord-test-0",
        image=None,  # type: ignore[arg-type]  # not touched by build_rows
        gold={"total.total_price": "10.00", "sub_total.tax_price": "1.00"},
        meta={"dataset_version": "cord-v2/test"},
    )


def test_build_rows_scores_only_gold_fields():
    extraction = Extraction(
        fields={"total.total_price": "$10.00", "sub_total.tax_price": "9.99"},
        latency_ms=42.0,
        cost_usd=0.0,
    )
    rows = build_rows(_doc(), extraction, model="tesseract", system="ocr", run_id=0)

    assert len(rows) == 2  # one per gold field, not per predicted field
    by_field = {r["field"]: r for r in rows}
    assert by_field["total.total_price"]["correct"] == 1  # currency-normalized match
    assert by_field["sub_total.tax_price"]["correct"] == 0  # 9.99 != 1.00


def test_rows_validate_against_contract_schema():
    from docbench.schema import COLUMNS, ResultsSchema

    extraction = Extraction(fields={}, latency_ms=1.0, cost_usd=0.0)
    rows = build_rows(_doc(), extraction, model="easyocr", system="ocr", run_id=1)
    frame = pd.DataFrame(rows)

    validated = ResultsSchema.validate(frame)
    assert list(validated.columns) == list(COLUMNS)
    # No prediction supplied -> every gold field is incorrect.
    assert validated["correct"].sum() == 0
