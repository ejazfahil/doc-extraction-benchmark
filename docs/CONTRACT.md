# Item-level results contract (v1)

This file is the **interface** between this benchmark (the *producer*) and any
downstream statistical-inference consumer (e.g. `bayesian-llm-eval`). Both sides
implement this schema independently — there is no code dependency between the
repos, only this contract.

## Grain
**One row per scored field instance**: `(doc_id, field, system, model, run_id)`.

## Columns

| column | type | null? | description |
|---|---|---|---|
| `doc_id` | str | no | CORD sample id |
| `doc_class` | str | no | Coarse document class (v1: `"receipt"`; reserved for multi-dataset use) |
| `layout_type` | str | no | `printed` \| `scanned` \| `photographed` (CORD = `photographed`) |
| `field` | str | no | Canonical field key, e.g. `total.total_price` |
| `system` | str | no | `ocr` \| `vlm` — the population factor of interest |
| `model` | str | no | Concrete model, e.g. `tesseract`, `pixtral-12b` |
| `run_id` | int | no | Repetition index (≥0) for run-to-run variance |
| `correct` | int | no | **The binary outcome**: 1 if prediction matches gold, else 0 |
| `pred_value` | str | yes | Raw predicted value (for error analysis) |
| `gold_value` | str | no | Ground-truth value |
| `cost_usd` | float | no | Marginal cost of producing this document's extraction (0.0 for local) |
| `latency_ms` | float | no | Wall-clock latency for this document's extraction |
| `prompt_tokens` | int | yes | null for OCR systems |
| `completion_tokens` | int | yes | null for OCR systems |
| `dataset_version` | str | no | e.g. `cord-v2/test` |
| `harness_version` | str | no | `docbench` version that produced the row |
| `timestamp` | str | no | ISO-8601 UTC |

## Scoring rules (v1)
- **Fields scored**: the scalar summary fields only (see `docbench.scoring.match.CANONICAL_FIELDS`).
  CORD `menu[]` line-items require sequence alignment and are **out of scope for v1**
  (documented future work).
- A `(doc, field)` is emitted **only when the field is present in the gold parse**
  (recall-oriented binary correctness). Hallucinated extra fields are *not* part of the
  binary outcome; they are summarised separately by the frequentist precision report.
- `cost_usd` / `latency_ms` are per-document but repeated on each of that document's
  field rows — consumers must de-duplicate on `doc_id` before summing cost/latency.

## Stability
Additive changes (new nullable columns) bump the minor contract version in this file's
title. Any change to grain or to `correct` semantics is a breaking change.
