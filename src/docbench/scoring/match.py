"""Field normalization and binary matching for CORD summary fields.

The binary `correct` outcome in the results contract is produced here. Keep this
module pure and well-tested: every downstream statistic (frequentist *and*
Bayesian) is a function of these 0/1 outcomes, so a bug here silently corrupts
the entire benchmark.
"""

from __future__ import annotations

import re

# Canonical scalar fields scored in v1. CORD `menu[]` line-items are excluded
# (they need sequence alignment); see docs/CONTRACT.md.
CANONICAL_FIELDS: tuple[str, ...] = (
    "sub_total.subtotal_price",
    "sub_total.tax_price",
    "sub_total.service_price",
    "sub_total.discount_price",
    "total.total_price",
    "total.cashprice",
    "total.changeprice",
)

_NON_NUMERIC = re.compile(r"[^0-9.\-]")


def normalize_price(raw: str | None) -> str:
    """Normalize a money-like string to a canonical comparable form.

    Strips currency symbols, thousands separators and whitespace. If the result
    parses as a number it is returned without trailing-zero noise (``"1.50"`` and
    ``"1.5"`` compare equal); otherwise the lowercased, whitespace-collapsed
    string is returned so non-numeric values still compare sensibly.
    """
    if raw is None:
        return ""
    text = raw.strip()
    if not text:
        return ""
    cleaned = _NON_NUMERIC.sub("", text.replace(",", ""))
    try:
        value = float(cleaned)
    except ValueError:
        return re.sub(r"\s+", " ", text.lower())
    # Normalize numeric form: drop trailing zeros, avoid "-0".
    normalized = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if normalized in ("", "-0") else normalized


def is_correct(pred: str | None, gold: str) -> int:
    """Binary correctness for one (doc, field): 1 if prediction matches gold.

    Both sides are passed through :func:`normalize_price` so formatting
    differences (``$1,234.00`` vs ``1234``) do not count as errors.
    """
    return int(normalize_price(pred) == normalize_price(gold))
