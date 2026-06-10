"""Tests for frequentist stats (needs pandas/numpy)."""

import pandas as pd
import pytest

from docbench.stats.frequentist import _exact_mcnemar_p, accuracy_with_ci, mcnemar


def _df() -> pd.DataFrame:
    # 2 docs, 2 fields, 2 models. tesseract worse than pixtral.
    recs = []
    for doc in ("d0", "d1"):
        for field in ("total.total_price", "sub_total.tax_price"):
            recs.append({"doc_id": doc, "field": field, "model": "tesseract", "correct": 0})
            recs.append({"doc_id": doc, "field": field, "model": "pixtral", "correct": 1})
    return pd.DataFrame(recs)


def test_accuracy_with_ci_ranks_models():
    cis = {c.group: c for c in accuracy_with_ci(_df(), n_boot=200, seed=1)}
    assert cis["pixtral"].accuracy == 1.0
    assert cis["tesseract"].accuracy == 0.0
    assert cis["pixtral"].n_docs == 2
    assert cis["pixtral"].lo <= cis["pixtral"].accuracy <= cis["pixtral"].hi


def test_mcnemar_detects_one_sided_advantage():
    r = mcnemar(_df(), "pixtral", "tesseract")
    assert r.a_better == 4  # pixtral wins all 4 discordant pairs
    assert r.b_better == 0
    # Exact two-sided p with 4 all-one-sided discordant pairs floors at 2*(1/2^4)=0.125.
    assert r.p_value == pytest.approx(0.125)


def test_mcnemar_reaches_significance_with_more_pairs():
    # 8 one-sided discordant pairs -> 2*(1/2^8) < 0.01.
    assert _exact_mcnemar_p(8, 0) == pytest.approx(2 / 256)


def test_exact_mcnemar_symmetry_and_bounds():
    assert _exact_mcnemar_p(0, 0) == 1.0
    assert _exact_mcnemar_p(5, 5) == 1.0
    assert 0.0 <= _exact_mcnemar_p(8, 0) <= 1.0
    assert _exact_mcnemar_p(8, 0) < 0.01
