"""Frequentist summaries over the item-level results.

NOTE on what is (and isn't) reported: the v1 binary outcome is *recall-oriented*
(only gold-present fields are scored — see docs/CONTRACT.md), so the headline
metric here is **field accuracy = recall**, not F1. Precision/F1 require emitting
rows for hallucinated extra fields, which is deferred; until then we do not report
an F1 number rather than report a misleading one.

Uncertainty uses a **document-clustered bootstrap** (rows within a document are
correlated), and model-vs-model comparison uses an **exact McNemar test** on
paired (doc, field) outcomes — no SciPy dependency (exact binomial via stdlib).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AccuracyCI:
    group: str
    accuracy: float  # mean(correct) = recall on gold-present fields
    lo: float
    hi: float
    n_items: int
    n_docs: int


def accuracy_with_ci(
    df: pd.DataFrame,
    *,
    group_col: str = "model",
    doc_col: str = "doc_id",
    outcome: str = "correct",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> list[AccuracyCI]:
    """Per-group accuracy with a document-clustered bootstrap CI."""
    rng = np.random.default_rng(seed)
    results: list[AccuracyCI] = []
    for group, gdf in df.groupby(group_col, sort=True):
        doc_ids = gdf[doc_col].unique()
        # Pre-split outcomes per document for fast resampling.
        per_doc = [gdf.loc[gdf[doc_col] == d, outcome].to_numpy() for d in doc_ids]
        point = float(np.concatenate(per_doc).mean())
        boot = np.empty(n_boot)
        n_docs = len(per_doc)
        for b in range(n_boot):
            pick = rng.integers(0, n_docs, size=n_docs)
            sample = np.concatenate([per_doc[i] for i in pick])
            boot[b] = sample.mean()
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        results.append(
            AccuracyCI(
                group=str(group),
                accuracy=point,
                lo=float(lo),
                hi=float(hi),
                n_items=int(np.concatenate(per_doc).size),
                n_docs=int(n_docs),
            )
        )
    return results


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial on discordant pairs, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


@dataclass
class McNemarResult:
    model_a: str
    model_b: str
    a_better: int  # discordant pairs where a correct, b wrong
    b_better: int  # discordant pairs where b correct, a wrong
    n_pairs: int
    p_value: float


def mcnemar(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    model_col: str = "model",
    outcome: str = "correct",
    pair_keys: tuple[str, ...] = ("doc_id", "field"),
) -> McNemarResult:
    """Exact McNemar test comparing two models on paired (doc, field) outcomes."""
    a = df[df[model_col] == model_a].set_index(list(pair_keys))[outcome]
    b = df[df[model_col] == model_b].set_index(list(pair_keys))[outcome]
    joined = pd.DataFrame({"a": a, "b": b}).dropna()
    a_better = int(((joined["a"] == 1) & (joined["b"] == 0)).sum())
    b_better = int(((joined["a"] == 0) & (joined["b"] == 1)).sum())
    return McNemarResult(
        model_a=model_a,
        model_b=model_b,
        a_better=a_better,
        b_better=b_better,
        n_pairs=int(len(joined)),
        p_value=_exact_mcnemar_p(a_better, b_better),
    )
