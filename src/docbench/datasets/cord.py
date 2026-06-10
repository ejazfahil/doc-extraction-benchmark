"""CORD-v2 loader.

Loads ``naver-clova-ix/cord-v2`` (CC-BY-4.0) and normalizes each example into a
:class:`Document`: the receipt image plus a flat dict of gold scalar summary
fields (the v1 scoring target). The nested ``menu[]`` line-items are deliberately
dropped here — see docs/CONTRACT.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from docbench.scoring.match import CANONICAL_FIELDS

if TYPE_CHECKING:
    from PIL.Image import Image

CORD_DATASET_ID = "naver-clova-ix/cord-v2"


@dataclass(frozen=True)
class Document:
    """A single receipt and its gold scalar fields."""

    doc_id: str
    image: Image
    gold: dict[str, str]  # canonical field -> gold value; only fields present in gt
    doc_class: str = "receipt"
    layout_type: str = "photographed"
    meta: dict[str, Any] = field(default_factory=dict)


def _flatten_gold(gt_parse: dict[str, Any]) -> dict[str, str]:
    """Extract the canonical scalar fields from a CORD ``gt_parse`` object.

    Only keys in :data:`CANONICAL_FIELDS` that are actually present are returned;
    absent fields are simply omitted (recall-oriented scoring).
    """
    gold: dict[str, str] = {}
    for canonical in CANONICAL_FIELDS:
        group, key = canonical.split(".", 1)
        section = gt_parse.get(group)
        if isinstance(section, dict) and key in section:
            value = section[key]
            if value is not None:
                gold[canonical] = str(value)
    return gold


def load_cord(split: str = "test", limit: int | None = None) -> Iterator[Document]:
    """Yield :class:`Document` objects from a CORD split.

    Args:
        split: ``"train"`` | ``"validation"`` | ``"test"``.
        limit: optional cap on number of documents (for smoke runs).

    The ``datasets`` import is local so the package can be imported (and the
    contract/scoring unit-tested) without the heavy dependency installed.
    """
    from datasets import load_dataset

    ds = load_dataset(CORD_DATASET_ID, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        raw = json.loads(row["ground_truth"])
        gt_parse = raw.get("gt_parse", {})
        yield Document(
            doc_id=f"cord-{split}-{i}",
            image=row["image"],
            gold=_flatten_gold(gt_parse),
            meta={"dataset_version": f"cord-v2/{split}"},
        )
