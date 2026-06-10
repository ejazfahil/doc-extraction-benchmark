"""Command-line interface: ``docbench run`` and ``docbench score``."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(add_completion=False, help="VLM-vs-OCR field-extraction benchmark on CORD.")


def _build_extractor(name: str):
    """Resolve a model name to an Extractor instance (heavy imports are lazy)."""
    key = name.strip().lower()
    if key == "tesseract":
        from docbench.extractors.ocr import OCRExtractor, TesseractEngine

        return OCRExtractor(TesseractEngine())
    if key == "easyocr":
        from docbench.extractors.ocr import EasyOCREngine, OCRExtractor

        return OCRExtractor(EasyOCREngine())
    if key.startswith("pixtral") or key.startswith("mistral"):
        from docbench.extractors.vlm_pixtral import PixtralExtractor

        return PixtralExtractor(model=name)
    raise typer.BadParameter(f"unknown model '{name}' (try: tesseract, easyocr, pixtral-12b-2409)")


@app.command()
def run(
    models: str = typer.Option(..., help="Comma-separated model names."),
    split: str = typer.Option("test", help="CORD split."),
    limit: int = typer.Option(0, help="Max documents (0 = all)."),
    runs: int = typer.Option(1, min=1, help="Repetitions per (doc, model) for variance."),
    out: Path = typer.Option(Path("results.parquet"), help="Output parquet path."),
) -> None:
    """Run extractors over CORD and write item-level results conforming to the contract."""
    import pandas as pd

    from docbench.datasets.cord import load_cord
    from docbench.schema import COLUMNS, ResultsSchema
    from docbench.scoring.to_items import build_rows

    extractors = [_build_extractor(m) for m in models.split(",") if m.strip()]
    docs = list(load_cord(split=split, limit=limit or None))
    rprint(f"[bold]{len(docs)}[/bold] docs × [bold]{len(extractors)}[/bold] models × {runs} run(s)")

    rows: list[dict[str, object]] = []
    for ex in extractors:
        for run_id in range(runs):
            for doc in docs:
                extraction = ex.extract(doc.image)
                rows.extend(
                    build_rows(doc, extraction, model=ex.name, system=ex.system, run_id=run_id)
                )

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    ResultsSchema.validate(frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    rprint(f"[green]wrote[/green] {len(frame)} rows → {out}")


@app.command()
def score(
    results: Path = typer.Argument(..., help="results.parquet from `docbench run`."),
    baseline: str = typer.Option("", help="Model to McNemar every other model against."),
) -> None:
    """Report per-model field accuracy (recall) with clustered bootstrap CIs."""
    import pandas as pd

    from docbench.stats.frequentist import accuracy_with_ci, mcnemar

    df = pd.read_parquet(results)
    cis = accuracy_with_ci(df, group_col="model")

    table = Table(title="Field accuracy (recall) — 95% clustered-bootstrap CI")
    table.add_column("model")
    table.add_column("accuracy", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("items", justify="right")
    table.add_column("docs", justify="right")
    for c in sorted(cis, key=lambda x: x.accuracy, reverse=True):
        table.add_row(
            c.group, f"{c.accuracy:.3f}", f"[{c.lo:.3f}, {c.hi:.3f}]", str(c.n_items), str(c.n_docs)
        )
    rprint(table)

    if baseline:
        others = [c.group for c in cis if c.group != baseline]
        for m in others:
            r = mcnemar(df, m, baseline)
            verdict = "ns" if r.p_value >= 0.05 else "*"
            rprint(
                f"{m} vs {baseline}: {m} better on {r.a_better}, {baseline} on {r.b_better} "
                f"of {r.n_pairs} discordant pairs — exact McNemar p={r.p_value:.4f} {verdict}"
            )


if __name__ == "__main__":  # pragma: no cover
    app()
