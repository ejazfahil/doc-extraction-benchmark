# doc-extraction-benchmark — when is a VLM actually worth it for document extraction?

![status](https://img.shields.io/badge/status-WIP%20scaffolding-yellow)
![python](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> **TL;DR** *(numbers land once the benchmark has run — see Status).* An honest,
> field-level extraction benchmark of **VLMs vs classical OCR** on the CORD
> receipt dataset. Scores per-field accuracy, cost, and latency, and **exports
> item-level results** so the decision boundary ("when does the expensive model
> actually pay off?") can be analysed rigorously — including by the companion
> [`bayesian-llm-eval`](#companion-project) package.

## Aim
Most "VLM beats OCR" claims report a single aggregate number with no cost axis and
no uncertainty. This project is designed so the VLM **can lose**: it measures
per-field correctness, dollars, and milliseconds on the same documents, and ships
the raw per-item scores for honest statistical inference.

**Pre-registered hypothesis (H1):** a VLM improves per-field accuracy by ≥5 points
over the OCR baseline on CORD. The benchmark is built to confirm *or refute* it.

## Tools
- **Data:** CORD-v2 (`naver-clova-ix/cord-v2`, CC-BY-4.0), 800/100/100 split.
- **OCR baselines:** Tesseract, EasyOCR.
- **VLM:** Pixtral (Mistral API) — EU-relevant; an open self-hosted VLM (Qwen2.5-VL via vLLM) planned.
- **Scoring:** currency-normalized binary field match → field-level P/R/F1.
- **Stats:** bootstrap CIs + McNemar (frequentist, in-repo); hierarchical Bayes (companion repo).

## Status
🚧 **Scaffolding.** The data contract, CORD loader, extractor protocol, and binary
scoring are implemented and unit-tested. **Not yet built:** the extractor backends,
the run/score CLI, and the statistics layer. **No results are reported yet** — this
section will carry a real table (with `n`, dataset split, hardware, and a
reproduce command) only after a genuine run. No placeholder numbers.

## Results
_Pending first run. Will include: per-field F1 by system (OCR vs VLM) with 95% CIs,
and an accuracy-vs-cost Pareto frontier._

## Quickstart
> Requires Python 3.11–3.12 (ML/stats wheels lag newer CPython).
```bash
uv venv --python 3.12 && source .venv/bin/activate   # or pyenv/conda
pip install -e ".[ocr,vlm,stats,dev]"
make test
# docbench run --models tesseract,pixtral-12b --split test   # (CLI WIP)
```

## Data contract
The producer/consumer interface is specified in [`docs/CONTRACT.md`](docs/CONTRACT.md):
one row per `(doc_id, field, system, model, run_id)` with a binary `correct`
outcome plus cost/latency. This file is the only coupling to the companion project.

## Companion project
[`bayesian-llm-eval`](https://github.com/ejazfahil) (separate repo) consumes this
benchmark's item-level export as its first case study, fitting a hierarchical
logistic model for posterior credible intervals on per-field / per-class accuracy.

## What a production system would add
- Confidence/abstention per field (cascading fallback, as in real OCR pipelines).
- Layout-aware line-item (`menu[]`) extraction with sequence alignment.
- Drift monitoring on field-accuracy over time.

## License
MIT (code). CORD-v2 data is CC-BY-4.0 — attribute accordingly.
