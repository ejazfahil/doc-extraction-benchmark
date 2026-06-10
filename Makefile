.PHONY: help setup lint type test smoke clean

help:
	@echo "setup  Install package + dev/ocr/vlm/stats extras (editable)"
	@echo "lint   ruff check + format --check"
	@echo "type   mypy (strict)"
	@echo "test   pytest"
	@echo "smoke  end-to-end run on a few CORD docs (WIP)"

setup:
	pip install -e ".[ocr,vlm,stats,dev]"

lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src

test:
	pytest -q

smoke:
	@echo "TODO: docbench run --split test --limit 5 (CLI in progress)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
