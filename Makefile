# Test gate. Needs pytest and the runtime deps (pip install -e ".[dev]").
.PHONY: test
test:
	python3 -m pytest -q
