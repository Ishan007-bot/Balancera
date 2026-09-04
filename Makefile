# Makefile for the reconciliation agent.
#
# Every target delegates to run.py so the same commands work on machines
# without make (Windows especially). If you have no make, use:
#     python run.py demo

PY ?= python
SEED ?= 42
HARD_RATIO ?= 0.4

.PHONY: demo demo-llm test generate validate clean

## demo: full deterministic pipeline, no API key required
demo:
	$(PY) run.py demo

## demo-llm: the same pipeline with the LLM layer enabled
demo-llm:
	$(PY) run.py demo-llm

## test: run the test suite
test:
	$(PY) -m pytest tests/ -q

generate:
	$(PY) -m recon.cli generate --seed $(SEED) --hard-ratio $(HARD_RATIO) --out data/

validate:
	$(PY) -m recon.cli validate data/

clean:
	$(PY) run.py clean
