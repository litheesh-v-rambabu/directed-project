.PHONY: setup test test-fast lint format typecheck data train ablate report eval export benchmark serve docker clean

PY ?= python
CONFIG ?= configs/fused_speaker.yaml
RUN ?= runs/fused_speaker_s0

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

typecheck:
	mypy

data:
	dsd generate-data --config $(CONFIG)

train:
	dsd train --config $(CONFIG)

ablate:
	dsd ablate --seeds 0 1 2
	dsd report

report:
	dsd report

eval:
	dsd evaluate --run-dir $(RUN)

export:
	dsd export --run-dir $(RUN)

benchmark:
	dsd benchmark --run-dir $(RUN)

serve:
	dsd serve --run-dir $(RUN) --backend fp32

docker:
	docker compose up --build

clean:
	rm -rf runs data .pytest_cache .mypy_cache .ruff_cache
