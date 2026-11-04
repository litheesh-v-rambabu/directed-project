# Contributing

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or any torch >= 2.2
make setup
pre-commit install
```

## Workflow
- Branch from `main`; keep changes focused. Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`).
- Run `make lint typecheck test` before pushing; CI runs the same on Python 3.11 and 3.12.
- New behaviour needs a test. Tests must run offline (use `tests/fixtures/mini_split.npz` or the generator); slow end-to-end tests carry `@pytest.mark.slow`.
- Regenerate the fixture with `python scripts/make_fixtures.py` only if the feature/data format changes.

## Reporting results
Numbers in the README, model card or resume bullets must come from a command recorded in `docs/EXPERIMENTS.md`. Never quote an estimate. Anything not run is labelled "not yet run". Synthetic or proxy data must be labelled as such wherever it is mentioned.

## Layout
`src/directed_speech_detector/{data,models,training,evaluation,serving}`, `export.py`, `benchmark.py`, `cli.py`; configs in `configs/`; decisions in `docs/DECISIONS.md`.
