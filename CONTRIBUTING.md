# Contributing to polyfit-image-compress

## Development Setup

```bash
git clone https://github.com/Jelsin29/polyfit-image-compress.git
cd polyfit-image-compress
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# For quantum features
pip install -e ".[dev,quantum]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=polyfit_compress --cov-report=term-missing
```

## Code Quality

Before every commit:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
mypy src/polyfit_compress/
```

## Commit Convention

Format: `<type>(<scope>): <description>`

Types: feat, fix, refactor, docs, test, bench, chore, ci

## Pull Requests

1. Create a feature branch from master
2. Make your changes with tests
3. Ensure all checks pass
4. Open a PR with a clear description
