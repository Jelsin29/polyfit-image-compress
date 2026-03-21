# Contributing to polyfit-image-compress

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone and set up the development environment
git clone https://github.com/Jelsin29/polyfit-image-compress.git
cd polyfit-image-compress
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# For quantum features
pip install -e ".[dev,quantum]"

# For visualization
pip install -e ".[all]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=polyfit_compress --cov-report=term-missing

# Run a specific test file
pytest tests/test_compressor.py -v

# Run only quantum tests (requires pennylane)
pytest tests/test_quantum.py -v
```

The CI pipeline runs tests across Python 3.10, 3.11, and 3.12. Ensure your changes pass on all supported versions.

## Code Quality

All code must pass linting and type checking before merge. Run these before every commit:

```bash
# Lint and auto-fix
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Type checking (strict mode)
mypy src/polyfit_compress/
```

Configuration for these tools lives in `pyproject.toml`.

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `bench`, `chore`, `ci`

**Examples:**
- `feat(core): add adaptive block size selection`
- `fix(pfic): handle corrupted header gracefully`
- `docs(quantum): update VQLSSolver usage examples`
- `bench(metrics): add SSIM benchmark for large images`
- `ci(workflow): add Python 3.12 to test matrix`

## Branch Strategy

- `master` -- stable release branch
- `feature/<name>` -- feature development branches
- `benchmarks-ci/<phase>` -- CI/benchmark infrastructure branches

Create your feature branch from `master`:

```bash
git checkout master
git pull origin master
git checkout -b feature/your-feature-name
```

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes with corresponding tests
3. Ensure all checks pass locally (`ruff check`, `ruff format`, `mypy`, `pytest`)
4. Push your branch and open a PR with a clear description
5. Wait for CI checks to pass (lint, typecheck, test matrix)
6. Address any review feedback

### PR Checklist

- [ ] Tests added or updated for the change
- [ ] All existing tests pass
- [ ] Code passes `ruff check` and `ruff format`
- [ ] Code passes `mypy` in strict mode
- [ ] Documentation updated if needed

## Quantum Module Development

The quantum module is optional and requires [PennyLane](https://pennylane.ai/). When working on quantum features:

- Install with: `pip install -e ".[dev,quantum]"`
- Quantum tests are skipped automatically when PennyLane is not installed
- The `HybridCompressor` must always provide a classical fallback
- Any new quantum model must satisfy the `PolynomialModel` protocol
- Test both the quantum path and the classical fallback path

## Project Structure

```
src/polyfit_compress/
  core/           # LeastSquaresCompressor, models, metrics
  handlers/       # CLI command handlers
  quantum/        # Optional quantum module (VQLSSolver, HybridCompressor)
  visualization/  # Image comparison and error heatmaps
  cli.py          # Typer CLI entry point
tests/            # Test suite
DOCS/             # Project documentation
notebooks/        # Tutorial notebooks
```
