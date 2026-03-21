# Sprint Plan — polyfit-image-compress

## Sprint Info

| Field         | Value                          |
| ------------- | ------------------------------ |
| Start Date    | 2026-03-21                     |
| Deadline      | Open-ended                     |
| Total Phases  | 6 (0-5)                        |
| Current Phase | Phase 3 — Quantum Module       |

## Phase 0: Documentation and Planning — COMPLETED

- [x] 01-Development-Guide.md
- [x] 02-Architecture.md
- [x] 03-Algorithm-Theory.md
- [x] 04-Quantum-Integration.md
- [x] 05-Developer-Workflow.md
- [x] 06-Benchmarks-Strategy.md
- [x] 07-Roadmap.md

## Phase 1: Package Extraction — COMPLETED

Goal: Extract notebook code into installable Python package
Merged: PR #7 to master (2026-03-20)

Tasks:
- [x] Create src/polyfit_compress/ package structure
- [x] Extract LeastSquaresCompressor to compressor.py (refactored with validation)
- [x] Create models.py with Strategy pattern (LinearModel, QuadraticModel)
- [x] Create metrics.py (PSNR, SSIM, MSE utilities)
- [x] Create pyproject.toml with metadata and dependencies
- [x] Remove google.colab dependency
- [x] Add RGB support (process channels independently)
- [x] Add input validation and custom exceptions
- [x] Add type hints on all public API
- [x] Write basic test suite (pytest)
- [x] Verify `pip install -e .` works

## Phase 2: File Format and CLI — COMPLETED

Goal: Make the compressor usable as a standalone tool
Merged: PR #13 to master (2026-03-21)

Tasks:
- [x] Design .pfic file format (header + coefficients)
- [x] Implement io.py (save/load compressed files)
- [x] Implement cli.py with Typer
- [x] CLI commands: compress, decompress, benchmark, info
- [x] Create visualization.py (comparison plots, error maps)
- [x] Move notebook to notebooks/ directory
- [x] Update README with pip install instructions

## Phase 3: Quantum Module

Goal: Add educational quantum computing integration with PennyLane

Tasks:
- [x] Create quantum/ subpackage with conditional PennyLane import
- [x] Implement VQLS solver (quantum/vqls.py)
- [x] Implement quantum feature maps (quantum/feature_maps.py)
- [x] Implement hybrid pipeline (quantum/hybrid.py)
- [x] Add [quantum] optional dependency in pyproject.toml
- [x] Write quantum tests (mock device for CI)
- [ ] Create quantum tutorial notebook
- [ ] Benchmark quantum vs classical (honest comparison)

## Phase 4: Benchmarks and CI/CD

Goal: Prove the approach works, enable contributions

Tasks:
- [ ] Create benchmark runner (benchmarks/run_benchmark.py)
- [ ] Benchmark on Kodak dataset (24 images)
- [ ] Benchmark on BSD68 dataset
- [ ] Compare against JPEG, WebP, SVD at matched sizes
- [ ] Generate rate-distortion curves
- [ ] Set up GitHub Actions CI (tests, lint, type check)
- [ ] Publish to PyPI (v0.1.0)
- [ ] Create CHANGELOG.md
- [ ] Create CONTRIBUTING.md

## Phase 5: Community and Visibility

Goal: Attract users and contributors

Tasks:
- [ ] Build interactive demo (Streamlit or Gradio)
- [ ] Write blog post / article
- [ ] Submit to relevant awesome-lists
- [ ] Create API documentation site (mkdocs)
- [ ] Update Colab notebook to use the package
- [ ] Create example gallery with different image types

## Progress Tracker

| Phase       | Status      | Tasks  | Completed |
| ----------- | ----------- | ------ | --------- |
| Phase 0     | COMPLETED   | 7      | 7         |
| Phase 1     | COMPLETED   | 11     | 11        |
| Phase 2     | COMPLETED   | 7      | 7         |
| Phase 3     | IN PROGRESS | 8      | 6         |
| Phase 4     | PENDING     | 9      | 0         |
| Phase 5     | PENDING     | 6      | 0         |
| **Total**   |             | **48** | **29**    |
