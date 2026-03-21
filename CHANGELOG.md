# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Benchmark runner for reproducible compression benchmarks
- GitHub Actions CI pipeline (lint, typecheck, test matrix)
- PyPI publishing workflow

## [0.1.0] - 2026-03-22

### Added
- Core `LeastSquaresCompressor` with configurable block size and polynomial models
- `LinearModel` and `QuadraticModel` implementing the `PolynomialModel` protocol
- Quality metrics: `psnr`, `ssim`, `mse`, `compression_ratio`
- `.pfic` binary file format with `PficHeader`, `save_pfic`, `load_pfic`
- CLI via Typer: `polyfit compress`, `polyfit decompress`, `polyfit info`, `polyfit benchmark`
- Visualization: `compare_images` (side-by-side), `error_heatmap`
- Quantum module (optional, requires PennyLane):
  - `VQLSSolver` with `StronglyEntanglingLayers` ansatz
  - `QuantumFeatureMap` satisfying `PolynomialModel` protocol
  - `HybridCompressor` with classical fallback
- RGB and grayscale image support
- Custom exceptions: `InvalidBlockSizeError`, `UnsupportedImageFormatError`, `CorruptedFileError`, `IncompatibleVersionError`
- Comprehensive test suite (48+ tests)
- Tutorial notebooks: original demo + quantum tutorial
- Full documentation in `DOCS/` (9 documents)
