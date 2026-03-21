# polyfit-image-compress

Image compression using **Least Squares Polynomial Surface Fitting** — a novel approach that approximates image blocks with polynomial surfaces instead of traditional frequency-domain transforms (DCT/JPEG).

Includes an optional **quantum computing module** (PennyLane) that replaces the classical solver with a Variational Quantum Linear Solver (VQLS) for educational purposes.

[![CI](https://github.com/Jelsin29/polyfit-image-compress/actions/workflows/ci.yml/badge.svg)](https://github.com/Jelsin29/polyfit-image-compress/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## How It Works

```
Original Image (H x W)
    |
    v
Split into B x B blocks (e.g., 8x8)
    |
    v
For each block: solve Ax = b via pseudoinverse
    A = polynomial design matrix (precomputed once)
    b = pixel intensities (flattened block)
    x = polynomial coefficients (3 for linear, 6 for quadratic)
    |
    v
Store only the coefficients --> compression!
    |
    v
Reconstruct: evaluate polynomial at each pixel coordinate
```

### Supported Models

| Model | Surface Type | Terms | Coefficients | 8x8 Ratio |
|-------|-------------|-------|:------------:|:---------:|
| **Linear** | Plane | `1, x, y` | 3 | 5.3:1 |
| **Quadratic** | Paraboloid | `1, x, y, xy, x², y²` | 6 | 2.7:1 |

---

## Installation

```bash
# From PyPI (when published)
pip install polyfit-image-compress

# From source (development)
git clone https://github.com/Jelsin29/polyfit-image-compress.git
cd polyfit-image-compress
pip install -e ".[dev]"

# With quantum module
pip install -e ".[dev,quantum]"

# With interactive demo
pip install -e ".[dev,demo]"

# Everything
pip install -e ".[dev,quantum,demo,docs]"
```

---

## Quick Start

### CLI

```bash
# Compress an image to .pfic format
polyfit compress photo.png compressed.pfic --model quadratic --block-size 8

# Decompress back to an image
polyfit decompress compressed.pfic restored.png

# View file metadata
polyfit info compressed.pfic
# Output:
#   PFIC File: compressed.pfic
#     Version:    1
#     Dimensions: 512x512
#     Channels:   1
#     Block size: 8
#     Model:      quadratic

# Run benchmarks comparing models and block sizes
polyfit benchmark photo.png
polyfit benchmark photo.png --model linear
```

### Python API

```python
import numpy as np
from skimage import data

from polyfit_compress import LeastSquaresCompressor, LinearModel, QuadraticModel
from polyfit_compress.metrics import psnr, ssim, mse
from polyfit_compress.io import save_pfic, load_pfic

# Load an image
image = data.camera()  # 512x512 grayscale

# --- Compress ---
compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
result = compressor.compress(image)

print(f"Compression ratio: {result.compression_ratio:.2f}x")
print(f"PSNR: {psnr(image, result.reconstructed):.2f} dB")
print(f"SSIM: {ssim(image, result.reconstructed):.4f}")

# --- Save to .pfic file ---
save_pfic("output.pfic", result)

# --- Load and decompress ---
header, coefficients = load_pfic("output.pfic")
reconstructed = compressor.decompress(
    coefficients,
    original_shape=(header.height, header.width),
    padded_shape=(header.padding_height, header.padding_width),
)

# --- Visualization ---
from polyfit_compress.visualization import compare_images, error_heatmap

fig = compare_images(image, result.reconstructed)
fig.savefig("comparison.png")

fig = error_heatmap(image, result.reconstructed)
fig.savefig("error_map.png")
```

### RGB Images

```python
from skimage import data

# RGB works out of the box — each channel is compressed independently
rgb_image = data.astronaut()  # 512x512x3
result = compressor.compress(rgb_image)
print(f"Shape preserved: {result.reconstructed.shape}")  # (512, 512, 3)
```

---

## Quantum Module

The quantum module is **educational** — it demonstrates how the classical least-squares solver maps to a quantum circuit. Simulators are slower than NumPy, not faster.

```python
from polyfit_compress.quantum import HybridCompressor, VQLSConfig, VQLSSolver

# Hybrid compression: VQLS per block, classical fallback on non-convergence
config = VQLSConfig(n_layers=4, max_iterations=100, stepsize=0.1)
hybrid = HybridCompressor(
    model=LinearModel(),
    block_size=4,  # Small blocks for quantum demos
    vqls_config=config,
    progress_callback=lambda i, total, ok: print(f"Block {i}/{total}: {'VQLS' if ok else 'fallback'}"),
)
result = hybrid.compress(small_image)

# Or solve a single system directly
solver = VQLSSolver(config)
coefficients, converged = solver.solve(design_matrix, pixel_block)
```

See the [quantum tutorial notebook](notebooks/quantum_tutorial.ipynb) for a full walkthrough.

---

## .pfic File Format

The `.pfic` (PolyFit Image Compressed) format stores compressed images:

```
[Header: 31 bytes]          [Payload: float32 coefficients]
magic(4) | version(1)       coefficients array
height(4) | width(4)        shape: (channels, num_coeffs, num_blocks)
channels(1) | block_size(2)
model_type(1) | padding(8)
reserved(6)
```

---

## Project Structure

```
polyfit-image-compress/
├── src/polyfit_compress/       # Core package
│   ├── compressor.py           #   LeastSquaresCompressor
│   ├── models.py               #   LinearModel, QuadraticModel (Strategy pattern)
│   ├── metrics.py              #   PSNR, SSIM, MSE
│   ├── io.py                   #   .pfic save/load
│   ├── cli.py                  #   Typer CLI
│   ├── visualization.py        #   Comparison plots, error heatmaps
│   ├── exceptions.py           #   Custom exceptions
│   └── quantum/                #   Optional quantum module
│       ├── vqls.py             #     Variational Quantum Linear Solver
│       ├── feature_maps.py     #     Quantum feature maps
│       ├── hybrid.py           #     HybridCompressor
│       └── utils.py            #     State preparation utilities
├── tests/                      # Test suite (48+ tests)
├── benchmarks/                 # Benchmark runner + dataset download
├── notebooks/                  # Jupyter notebooks (demo + quantum tutorial)
├── demo/                       # Gradio interactive demo
├── docs/                       # MkDocs documentation site
├── DOCS/                       # Project documentation (10 documents)
└── examples/                   # Example gallery generator
```

---

## Example Results

| Method | Block Size | PSNR (dB) | SSIM | Compression Ratio |
|--------|:---------:|:---------:|:----:|:-----------------:|
| Linear | 4x4 | ~28 | 0.89 | 1.3:1 |
| Linear | 8x8 | ~22 | 0.71 | 5.3:1 |
| Linear | 16x16 | ~18 | 0.55 | 21.3:1 |
| Quadratic | 4x4 | ~35 | 0.96 | 0.7:1 |
| Quadratic | 8x8 | ~25 | 0.82 | 2.7:1 |
| Quadratic | 16x16 | ~20 | 0.68 | 10.7:1 |

> JPEG at quality 50 typically achieves ~32-35 dB at ~15:1. This project trades quality for algorithmic simplicity and educational value.

---

## Interactive Demo

Run the Gradio demo locally:

```bash
pip install -e ".[demo]"
python demo/app.py
```

Or try the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jelsin29/polyfit-image-compress/blob/main/notebooks/polyfit_image_compress.ipynb)

---

## Running Benchmarks

```bash
# Download the Kodak dataset
python benchmarks/download_datasets.py --dataset kodak

# Run benchmarks
python benchmarks/run_benchmark.py --dataset benchmarks/datasets/kodak --output benchmarks/results
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=polyfit_compress --cov-report=term-missing

# Lint + format
ruff check --fix src/ tests/
ruff format src/ tests/

# Type checking
mypy src/polyfit_compress/

# Build docs locally
mkdocs serve
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

---

## Documentation

- [Development Guide](DOCS/01-Development-Guide.md) — Project overview and implementation details
- [Architecture](DOCS/02-Architecture.md) — Design patterns, data flow, file format
- [Algorithm Theory](DOCS/03-Algorithm-Theory.md) — Mathematical foundations
- [Quantum Integration](DOCS/04-Quantum-Integration.md) — VQLS, feature maps, honest assessment
- [Developer Workflow](DOCS/05-Developer-Workflow.md) — Conventions, commits, PRs
- [Benchmarks Strategy](DOCS/06-Benchmarks-Strategy.md) — Reproducible benchmark protocol
- [Roadmap](DOCS/07-Roadmap.md) — Project phases and milestones
- [API Reference](docs/) — Auto-generated from docstrings (MkDocs)

---

## License

MIT
