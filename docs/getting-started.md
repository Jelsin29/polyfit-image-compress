# Getting Started

## Installation

### From source (development mode)

```bash
git clone https://github.com/Jelsin29/polyfit-image-compress.git
cd polyfit-image-compress
pip install -e ".[dev]"
```

### With quantum module (optional)

```bash
pip install -e ".[dev,quantum]"
```

## Quick Start

```python
from polyfit_compress import LeastSquaresCompressor, QuadraticModel
from polyfit_compress.io import save_pfic, load_pfic
from polyfit_compress.metrics import psnr, ssim

# Compress
compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
result = compressor.compress(image)

print(f"Ratio: {result.compression_ratio:.2f}x")
print(f"PSNR: {psnr(image, result.reconstructed):.2f} dB")

# Save to .pfic format
save_pfic("compressed.pfic", result)

# Load and decompress
header, coefficients = load_pfic("compressed.pfic")
reconstructed = compressor.decompress(
    coefficients,
    original_shape=(header.height, header.width),
    padded_shape=(header.padding_height, header.padding_width),
)
```

## CLI Usage

### Compress an image

```bash
polyfit compress input.png output.pfic --model quadratic --block-size 8
```

### Decompress a .pfic file

```bash
polyfit decompress output.pfic restored.png
```

### View file metadata

```bash
polyfit info output.pfic
```

### Run benchmarks

```bash
polyfit benchmark input.png
polyfit benchmark input.png --model linear
```

## Next Steps

Explore the full [API Reference](api/compressor.md) for detailed documentation on each module:

- [Compressor](api/compressor.md) — Core compression engine
- [Models](api/models.md) — Polynomial surface models
- [Metrics](api/metrics.md) — Quality metrics (PSNR, SSIM, MSE)
- [I/O](api/io.md) — .pfic file format reading and writing
- [Visualization](api/visualization.md) — Comparison plots and error heatmaps
- [Quantum](api/quantum.md) — Quantum computing extensions
