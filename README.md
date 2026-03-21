# polyfit-image-compress

Image compression using **Least Squares Polynomial Surface Fitting** — a numerical analysis approach that approximates image blocks with polynomial surfaces instead of traditional frequency-domain transforms.

## How It Works

1. The image is divided into non-overlapping blocks (e.g., 8x8 or 16x16 pixels)
2. Each block is modeled as a polynomial surface using an overdetermined system **Ax = b**
3. Coefficients are computed via the **Moore-Penrose Pseudoinverse**: `c = (A^T A)^{-1} A^T b`
4. The image is reconstructed by evaluating the polynomial at each pixel coordinate

### Supported Models

| Model | Terms | Coefficients per block |
|-------|-------|----------------------|
| **Linear** (plane) | `1, x, y` | 3 |
| **Quadratic** (paraboloid) | `1, x, y, xy, x^2, y^2` | 6 |

## Installation

```bash
# From source (development mode)
git clone https://github.com/Jelsin29/polyfit-image-compress.git
cd polyfit-image-compress
pip install -e ".[dev]"

# With quantum module (optional)
pip install -e ".[dev,quantum]"
```

## CLI Usage

```bash
# Compress an image to .pfic format
polyfit compress input.png output.pfic --model quadratic --block-size 8

# Decompress a .pfic file back to an image
polyfit decompress output.pfic restored.png

# View .pfic file metadata
polyfit info output.pfic

# Run benchmarks on an image
polyfit benchmark input.png
polyfit benchmark input.png --model linear
```

## Python API

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

## Features

- Configurable block sizes and polynomial degree
- Vectorized compression/reconstruction using NumPy
- `.pfic` binary file format for saving/loading compressed data
- CLI for compress, decompress, info, and benchmark operations
- Quality metrics: PSNR, SSIM, MSE
- Visualization: side-by-side comparison plots, error heatmaps
- RGB and grayscale support
- Handles arbitrary image dimensions via dynamic padding

## Interactive Notebook

The original interactive demo is available as a Jupyter notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jelsin29/polyfit-image-compress/blob/main/notebooks/polyfit_image_compress.ipynb)

## Example Results

| Method | Block Size | PSNR | Compression Ratio |
|--------|-----------|------|-------------------|
| Linear | 8x8 | ~22 dB | 5.3:1 |
| Quadratic | 8x8 | ~25 dB | 2.7:1 |
| Quadratic | 16x16 | ~20 dB | 10.5:1 |

## Theory

The core idea is solving an **overdetermined system** for each image block. With `N^2` pixels and `p` polynomial terms (`p << N^2`), the system is overdetermined. The least squares solution minimizes the sum of squared residuals, effectively fitting the best polynomial surface through the pixel intensities.

The **Design Matrix** `A` is precomputed once and reused for all blocks, making the approach efficient. The pseudoinverse `A+` transforms the problem into a simple matrix multiplication.

## License

MIT
