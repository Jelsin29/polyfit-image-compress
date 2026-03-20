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

## Features

- Configurable block sizes and polynomial degree
- Vectorized compression/reconstruction using NumPy
- Quality metrics: PSNR, MSE
- Comparison against SVD-based compression
- Error distribution analysis and cross-section visualization
- Handles arbitrary image dimensions via dynamic padding

## Quick Start

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jelsin29/polyfit-image-compress/blob/main/polyfit_image_compress.ipynb)

### Requirements

```
numpy
matplotlib
scipy
scikit-image
```

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
