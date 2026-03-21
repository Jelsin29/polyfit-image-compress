# polyfit-image-compress

Image compression using **Least Squares Polynomial Surface Fitting** — a numerical analysis approach that approximates image blocks with polynomial surfaces instead of traditional frequency-domain transforms.

## Overview

polyfit-image-compress divides an image into non-overlapping blocks and models each block as a polynomial surface. Coefficients are computed via the **Moore-Penrose Pseudoinverse**, and the image is reconstructed by evaluating the polynomial at each pixel coordinate.

### Supported Models

| Model | Terms | Coefficients per block |
|-------|-------|----------------------|
| **Linear** (plane) | `1, x, y` | 3 |
| **Quadratic** (paraboloid) | `1, x, y, xy, x^2, y^2` | 6 |

## Features

- Configurable block sizes and polynomial degree
- Vectorized compression/reconstruction using NumPy
- `.pfic` binary file format for saving/loading compressed data
- CLI for compress, decompress, info, and benchmark operations
- Quality metrics: PSNR, SSIM, MSE
- Visualization: side-by-side comparison plots, error heatmaps
- RGB and grayscale support
- Handles arbitrary image dimensions via dynamic padding

## Example Results

| Method | Block Size | PSNR | Compression Ratio |
|--------|-----------|------|-------------------|
| Linear | 8x8 | ~22 dB | 5.3:1 |
| Quadratic | 8x8 | ~25 dB | 2.7:1 |
| Quadratic | 16x16 | ~20 dB | 10.5:1 |

## Theory

The core idea is solving an **overdetermined system** for each image block. With `N^2` pixels and `p` polynomial terms (`p << N^2`), the system is overdetermined. The least squares solution minimizes the sum of squared residuals, effectively fitting the best polynomial surface through the pixel intensities.

The **Design Matrix** `A` is precomputed once and reused for all blocks, making the approach efficient. The pseudoinverse `A+` transforms the problem into a simple matrix multiplication.

## Quick Links

- [Getting Started](getting-started.md) — Installation and first steps
- [API Reference](api/compressor.md) — Full module documentation
- [Interactive Notebook](https://colab.research.google.com/github/Jelsin29/polyfit-image-compress/blob/main/notebooks/polyfit_image_compress.ipynb) — Try it in Google Colab
