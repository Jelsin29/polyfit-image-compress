# Compressing Images with Polynomial Surfaces -- And a Quantum Twist

## What if you could compress an image using nothing but polynomials?

Most image compression algorithms -- JPEG, WebP, AVIF -- rely on frequency-domain transforms like the DCT or wavelets. They work brilliantly, but they hide the geometry of the image behind abstract coefficients. What if, instead, you approximated each region of an image with a smooth mathematical surface?

That is exactly what **polyfit-image-compress** does. It splits an image into small blocks and fits a polynomial surface (a plane or a paraboloid) to each one using least squares. The result is a compressed representation that is conceptually transparent: every block is described by a handful of polynomial coefficients, and reconstruction is just evaluating those polynomials on a grid.

This is not meant to replace JPEG. It is meant to *teach* -- to make compression tangible by connecting it to linear algebra concepts you already know.

## The Math Behind It

### Block Decomposition

The image is divided into non-overlapping NxN blocks (typically 4x4, 8x8, or 16x16). Each block is treated as a discrete surface z(x, y) where z is pixel intensity and (x, y) are local coordinates.

### The Design Matrix

For a quadratic model, we approximate each block as:

```
z(x, y) = c0 + c1*x + c2*y + c3*x*y + c4*x^2 + c5*y^2
```

This gives us an overdetermined system **Ac = b**, where:

- **A** is the design matrix (N^2 rows x 6 columns for quadratic, 3 for linear)
- **c** is the coefficient vector we want to find
- **b** is the flattened pixel values of the block

### The Pseudoinverse

The least squares solution is:

```
c = A+ * b = (A^T A)^(-1) A^T * b
```

The key insight is that **A depends only on block size and model type, not on pixel values**. So the pseudoinverse A+ is precomputed once and reused for every block. Compression becomes a single matrix multiplication per block -- no iterative optimization, no convergence issues.

### Vectorized Processing

Rather than looping over blocks, all blocks are stacked into a single matrix and solved simultaneously. This makes the implementation fast despite being pure NumPy/SciPy.

## How It Compares

| Method | Type | Artifacts | Strength |
|--------|------|-----------|----------|
| **Polynomial (ours)** | Local surface fitting | Blocking at boundaries | Transparent math, educational |
| **JPEG** | Block DCT + quantization | Blocking + mosquito noise | Universal, hardware support |
| **WebP** | Prediction + transforms | Smooth blurring | Better compression than JPEG |
| **SVD** | Global matrix decomposition | Smooth blurring | Optimal for global MSE |

### Honest Assessment

At equivalent file sizes, polynomial surface fitting typically achieves 20-30 dB PSNR on natural images. This is lower than JPEG (which hits 30-40 dB at similar ratios) because polynomials cannot capture high-frequency texture within a block -- they smooth it out.

Where polynomial compression shines:

- **Pedagogical value**: every step maps to a linear algebra concept
- **Smooth gradients**: sky, skin, and background regions compress beautifully
- **Predictable behavior**: quality degrades gracefully with larger block sizes
- **No quantization tables**: the compression ratio is determined entirely by block size and model order

Where it struggles:

- **Edges and texture**: sharp transitions cause visible blocking artifacts
- **Compression ratio**: 3-10x is typical, far below JPEG's 10-50x
- **No entropy coding**: coefficients are stored as raw floats (future improvement)

## The Quantum Module

The project includes an experimental quantum computing module that solves the least squares system using the Variational Quantum Linear Solver (VQLS) algorithm.

### What VQLS Does

Instead of computing the classical pseudoinverse, VQLS encodes the linear system into a quantum circuit and uses variational optimization to find the coefficient vector. The `VQLSSolver` class handles circuit construction, cost function evaluation, and coefficient extraction.

### HybridCompressor

The `HybridCompressor` class provides a practical interface: it attempts quantum solving for each block but falls back to classical least squares if the quantum solver fails to converge or if Qiskit/PennyLane are not installed. This makes the quantum path entirely optional -- the package works without any quantum dependencies.

### Why It Matters (Educationally)

This is not practical quantum advantage. Current quantum simulators are orders of magnitude slower than `numpy.linalg.pinv` for these small systems. But it demonstrates:

- How to formulate a real-world problem for a quantum computer
- The VQLS algorithm in a concrete, visual context
- The gap between quantum theory and practical utility

## Try It Yourself

### Install

```bash
pip install polyfit-image-compress
```

### CLI

```bash
# Compress an image
polyfit compress input.png output.pfic --model quadratic --block-size 8

# Decompress
polyfit decompress output.pfic restored.png

# Get metrics
polyfit info output.pfic
```

### Python API

```python
from polyfit_compress import LeastSquaresCompressor, QuadraticModel
from polyfit_compress.metrics import psnr, ssim

compressor = LeastSquaresCompressor(model=QuadraticModel(), block_size=8)
result = compressor.compress(image)

print(f"PSNR: {psnr(image, result.reconstructed):.2f} dB")
print(f"SSIM: {ssim(image, result.reconstructed):.4f}")
print(f"Compression ratio: {result.compression_ratio:.1f}:1")
```

### Generate a Gallery

```bash
python examples/example_gallery.py --output gallery/
```

## What's Next

The current implementation stores raw float32 coefficients. Several improvements are on the roadmap:

- **Coefficient quantization**: reduce 32-bit floats to 8-16 bits with minimal quality loss
- **Entropy coding**: Huffman or arithmetic coding on quantized coefficients
- **Adaptive block sizes**: use smaller blocks near edges, larger blocks in smooth regions
- **Higher-order models**: cubic or bicubic surfaces for better edge preservation
- **Color-aware compression**: exploit YCbCr chroma subsampling

## Links

- **GitHub**: [github.com/Jelsin29/polyfit-image-compress](https://github.com/Jelsin29/polyfit-image-compress)
- **Docs**: [jelsin29.github.io/polyfit-image-compress/](https://jelsin29.github.io/polyfit-image-compress/)
- **PyPI**: [pypi.org/project/polyfit-image-compress/](https://pypi.org/project/polyfit-image-compress/)
