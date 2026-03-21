"""Generate example gallery images comparing compression configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from skimage import data

from polyfit_compress import LeastSquaresCompressor, LinearModel, QuadraticModel
from polyfit_compress.metrics import psnr, ssim
from polyfit_compress.visualization import compare_images, error_heatmap

matplotlib.use("Agg")


def generate_gallery(output_dir: Path) -> None:
    """Generate comparison images for different compression configs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test image
    img = data.camera()  # 512x512 grayscale

    configs = [
        ("linear_4x4", LinearModel(), 4),
        ("linear_8x8", LinearModel(), 8),
        ("linear_16x16", LinearModel(), 16),
        ("quadratic_4x4", QuadraticModel(), 4),
        ("quadratic_8x8", QuadraticModel(), 8),
        ("quadratic_16x16", QuadraticModel(), 16),
    ]

    print(f"Generating gallery in {output_dir}/")
    print(f"{'Config':<20} {'PSNR (dB)':<12} {'SSIM':<10} {'Ratio':<10}")
    print("-" * 55)

    for name, model, block_size in configs:
        compressor = LeastSquaresCompressor(model=model, block_size=block_size)
        result = compressor.compress(img)

        psnr_val = psnr(img, result.reconstructed)
        ssim_val = ssim(img, result.reconstructed)

        print(f"{name:<20} {psnr_val:<12.2f} {ssim_val:<10.4f} {result.compression_ratio:<10.2f}")

        # Save comparison image
        fig = compare_images(
            img,
            result.reconstructed,
            title=f"{name} (PSNR: {psnr_val:.1f} dB)",
        )
        fig.savefig(output_dir / f"comparison_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save error heatmap
        fig = error_heatmap(
            img,
            result.reconstructed,
            title=f"Error: {name}",
        )
        fig.savefig(output_dir / f"error_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nGallery saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example gallery")
    parser.add_argument("--output", default="examples/output", help="Output directory")
    args = parser.parse_args()
    generate_gallery(Path(args.output))
