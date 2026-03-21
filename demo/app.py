"""Interactive demo for polyfit-image-compress using Gradio."""

from __future__ import annotations

import gradio as gr
import numpy as np
from skimage.color import rgb2gray

from polyfit_compress import LeastSquaresCompressor, LinearModel, QuadraticModel
from polyfit_compress.metrics import mse, psnr, ssim


def compress_image(
    image: np.ndarray,
    model_name: str,
    block_size: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compress an image and return results.

    Returns (reconstructed, error_heatmap, metrics_text)
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Drop alpha channel if RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert to grayscale
    if image.ndim == 3:
        gray = (rgb2gray(image) * 255).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    # Select model
    model = LinearModel() if model_name == "Linear" else QuadraticModel()

    # Compress (cast block_size to int — Gradio Slider returns float)
    try:
        compressor = LeastSquaresCompressor(model=model, block_size=int(block_size))
        result = compressor.compress(gray)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    # Compute metrics
    psnr_val = psnr(gray, result.reconstructed)
    ssim_val = ssim(gray, result.reconstructed)
    mse_val = mse(gray, result.reconstructed)

    # Error map
    error = np.abs(gray.astype(float) - result.reconstructed.astype(float))
    error_normalized = (
        (error / error.max() * 255).astype(np.uint8) if error.max() > 0 else error.astype(np.uint8)
    )

    # Metrics text
    psnr_str = f"{psnr_val:.2f} dB" if psnr_val != float("inf") else "∞ dB (lossless)"
    metrics = (
        f"**PSNR**: {psnr_str}\n"
        f"**SSIM**: {ssim_val:.4f}\n"
        f"**MSE**: {mse_val:.2f}\n"
        f"**Compression Ratio**: {result.compression_ratio:.2f}x\n"
        f"**Original Size**: {result.original_size_bytes / 1024:.1f} KB\n"
        f"**Compressed Size**: {result.compressed_size_bytes / 1024:.1f} KB"
    )

    return result.reconstructed, error_normalized, metrics


# Build Gradio interface
with gr.Blocks(title="polyfit-image-compress") as app:
    gr.Markdown(
        "# polyfit-image-compress\n"
        "Image compression via **Least Squares Polynomial Surface Fitting**\n\n"
        "Upload an image, select a model and block size, then click Compress.\n\n"
        "_Note: images are converted to grayscale for this demo._"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            model_choice = gr.Radio(
                choices=["Linear", "Quadratic"],
                value="Quadratic",
                label="Polynomial Model",
            )
            block_size = gr.Slider(
                minimum=4,
                maximum=32,
                step=4,
                value=8,
                label="Block Size",
            )
            compress_btn = gr.Button("Compress", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Reconstructed")
            error_image = gr.Image(label="Error Heatmap")
            gr.Markdown("### Metrics")
            metrics_output = gr.Markdown()

    compress_btn.click(
        fn=compress_image,
        inputs=[input_image, model_choice, block_size],
        outputs=[output_image, error_image, metrics_output],
    )

if __name__ == "__main__":
    app.launch()
