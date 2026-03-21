"""Command-line interface for polyfit-image-compress."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import typer
from numpy.typing import NDArray
from skimage import io as skio

from polyfit_compress.compressor import LeastSquaresCompressor
from polyfit_compress.io import MODEL_TYPE_MAP, PficHeader, load_pfic, save_pfic
from polyfit_compress.metrics import psnr, ssim
from polyfit_compress.models import LinearModel, QuadraticModel

app = typer.Typer(
    name="polyfit",
    help="Image compression via polynomial surface fitting.",
    add_completion=False,
)

MODEL_REGISTRY: dict[str, type[LinearModel | QuadraticModel]] = {
    "linear": LinearModel,
    "quadratic": QuadraticModel,
}


def _load_image(path: Path) -> NDArray[np.uint8]:
    """Load an image and convert RGBA to RGB if needed."""
    image = skio.imread(str(path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def _resolve_model(name: str) -> LinearModel | QuadraticModel:
    """Resolve a model name string to a model instance.

    Parameters
    ----------
    name : str
        Model name (``"linear"`` or ``"quadratic"``).

    Returns
    -------
    LinearModel | QuadraticModel
        The corresponding model instance.

    Raises
    ------
    typer.BadParameter
        If the model name is not recognized.
    """
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise typer.BadParameter(
            f"Unknown model '{name}'. Choose from: {', '.join(MODEL_REGISTRY)}"
        )
    return cls()


@app.command()
def compress(
    input_path: Path = typer.Argument(..., help="Input image path"),
    output_path: Path = typer.Argument(..., help="Output .pfic file path"),
    model: str = typer.Option("quadratic", help="Polynomial model: linear or quadratic"),
    block_size: int = typer.Option(8, help="Block size for compression"),
) -> None:
    """Compress an image to .pfic format."""
    try:
        if not input_path.exists():
            typer.echo(f"Error: Input file not found: {input_path}", err=True)
            raise typer.Exit(code=1)

        image = _load_image(input_path)
        poly_model = _resolve_model(model)
        compressor = LeastSquaresCompressor(poly_model, block_size=block_size)
        result = compressor.compress(image)
        save_pfic(output_path, result)

        original_kb = result.original_size_bytes / 1024
        compressed_kb = result.compressed_size_bytes / 1024
        psnr_val = psnr(image, result.reconstructed)

        typer.echo(f"Compressed: {input_path} -> {output_path}")
        typer.echo(f"  Original size:   {original_kb:.1f} KB")
        typer.echo(f"  Compressed size: {compressed_kb:.1f} KB")
        typer.echo(f"  Ratio:           {result.compression_ratio:.2f}x")
        typer.echo(f"  PSNR:            {psnr_val:.2f} dB")
    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def decompress(
    input_path: Path = typer.Argument(..., help="Input .pfic file path"),
    output_path: Path = typer.Argument(..., help="Output image path"),
) -> None:
    """Decompress a .pfic file to an image."""
    try:
        if not input_path.exists():
            typer.echo(f"Error: Input file not found: {input_path}", err=True)
            raise typer.Exit(code=1)

        header, coefficients = load_pfic(input_path)

        model_cls = MODEL_TYPE_MAP.get(header.model_type)
        if model_cls is None:
            typer.echo(f"Error: Unknown model type: {header.model_type}", err=True)
            raise typer.Exit(code=1)

        poly_model = model_cls()
        compressor = LeastSquaresCompressor(poly_model, block_size=header.block_size)

        original_shape: tuple[int, ...] = (header.height, header.width)
        if header.channels > 1:
            original_shape = (header.height, header.width, header.channels)

        reconstructed = compressor.decompress(
            coefficients,
            original_shape=original_shape,
            padded_shape=(header.padding_height, header.padding_width),
        )

        skio.imsave(str(output_path), reconstructed)
        typer.echo(f"Decompressed: {input_path} -> {output_path}")
    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def info(
    input_path: Path = typer.Argument(..., help="Input .pfic file path"),
) -> None:
    """Display information about a .pfic file."""
    try:
        if not input_path.exists():
            typer.echo(f"Error: Input file not found: {input_path}", err=True)
            raise typer.Exit(code=1)

        raw = input_path.read_bytes()
        header = PficHeader.unpack(raw)

        model_names = {0: "linear", 1: "quadratic"}
        model_name = model_names.get(header.model_type, f"unknown({header.model_type})")

        typer.echo(f"PFIC File: {input_path}")
        typer.echo(f"  Version:    {header.version}")
        typer.echo(f"  Dimensions: {header.width}x{header.height}")
        typer.echo(f"  Channels:   {header.channels}")
        typer.echo(f"  Block size: {header.block_size}")
        typer.echo(f"  Model:      {model_name}")
        typer.echo(f"  Padding:    {header.padding_width}x{header.padding_height}")
    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def benchmark(
    input_path: Path = typer.Argument(..., help="Input image path"),
    model_filter: str | None = typer.Option(  # noqa: UP007
        None, "--model", help="Filter by model: linear or quadratic"
    ),
) -> None:
    """Run compression benchmarks on an image."""
    try:
        if not input_path.exists():
            typer.echo(f"Error: Input file not found: {input_path}", err=True)
            raise typer.Exit(code=1)

        image = _load_image(input_path)

        models_to_test = list(MODEL_REGISTRY.keys())
        if model_filter is not None:
            if model_filter not in MODEL_REGISTRY:
                typer.echo(
                    f"Error: Unknown model '{model_filter}'. "
                    f"Choose from: {', '.join(MODEL_REGISTRY)}",
                    err=True,
                )
                raise typer.Exit(code=1)
            models_to_test = [model_filter]

        block_sizes = [4, 8, 16]

        # Print table header
        typer.echo(
            f"{'Model':<12} {'Block':<6} {'PSNR (dB)':<12} "
            f"{'SSIM':<8} {'Ratio':<8} {'Time (s)':<10}"
        )
        typer.echo("-" * 60)

        for model_name in models_to_test:
            poly_model = MODEL_REGISTRY[model_name]()
            for bs in block_sizes:
                # Skip block sizes larger than image dimensions
                if image.shape[0] < bs or image.shape[1] < bs:
                    continue

                compressor = LeastSquaresCompressor(poly_model, block_size=bs)

                t_start = time.perf_counter()
                result = compressor.compress(image)
                elapsed = time.perf_counter() - t_start

                psnr_val = psnr(image, result.reconstructed)
                ssim_val = ssim(image, result.reconstructed)

                typer.echo(
                    f"{model_name:<12} {bs:<6} {psnr_val:<12.2f} "
                    f"{ssim_val:<8.4f} {result.compression_ratio:<8.2f} {elapsed:<10.3f}"
                )
    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
