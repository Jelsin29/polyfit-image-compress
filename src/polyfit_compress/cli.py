"""Command-line interface for polyfit-image-compress."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="polyfit",
    help="Image compression via polynomial surface fitting.",
    add_completion=False,
)


@app.command()
def compress(
    input_path: Path = typer.Argument(..., help="Input image path"),
    output_path: Path = typer.Argument(..., help="Output .pfic file path"),
    model: str = typer.Option("quadratic", help="Polynomial model: linear or quadratic"),
    block_size: int = typer.Option(8, help="Block size for compression"),
) -> None:
    """Compress an image to .pfic format."""
    raise NotImplementedError("Stub — implement in phase2/cli")


@app.command()
def decompress(
    input_path: Path = typer.Argument(..., help="Input .pfic file path"),
    output_path: Path = typer.Argument(..., help="Output image path"),
) -> None:
    """Decompress a .pfic file to an image."""
    raise NotImplementedError("Stub — implement in phase2/cli")


@app.command()
def info(
    input_path: Path = typer.Argument(..., help="Input .pfic file path"),
) -> None:
    """Display information about a .pfic file."""
    raise NotImplementedError("Stub — implement in phase2/cli")


@app.command()
def benchmark(
    input_path: Path = typer.Argument(..., help="Input image path"),
    model: str | None = typer.Option(None, help="Filter by model: linear or quadratic"),
) -> None:
    """Run compression benchmarks on an image."""
    raise NotImplementedError("Stub — implement in phase2/cli")
