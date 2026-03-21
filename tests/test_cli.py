"""Tests for CLI commands using typer.testing.CliRunner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io as skio
from typer.testing import CliRunner

from polyfit_compress.cli import app

runner = CliRunner()


def _create_test_image(path: Path, shape: tuple[int, ...] = (32, 32)) -> Path:
    """Create a small test image and save it to disk.

    Parameters
    ----------
    path : Path
        File path where the image will be saved.
    shape : tuple[int, ...]
        Shape of the image array (default: 32x32 grayscale).

    Returns
    -------
    Path
        The path where the image was saved.
    """
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    skio.imsave(str(path), image)
    return path


class TestCompress:
    """Tests for the compress command."""

    def test_compress_creates_pfic_file(self, tmp_path: Path) -> None:
        """Compressing an image should create a .pfic file."""
        img_path = _create_test_image(tmp_path / "test.png")
        pfic_path = tmp_path / "output.pfic"

        result = runner.invoke(
            app,
            [
                "compress",
                str(img_path),
                str(pfic_path),
                "--model",
                "linear",
                "--block-size",
                "8",
            ],
        )

        assert result.exit_code == 0, result.output
        assert pfic_path.exists()
        assert pfic_path.stat().st_size > 0

    def test_compress_invalid_model(self, tmp_path: Path) -> None:
        """Using an invalid model name should exit with error."""
        img_path = _create_test_image(tmp_path / "test.png")
        pfic_path = tmp_path / "output.pfic"

        result = runner.invoke(
            app,
            [
                "compress",
                str(img_path),
                str(pfic_path),
                "--model",
                "cubic",
            ],
        )

        assert result.exit_code != 0

    def test_compress_missing_file(self, tmp_path: Path) -> None:
        """Compressing a non-existent file should exit with error."""
        pfic_path = tmp_path / "output.pfic"

        result = runner.invoke(
            app,
            [
                "compress",
                str(tmp_path / "nonexistent.png"),
                str(pfic_path),
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


class TestDecompress:
    """Tests for the decompress command."""

    def test_decompress_produces_image(self, tmp_path: Path) -> None:
        """Decompressing a .pfic file should produce an image file."""
        img_path = _create_test_image(tmp_path / "test.png")
        pfic_path = tmp_path / "compressed.pfic"
        out_path = tmp_path / "restored.png"

        # First compress
        result = runner.invoke(
            app,
            [
                "compress",
                str(img_path),
                str(pfic_path),
            ],
        )
        assert result.exit_code == 0, result.output

        # Then decompress
        result = runner.invoke(
            app,
            [
                "decompress",
                str(pfic_path),
                str(out_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()

    def test_compress_decompress_roundtrip(self, tmp_path: Path) -> None:
        """A full compress-decompress roundtrip should preserve image shape."""
        img_path = _create_test_image(tmp_path / "test.png")
        pfic_path = tmp_path / "compressed.pfic"
        out_path = tmp_path / "restored.png"

        runner.invoke(app, ["compress", str(img_path), str(pfic_path)])
        runner.invoke(app, ["decompress", str(pfic_path), str(out_path)])

        original = skio.imread(str(img_path))
        restored = skio.imread(str(out_path))

        assert original.shape == restored.shape


class TestInfo:
    """Tests for the info command."""

    def test_info_shows_header(self, tmp_path: Path) -> None:
        """Info command should display header fields from a .pfic file."""
        img_path = _create_test_image(tmp_path / "test.png")
        pfic_path = tmp_path / "compressed.pfic"

        runner.invoke(app, ["compress", str(img_path), str(pfic_path)])

        result = runner.invoke(app, ["info", str(pfic_path)])

        assert result.exit_code == 0, result.output
        output = result.output.lower()
        assert "version" in output
        assert "dimensions" in output
        assert "channels" in output
        assert "block size" in output
        assert "model" in output


class TestBenchmark:
    """Tests for the benchmark command."""

    def test_benchmark_runs(self, tmp_path: Path) -> None:
        """Benchmark should run and produce tabular output."""
        img_path = _create_test_image(tmp_path / "test.png")

        result = runner.invoke(app, ["benchmark", str(img_path)])

        assert result.exit_code == 0, result.output
        output = result.output.lower()
        assert "psnr" in output
        assert "ssim" in output
        assert "linear" in output
        assert "quadratic" in output
